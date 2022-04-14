from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import GPUEwaldSplitter, EwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson, BackwardEuler
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from FileIO import prepareOutFile, writeArray
import numpy as np
import time, sys
from math import pi

"""
"""

def saveCurvaturesAndStrains(allFibers,CLNet,OutputFileName,wora='a'):
    Xf = allFibers.getX();
    LinkStrains = CLNet.calcLinkStrains(allFibers.getUniformPoints(Xf), Dom);
    LinkStrainSqu = np.sum(LinkStrains**2);
    FibCurvatures = allFibers.calcCurvatures(Xf);
    writeArray('DynamicRheo/LinkStrains'+OutputFileName,[LinkStrainSqu],wora=wora)
    writeArray('DynamicRheo/FibCurves'+OutputFileName,FibCurvatures,wora=wora)

try:
	Input = open('StrainInputFile.txt','r')
	for iLine in Input:
		exec(iLine);
	InputCopyName='DynamicRheo/InputFile_'+outFile;
	copyInput = open(InputCopyName,'w')
	Input = open('StrainInputFile.txt','r')
	for iLine in Input:
		copyInput.write(iLine);
	copyInput.write('COMMAND LINE ARGS \n')
	copyInput.write('CLseed = '+str(CLseed)+'\n')
	copyInput.write('Omega = '+str(omegasToDo[0])+'\n')
	copyInput.write('Strain = '+str(maxStrain)+'\n')
	copyInput.close();
	Input.close()
except:
	raise ValueError('This file takes three command line arguments: the seed (for CL network),\
the frequency omega to shear at, and the maximum strain (on [0,1]), in that order')


# Array of frequencies in Hz 
for omHz in omegasToDo:
    omega = 2*pi*omHz;
    # Set up simulation variables 
    gam0 = maxStrain*omega    # strain amplitude
    T = 1/omHz; # one period
    nCyc = tf*omHz;
    stressdt = T/numStressPerCycle; # 50 times per cycle
    saveEvery = int(savedt/dt+1e-10);  
    stressEvery = int(stressdt/dt+1e-10);  
    print('Omega %f: stopping time %f, saveEvery %f, stressEvery %f' %(omHz,tf,saveEvery*dt,stressEvery*dt))

    # Initialize the domain and spatial database
    Dom = PeriodicShearedDomain(Ld,Ld,Ld);

    # Initialize fiber discretization
    fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=0.1,nptsUniform=Nuniformsites,NupsampleForDirect=NupsampleForDirect);

    # Initialize the master list of fibers
    allFibers = fiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,nThreads=nThr);

    # Initialize Ewald for non-local velocities
    if (nonLocal > 0 and nonLocal < 4):
        Ewald = GPUEwaldSplitter(allFibers.getaRPY()*eps*Lf,mu,xi*1.4*(fibDisc.getNumDirect()/N)**(1/3),Dom,NupsampleForDirect*nFib);
    else: # This is so it doesn't try to initialize a GPU code if there isn't a GPU
        Ewald = EwaldSplitter(allFibers.getaRPY()*eps*Lf,mu,xi*1.4*(fibDisc.getNumDirect()/N)**(1/3),Dom,NupsampleForDirect*nFib);

    # Initialize the fiber list (straight fibers)
    fibList = [None]*nFib;
    print('Loading from steady state')
    try:
        XFile = 'DynamicStress/DynamicSSLocs'+inFile;
        XsFile = 'DynamicStress/DynamicSSTanVecs'+inFile;
        allFibers.initFibList(fibList,Dom,pointsfileName=XFile,tanvecfileName=XsFile);
    except:
        print('Redirecting input file directory')
        XFile = 'DynamicRheo/Time60Locs'+inFile;
        XsFile = 'DynamicRheo/Time60TanVecs'+inFile
        allFibers.initFibList(fibList,Dom,pointsfileName=XFile,tanvecfileName=XsFile);
        
    allFibers.fillPointArrays();

    # Initialize the network of cross linkers
    # New seed for CLs
    np.random.seed(CLseed);
    CLNet = DoubleEndedCrossLinkedNetwork(nFib,N,fibDisc.getNumUniform(),Lf,Kspring,\
        rl,konCL,koffCL,konSecond,koffSecond,CLseed,Dom,fibDisc,nThreads=nThr);
    print('Name of the input file: '+ inFile)
    try:
        CLNet.setLinksFromFile('DynamicStress/DynamicSSLinks'+inFile,'DynamicStress/DynSSFreeLinkBound'+inFile);
    except:
        print('Redirecting CL infile')
        CLNet.setLinksFromFile('DynamicRheo/Time60Links'+inFile,'DynamicRheo/Time60FreeLinkBound'+inFile);
        
    print('Number of links initially %d' %CLNet._nDoubleBoundLinks)
    
    # Initialize the temporal integrator
    if (useBackwardEuler):
        TIntegrator = BackwardEuler(allFibers, CLNet);
    else:
        TIntegrator = CrankNicolson(allFibers, CLNet);
    TIntegrator.setMaxIters(nIts);
    TIntegrator.setLargeTol(LargeTol);

    # Prepare the output file and write initial locations
    of = prepareOutFile('DynamicRheo/Locs'+outFile);
    allFibers.writeFiberLocations(of);
    saveCurvaturesAndStrains(allFibers,CLNet,outFile,wora='w')
    ofCL = prepareOutFile('DynamicRheo/Step'+str(0)+'Links'+outFile);
    CLNet.writeLinks(ofCL)
    ofCL.close()

    # Run to steady state (no flow)
    stopcount = int(tf/dt+1e-10);
    numSaves = stopcount//saveEvery+1;
    numStress = stopcount//stressEvery;
    Lamstress = np.zeros(numStress); 
    Elstress = np.zeros(numStress); 
    CLstress = np.zeros(numStress);

    NumFibsConnected =  np.zeros((numSaves,nFib),dtype=np.int64)
    labels =  np.zeros((numSaves,nFib),dtype=np.int64)
    AllLocalAlignment =  np.zeros((numSaves,nFib))
    AvgTangentVectors = np.zeros((numSaves*nFib,3));
    AvgTangentVectors[:nFib,:] = allFibers.averageTangentVectors()

    numLinksByFib = np.zeros((numSaves,nFib),dtype=np.int64);
    numLinksByFib[0,:] = CLNet.numLinksOnEachFiber();

    numBundlesSep =  np.zeros(numSaves,dtype=np.int64)
    numBundlesSep[0], labels[0,:] = CLNet.FindBundles(bunddist);
    AllOrders_Sep, NPerBundleAll_Sep,AllaverageBundleTangents = CLNet.BundleOrderParameters(allFibers,numBundlesSep[0], labels[0,:],minPerBundle=2)
    numBundlesSep[0] = len(AllOrders_Sep);
    avgBundleAlignment_Sep = np.zeros(numSaves)
    avgBundleAlignment_Sep[0] = CLNet.avgBundleAlignment(AllOrders_Sep,NPerBundleAll_Sep);

    LocalAlignment,numCloseBy = CLNet.LocalOrientations(1,allFibers)
    NumFibsConnected[0,:] = numCloseBy;
    AllLocalAlignment[0,:] = LocalAlignment;
    
    if (nonLocal == 3):
        writeArray('DynamicRheo/ItsNeeded'+outFile,[-1],wora='w')
    for iT in range(stopcount): 
        wr=0;
        if ((iT % saveEvery) == (saveEvery-1)):
            print('Time %1.2E' %(float(iT)*dt));
            wr=1;
        mythist = time.time()
        doStress = False;
        if ((iT % stressEvery) == (stressEvery-1)):
            doStress = True;
        maxX, itsneeded, stressArray = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,\
            outfile=of,write=wr,updateNet=updateNet,turnoverFibs=turnover,stress=doStress); 
        if (nonLocal == 3):
            writeArray('DynamicRheo/ItsNeeded'+outFile,[itsneeded],wora='a')
        #print('MAIN Time step time %f ' %(time.time()-mythist));
        #print('Max x: %f' %(maxX));
        if ((iT % stressEvery) == (stressEvery-1)):
            stressIndex = (iT+1)//stressEvery-1;
            Lamstress[stressIndex]= stressArray[0];
            Elstress[stressIndex] = stressArray[1];
            CLstress[stressIndex] = stressArray[2];
            #print('Max x (just to check stability): %f' %(maxX));
        if (wr==1): # save curvatures and strains
            print('MAIN Time step time %f ' %(time.time()-mythist));
            print('Max x: %f' %maxX)
            thist = time.time();
            print('Number of links %d' %CLNet._nDoubleBoundLinks)
            saveCurvaturesAndStrains(allFibers,CLNet,outFile);
            saveIndex = (iT+1)//saveEvery;
            numLinksByFib[saveIndex,:] = CLNet.numLinksOnEachFiber();
            ofCL = prepareOutFile('DynamicRheo/Step'+str(saveIndex)+'Links'+outFile);
            CLNet.writeLinks(ofCL)
            ofCL.close()
                   
            # Bundles where connections are 2 links separated by 2*restlen
            numBundlesSep[saveIndex], labels[saveIndex,:] = CLNet.FindBundles(bunddist);
            print('Number bundles (1 per bundle possible) %d' %numBundlesSep[saveIndex])
            Orders, NPerBundle, avgTangents = CLNet.BundleOrderParameters(allFibers,numBundlesSep[saveIndex], labels[saveIndex,:],minPerBundle=2)
            numBundlesSep[saveIndex] = len(Orders);
            print('Number bundles (excluding 1 per bundle) %d' %numBundlesSep[saveIndex])
            NPerBundleAll_Sep = np.concatenate((NPerBundleAll_Sep,NPerBundle));
            AllOrders_Sep = np.concatenate((AllOrders_Sep,Orders));
            AllaverageBundleTangents = np.concatenate((AllaverageBundleTangents,avgTangents));
            avgBundleAlignment_Sep[saveIndex] = CLNet.avgBundleAlignment(Orders,NPerBundle);
            
            # Local alignment stats
            LocalAlignment,numCloseBy = CLNet.LocalOrientations(1,allFibers)
            NumFibsConnected[saveIndex,:] = numCloseBy;
            AllLocalAlignment[saveIndex,:] = LocalAlignment;
            AvgTangentVectors[nFib*saveIndex:nFib*(saveIndex+1),:] = allFibers.averageTangentVectors()
      
    np.savetxt('DynamicRheo/nLinksPerFib'+outFile,numLinksByFib);  
    np.savetxt('DynamicRheo/NumFibsConnectedPerFib'+outFile,NumFibsConnected);
    np.savetxt('DynamicRheo/LocalAlignmentPerFib'+outFile,AllLocalAlignment);
    np.savetxt('DynamicRheo/NumberOfBundles'+outFile,numBundlesSep);    
    np.savetxt('DynamicRheo/BundleOrderParams'+outFile,AllOrders_Sep);
    np.savetxt('DynamicRheo/NFibsPerBundle'+outFile,NPerBundleAll_Sep);
    np.savetxt('DynamicRheo/FinalLabels'+outFile,labels);
    np.savetxt('DynamicRheo/AvgBundleTangents'+outFile,AllaverageBundleTangents);
    np.savetxt('DynamicRheo/AvgTangentVectors'+outFile,AvgTangentVectors);

    np.savetxt('DynamicRheo/LamStress'+outFile,Lamstress);
    np.savetxt('DynamicRheo/ElStress'+outFile,Elstress);
    np.savetxt('DynamicRheo/CLStress'+outFile,CLstress);
    
    if (saveFinalState):
        # Write the state to file for restart
        np.savetxt('DynamicRheo/Time'+str(tf)+'Locs'+outFile,allFibers._ptsCheb);
        np.savetxt('DynamicRheo/Time'+str(tf)+'TanVecs'+outFile,allFibers._tanvecs);
        ofCL = prepareOutFile('DynamicRheo/Time'+str(tf)+'Links'+outFile);
        CLNet.writeLinks(ofCL)
        np.savetxt('DynamicRheo/Time'+str(tf)+'FreeLinkBound'+outFile, CLNet._FreeLinkBound);

    # Destruction and cleanup
    of.close();
    del Dom;
    del Ewald;
    del fibDisc;
    del allFibers;
    del TIntegrator; 
    del CLNet;



