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

# Array of frequencies in Hz 
for omHz in omegasToDo:
    omega = 2*pi*omHz;
    if (omHz > 0):
        T = 1/omHz; # one period
        gam0 = maxStrain*omega    # strain amplitude 
    else:
        gam0 = maxStrain;
        T = 4;     
    # Set up simulation variables  
    stressEvery = 1; 
    saveEvery = int(savedt/dt+1e-10);  
    print('Omega %f: stopping time %f, saveEvery %f, stressEvery %f' %(omHz,tf,saveEvery*dt,stressEvery*dt))      
    # Run to steady state (no flow)
    of = prepareOutFile('DynamicRheo/Locs'+outFile);
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
    numLinksByFib = np.zeros((numSaves,nFib),dtype=np.int64);
    numBundlesSep =  np.zeros(numSaves,dtype=np.int64)
    avgBundleAlignment_Sep = np.zeros(numSaves)

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
    XFile = 'DynamicStress/DynamicSSLocs'+inFile;
    XsFile = 'DynamicStress/DynamicSSTanVecs'+inFile
    fibList = [None]*nFib;
    allFibers.initFibList(fibList,Dom,pointsfileName=XFile,tanvecfileName=XsFile);
        
    allFibers.fillPointArrays();

    # Initialize the network of cross linkers
    # New seed for CLs
    CLNet = DoubleEndedCrossLinkedNetwork(nFib,N,fibDisc.getNumUniform(),Lf,Kspring,\
        rl,konCL,koffCL,konSecond,koffSecond,CLseed,Dom,fibDisc,nThreads=nThr);
    print('Name of the input file: '+ inFile)
    CLNet.setLinksFromFile('DynamicStress/DynamicSSLinks'+inFile,'DynamicStress/DynSSFreeLinkBound'+inFile);
        
    print('Number of links initially %d' %CLNet._nDoubleBoundLinks)
    
    # Initialize the temporal integrator
    if (useBackwardEuler):
        TIntegrator = BackwardEuler(allFibers, CLNet);
    else:
        TIntegrator = CrankNicolson(allFibers, CLNet);
    TIntegrator.setMaxIters(nIts);
    TIntegrator.setLargeTol(LargeTol);

    # Prepare the output file and write initial locations
    
    allFibers.writeFiberLocations(of);
    ofCL = prepareOutFile('DynamicRheo/Step'+str(0)+'Links'+outFile);
    CLNet.writeLinks(ofCL)
    ofCL.close()
    saveCurvaturesAndStrains(allFibers,CLNet,outFile,wora='w')
    
    saveIndex = 0;
    numLinksByFib[saveIndex,:] = CLNet.numLinksOnEachFiber();
    AvgTangentVectors[nFib*saveIndex:nFib*(saveIndex+1),:] = allFibers.averageTangentVectors()
    numBundlesSep[saveIndex], labels[saveIndex,:] = CLNet.FindBundles(Lf/4);
    AllOrders_Sep, NPerBundleAll_Sep,AllaverageBundleTangents = CLNet.BundleOrderParameters(allFibers,\
        numBundlesSep[saveIndex], labels[saveIndex,:],minPerBundle=2)
    numBundlesSep[saveIndex] = len(AllOrders_Sep);
    avgBundleAlignment_Sep[saveIndex] = CLNet.avgBundleAlignment(AllOrders_Sep,NPerBundleAll_Sep);
    LocalAlignment,numCloseBy = CLNet.LocalOrientations(1,allFibers)
    NumFibsConnected[saveIndex,:] = numCloseBy;
    AllLocalAlignment[saveIndex,:] = LocalAlignment;
    
    for iT in range(stopcount): 
        wr=0;
        if ((iT % saveEvery) == (saveEvery-1)):
            print('Time %1.2E' %(float(iT)*dt));
            wr=1;
        fixedg = None;
        if (iT*dt > T/4):
            fixedg = maxStrain;
            allFibers._gam0=0; 
        mythist = time.time()
        maxX, _, stressArray= TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,outfile=of,write=wr,\
            updateNet=updateNet,turnoverFibs=turnover,fixedg=fixedg,stress=True);
        #print('MAIN Time step time %f ' %(time.time()-mythist));
        #print('Max x: %f' %(maxX));
        #print('Strain before stress %f' %Dom._g)
        stressIndex =iT;
        Lamstress[stressIndex]= stressArray[0];
        Elstress[stressIndex] = stressArray[1];
        CLstress[stressIndex] = stressArray[2];
        #print('After stress %f' %Dom._g)
        #print('Max x (just to check stability): %f' %(maxX));
        if (wr==1 and False): # save curvatures and strains
            print('MAIN Time step time %f ' %(time.time()-mythist));
            print('Max x: %f' %maxX)
            thist = time.time();
            print('Number of links %d' %CLNet._nDoubleBoundLinks)
            saveCurvaturesAndStrains(allFibers,CLNet,outFile);
            saveIndex = (iT+1)//saveEvery;
            numLinksByFib[saveIndex,:] = CLNet.numLinksOnEachFiber();
            #ofCL = prepareOutFile('DynamicRheo/Step'+str(saveIndex)+'Links'+outFile);
            CLNet.writeLinks(ofCL)
            ofCL.close()
                   
            # Bundles where connections are 2 links separated by 2*restlen
            numBundlesSep[saveIndex], labels[saveIndex,:] = CLNet.FindBundles(Lf/4);
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
    
    if (False):        
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

    del Dom;
    del Ewald;
    del fibDisc;
    del allFibers;
    del TIntegrator; 
    del CLNet;

