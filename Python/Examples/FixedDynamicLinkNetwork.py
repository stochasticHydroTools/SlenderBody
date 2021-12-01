from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import GPUEwaldSplitter, EwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson, BackwardEuler
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from FileIO import prepareOutFile, writeArray
import numpy as np
import time, sys

"""
"""

def saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,OutputFileName,wora='a'):
    Xf = allFibers.getX();
    LinkStrains = CLNet.calcLinkStrains(allFibers.getUniformPoints(Xf), Dom);
    LinkStrainSqu = np.sum(LinkStrains**2);
    FibCurvatures = allFibers.calcCurvatures(Xf);
    writeArray('BundlingBehavior/LinkStrains'+OutputFileName,[LinkStrainSqu],wora=wora)
    writeArray('BundlingBehavior/FibCurvesF'+OutputFileName,FibCurvatures,wora=wora)

# Inputs for the slender body simulation

# Fiber parameters
Input = open('FixedNetworkInputFile.txt','r')
for iLine in Input:
    exec(iLine);
InputCopyName='BundlingBehavior/FixedNetInputFile_'+OutputFileName;
copyInput = open(InputCopyName,'w')
Input = open('FixedNetworkInputFile.txt','r')
for iLine in Input:
    copyInput.write(iLine);
copyInput.write('COMMAND LINE ARGS \n')
copyInput.write('CLseed = '+str(CLseed)+'\n')
copyInput.close();
Input.close()
saveEvery = int(savedt/dt+1e-10);  

# Initialize the domain and spatial database
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=0.1,nptsUniform=Nuniformsites,NupsampleForDirect=NupsampleForDirect,rigid=rigidFibs);

# Initialize the master list of fibers
allFibers = fiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,1,0,Dom,nThreads=nThr); # No flow, no nonlocal hydro (nothing to OMP parallelize)

# Initialize Ewald for non-local velocities
if (nonLocal==0 or nonLocal == 4):
    Ewald = EwaldSplitter(allFibers.getaRPY()*eps*Lf,mu,xi*1.4*(fibDisc.getNumDirect()/N)**(1/3),Dom,NupsampleForDirect*nFib);
else: 
    Ewald = GPUEwaldSplitter(allFibers.getaRPY()*eps*Lf,mu,xi*1.4*(fibDisc.getNumDirect()/N)**(1/3),Dom,NupsampleForDirect*nFib);

# Initialize the fiber list (straight fibers)
np.random.seed(CLseed);
fibList = [None]*nFib;
if (initFile is None):
    print('Initializing fresh')
    allFibers.initFibList(fibList,Dom);
else: 
    print('Initializing from file '+initFile);
    XFile = 'DynamicStress/DynamicSSLocs'+initFile;
    XsFile ='DynamicStress/DynamicSSTanVecs'+initFile;
    allFibers.initFibList(fibList,Dom,pointsfileName=XFile,tanvecfileName=XsFile);

# Initialize the network of cross linkers
# New seed for CLs
np.random.seed(CLseed);
# Initialize network
print('Number uniform sites %d' %fibDisc.getNumUniform())
if (BrownianFluct):
    kBt = 4e-3;
else:
    kBt = 0;
CLNet = DoubleEndedCrossLinkedNetwork(nFib,N,fibDisc.getNumUniform(),Lf,Kspring,\
    rl,konCL,koffCL,konSecond,koffSecond,CLseed,Dom,fibDisc,nThreads=nThr,\
    bindingSiteWidth=bindingSiteWidth,kT=kBt);
#Pairs, _ = CLNet.getPairsThatCanBind(allFibers,Dom);
#np.savetxt('PairsN'+str(fibDisc.getNumUniform())+'.txt',Pairs);
if (initFile is None):
    CLNet.updateNetwork(allFibers,Dom,100.0/min(konCL*Lf,konSecond*Lf,koffCL,koffSecond)) # just to load up CLs
else:
    CLNet.setLinksFromFile('DynamicStress/DynamicSSLinks'+initFile,'DynamicStress/DynSSFreeLinkBound'+initFile);
print('Number of links initially %d' %CLNet._nDoubleBoundLinks)
        
# Initialize the temporal integrator
if (BrownianFluct or useBackwardEuler):
    TIntegrator = BackwardEuler(allFibers, CLNet);
else:  
    TIntegrator = CrankNicolson(allFibers, CLNet);
TIntegrator.setMaxIters(nIts);
TIntegrator.setLargeTol(LargeTol);

# Prepare the output file and write initial network information
of = prepareOutFile('BundlingBehavior/Locs'+OutputFileName);
allFibers.writeFiberLocations(of);
saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,OutputFileName,wora='w')
ofCL = prepareOutFile('BundlingBehavior/Step'+str(0)+'Links'+OutputFileName);
CLNet.writeLinks(ofCL)
ofCL.close()

stopcount = int(tf/dt+1e-10);
numSaves = stopcount//saveEvery+1;

NumFibsConnected =  np.zeros((numSaves,nFib),dtype=np.int64)
AllLocalAlignment =  np.zeros((numSaves,nFib))
AllLabels = np.zeros((numSaves,nFib),dtype=np.int64);

numLinksByFib = np.zeros((numSaves,nFib),dtype=np.int64);
numLinksByFib[0,:] = CLNet.numLinksOnEachFiber();

numBundlesSep =  np.zeros(numSaves,dtype=np.int64)
numBundlesSep[0], AllLabels[0,:] = CLNet.FindBundles(bunddist);
AllOrders_Sep, NPerBundleAll_Sep, AllaverageBundleTangents = CLNet.BundleOrderParameters(allFibers,numBundlesSep[0], AllLabels[0,:],minPerBundle=2)
numBundlesSep[0] = len(AllOrders_Sep);
avgBundleAlignment_Sep = np.zeros(numSaves)
avgBundleAlignment_Sep[0] = CLNet.avgBundleAlignment(AllOrders_Sep,NPerBundleAll_Sep);

LocalAlignment,numCloseBy = CLNet.LocalOrientations(1,allFibers)
NumFibsConnected[0,:] = numCloseBy;
AllLocalAlignment[0,:] = LocalAlignment;


# Simulate 
for iT in range(stopcount): 
    wr=0;
    if ((iT % saveEvery) == (saveEvery-1)):
        print('Time %1.2E' %(float(iT)*dt));
        wr=1;
        mythist = time.time()
    maxX, _, _ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,outfile=of,write=wr,\
        updateNet=updateNet,turnoverFibs=turnover,BrownianUpdate=BrownianFluct,kBt=kBt);
    if (wr==1):
        print('MAIN Time step time %f ' %(time.time()-mythist));
        print('Max x: %f' %maxX)
        thist = time.time();
        print('Number of links %d' %CLNet._nDoubleBoundLinks)
        saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,OutputFileName);
        saveIndex = (iT+1)//saveEvery;
        numLinksByFib[saveIndex,:] = CLNet.numLinksOnEachFiber();
        ofCL = prepareOutFile('BundlingBehavior/Step'+str(saveIndex)+'Links'+OutputFileName);
        CLNet.writeLinks(ofCL)
        ofCL.close()
               
        # Bundles where connections are 2 links separated by 2*restlen
        numBundlesSep[saveIndex], AllLabels[saveIndex,:] = CLNet.FindBundles(bunddist);
        print('Number bundles (1 per bundle possible) %d' %numBundlesSep[saveIndex])
        Orders, NPerBundle, avgTangents = CLNet.BundleOrderParameters(allFibers,numBundlesSep[saveIndex], AllLabels[saveIndex,:],minPerBundle=2)
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
        
        print('Time to compute network info %f ' %(time.time()-thist));
       
if (True):  
    np.savetxt('BundlingBehavior/nLinksPerFib'+OutputFileName,numLinksByFib);  
    np.savetxt('BundlingBehavior/NumFibsConnectedPerFib'+OutputFileName,NumFibsConnected);
    np.savetxt('BundlingBehavior/LocalAlignmentPerFib'+OutputFileName,AllLocalAlignment);
    np.savetxt('BundlingBehavior/NumberOfBundles_Sep'+OutputFileName,numBundlesSep);    
    np.savetxt('BundlingBehavior/BundleOrderParams_Sep'+OutputFileName,AllOrders_Sep);
    np.savetxt('BundlingBehavior/NFibsPerBundle_Sep'+OutputFileName,NPerBundleAll_Sep);
    np.savetxt('BundlingBehavior/FinalLabels_Sep'+OutputFileName,AllLabels);
    np.savetxt('BundlingBehavior/AvgTangents'+OutputFileName,AllaverageBundleTangents);

if (True):
    # Write the state to file for restart
    np.savetxt('DynamicStress/DynamicSSLocs'+OutputFileName,allFibers._ptsCheb);
    np.savetxt('DynamicStress/DynamicSSTanVecs'+OutputFileName,allFibers._tanvecs);
    np.savetxt('DynamicStress/DynSSFreeLinkBound'+OutputFileName, CLNet._FreeLinkBound);
    ofCL = prepareOutFile('DynamicStress/DynSSLinks'+OutputFileName);
    CLNet.writeLinks(ofCL)
    ofCL.close()

# Destruction and cleanup
of.close();
del Dom;
del Ewald;
del fibDisc;
del allFibers;
del TIntegrator; 
