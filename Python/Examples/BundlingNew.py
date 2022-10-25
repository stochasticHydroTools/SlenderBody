from fiberCollectionNew import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretizationNew import ChebyshevDiscretization
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler
from DiscretizedFiberNew import DiscretizedFiber
from FileIO import prepareOutFile, writeArray
import numpy as np
import chebfcns as cf
from math import exp
import sys, time
from warnings import warn

def saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,OutputFileName,wora='a'):
    Xf = allFibers.getX();
    LinkStrains = CLNet.calcLinkStrains(allFibers.getUniformPoints(Xf), Dom);
    LinkStrainSqu = np.sum(LinkStrains**2);
    FibCurvatures = allFibers.calcCurvatures(Xf);
    writeArray('BundlingBehavior/LinkStrains'+OutputFileName,[LinkStrainSqu],wora=wora)
    writeArray('BundlingBehavior/FibCurvesF'+OutputFileName,FibCurvatures,wora=wora)

Input = open('SemiflexBundleInputFile.txt','r')
for iLine in Input:
	exec(iLine);
InputCopyName='BundlingBehavior/SemiflexBundleInputFile_'+FileString;
copyInput = open(InputCopyName,'w')
Input = open('SemiflexBundleInputFile.txt','r')
for iLine in Input:
	copyInput.write(iLine);
copyInput.write('COMMAND LINE ARGS \n')
copyInput.write('CLseed = '+str(seed)+'\n')
copyInput.write('dt = '+str(dt)+'\n')
copyInput.close();
Input.close();
    
# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=deltaLocal,\
    trueRPYMobility=truRPYMob,UseEnergyDisc=True,nptsUniform=Nuniformsites);

# Initialize the master list of fibers
if (FluctuatingFibs):
    allFibers = SemiflexiblefiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads=nThr);
else:
    allFibers = fiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,nThreads=nThr);

# Initialize the fiber list (straight fibers)
np.random.seed(seed);
fibList = [None]*nFib;
allFibers.initFibList(fibList,Dom);

# Initialize the network of cross linkers
# New seed for CLs
np.random.seed(seed);
# Initialize network
print('Number uniform sites %d' %fibDisc._nptsUniform)
CLNet = DoubleEndedCrossLinkedNetwork(nFib,fibDisc._Nx,fibDisc._nptsUniform,Lf,Kspring,\
    rl,konCL,koffCL,konSecond,koffSecond,seed,Dom,fibDisc,nThreads=nThr,\
    bindingSiteWidth=bindingSiteWidth,kT=kbT);
CLNet.updateNetwork(allFibers,Dom,100.0/min(konCL*Lf,konSecond*Lf,koffCL,koffSecond)) # just to load up CLs
print('Number of links initially %d' %CLNet._nDoubleBoundLinks)

# Initialize the temporal integrator
TIntegrator = BackwardEuler(allFibers,CLNet,FPimp=(nonLocal > 0));
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
TIntegrator.setMaxIters(1);

# Prepare the output file and write initial locations
np.random.seed(seed);
# Barymat for the quarter points
LocsFileName = 'BundlingBehavior/Locs'+FileString;
allFibers.writeFiberLocations(LocsFileName,'w');
#if (seed==1):
#    ofCL = prepareOutFile('BundlingBehavior/Step'+str(0)+'Links'+FileString);
#    CLNet.writeLinks(ofCL)
#    ofCL.close()

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
        wr=1;
        mythist = time.time()
    maxX, _, _ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,outfile=LocsFileName,write=wr,updateNet=updateNet);
    if (wr==1):
        print('Time %1.2E' %(float(iT+1)*dt));
        print('MAIN Time step time %f ' %(time.time()-mythist));
        print('Max x: %f' %maxX)
        thist = time.time();
        print('Number of links %d' %CLNet._nDoubleBoundLinks)
        saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,FileString);
        saveIndex = (iT+1)//saveEvery;
        numLinksByFib[saveIndex,:] = CLNet.numLinksOnEachFiber();
        #if (seed==1):
        #    ofCL = prepareOutFile('BundlingBehavior/Step'+str(saveIndex)+'Links'+FileString);
        #    CLNet.writeLinks(ofCL)
        #    ofCL.close()
               
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
    np.savetxt('BundlingBehavior/nLinksPerFib'+FileString,numLinksByFib);  
    np.savetxt('BundlingBehavior/NumFibsConnectedPerFib'+FileString,NumFibsConnected);
    np.savetxt('BundlingBehavior/LocalAlignmentPerFib'+FileString,AllLocalAlignment);
    np.savetxt('BundlingBehavior/NumberOfBundles_Sep'+FileString,numBundlesSep);    
    np.savetxt('BundlingBehavior/BundleOrderParams_Sep'+FileString,AllOrders_Sep);
    np.savetxt('BundlingBehavior/NFibsPerBundle_Sep'+FileString,NPerBundleAll_Sep);
    np.savetxt('BundlingBehavior/FinalLabels_Sep'+FileString,AllLabels);
    np.savetxt('BundlingBehavior/AvgTangents'+FileString,AllaverageBundleTangents);
