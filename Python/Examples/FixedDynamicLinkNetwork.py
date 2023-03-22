from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from RPYVelocityEvaluator import GPUEwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, MidpointDriftIntegrator
from DiscretizedFiber import DiscretizedFiber
from FileIO import prepareOutFile, writeArray
from StericForceEvaluator import StericForceEvaluator
import numpy as np
import chebfcns as cf
from math import exp
import sys, time
from warnings import warn

"""
This file runs the dynamics of a network of fibers without a background flow. 
This can be used (a) to create steady state networks with fiber turnover whose mechanics
we want to study or (b) simulate the dynamics of bundling of actin filaments. 

The first thing this script does is read the input file (SemiflexBundleInputFile.txt) and write a copy in
the folder BundlingBehavior (create such a folder if you do not have one). In the input file, there is a list of inputs
to the simulation. It is currently set up to take 3 command line arguments in this order: the seed, the time step size, and 
the persistence length. So, for example, running
python BundlingNew.py 1 0.0001 10
will simulate fibers with persistence length 10 using a time step 0.0001 and a seed of 1. 
There are a number of outputs which are detailed in the README file. 
"""


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
copyInput.write('lp = '+str(lp)+'\n')
copyInput.close();
Input.close();
    
# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=deltaLocal,\
    NupsampleForDirect=NupsampleForDirect,RPYSpecialQuad=RPYQuad,RPYDirectQuad=RPYDirect,RPYOversample=RPYOversample,
    UseEnergyDisc=True,nptsUniform=Nuniformsites,FPIsLocal=(nonLocal>0));

# Initialize the master list of fibers
if (FluctuatingFibs):
    allFibers = SemiflexiblefiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,nThreads=nThr,rigidFibs=rigidFibs);
else:
    allFibers = fiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,nThreads=nThr,rigidFibs=rigidFibs);

Ewald = None;
if (nonLocal==1):
    totnumDir = fibDisc._nptsDirect*nFib;
    xi = 3*totnumDir**(1/3)/Ld; # Ewald param
    Ewald = GPUEwaldSplitter(fibDisc._a,mu,xi,Dom,fibDisc._nptsDirect*nFib);   

# Initialize the fiber list (straight fibers)
nStericPts = int(1/eps);
FibDiameter = 2*eps*Lf;
StericEval = None;
if (Sterics):
    StericEval = StericForceEvaluator(nFib,fibDisc._Nx,nStericPts,fibDisc,allFibers._ptsCheb, Dom, FibDiameter,nThr);
    
np.random.seed(seed);
fibList = [None]*nFib;
if (Sterics):
    allFibers.RSAFibers(fibList,Dom,StericEval,nDiameters=4);
else:
    allFibers.initFibList(fibList,Dom);

# Initialize the network of cross linkers
# New seed for CLs
np.random.seed(seed);
# Initialize network
print('Number uniform sites %d' %fibDisc._nptsUniform)
CLNet = DoubleEndedCrossLinkedNetwork(nFib,fibDisc._Nx,fibDisc._nptsUniform,Lf,Kspring,\
    rl,konCL,koffCL,konSecond,koffSecond,seed,Dom,fibDisc,nThreads=nThr,\
    bindingSiteWidth=bindingSiteWidth,kT=kbT,smoothForce=smForce);
CLNet.updateNetwork(allFibers,Dom,100.0/min(konCL*Lf,konSecond*Lf,koffCL,koffSecond)) # just to load up CLs
print('Number of links initially %d' %CLNet._nDoubleBoundLinks)

# Initialize the temporal integrator
if (FluctuatingFibs):
    TIntegrator = MidpointDriftIntegrator(allFibers,CLNet);
else:
    TIntegrator = BackwardEuler(allFibers,CLNet);
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
#   ofCL.close()

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
saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,FileString);

ItsNeed = np.zeros(stopcount);
nContacts = np.zeros(stopcount);
        
# Simulate 
for iT in range(stopcount): 
    wr=0;
    if ((iT % saveEvery) == (saveEvery-1)):
        wr=1;
        mythist = time.time()
    maxX, ItsNeed[iT], _, nContacts[iT] = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,outfile=LocsFileName,write=wr,\
        updateNet=updateNet,BrownianUpdate=RigidDiffusion,Ewald=Ewald,turnoverFibs=turnover,StericEval=StericEval);
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
        #   ofCL.close()
               
        # Bundles where connections are 2 links
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
    np.savetxt('BundlingBehavior/ItsNeeded'+FileString,ItsNeed);
    np.savetxt('BundlingBehavior/nContacts'+FileString,nContacts);
    np.savetxt('BundlingBehavior/nLinksPerFib'+FileString,numLinksByFib);  
    np.savetxt('BundlingBehavior/NumFibsConnectedPerFib'+FileString,NumFibsConnected);
    np.savetxt('BundlingBehavior/LocalAlignmentPerFib'+FileString,AllLocalAlignment);
    np.savetxt('BundlingBehavior/NumberOfBundles_Sep'+FileString,numBundlesSep);    
    np.savetxt('BundlingBehavior/BundleOrderParams_Sep'+FileString,AllOrders_Sep);
    np.savetxt('BundlingBehavior/NFibsPerBundle_Sep'+FileString,NPerBundleAll_Sep);
    np.savetxt('BundlingBehavior/FinalLabels_Sep'+FileString,AllLabels);
    np.savetxt('BundlingBehavior/AvgTangents'+FileString,AllaverageBundleTangents);
    allFibers.writeFiberLocations('BundlingBehavior/FinalLocs'+FileString,'w');
    np.savetxt('BundlingBehavior/FinalFreeLinkBound'+FileString, CLNet._FreeLinkBound);
    ofCL = prepareOutFile('BundlingBehavior/FinalLinks'+FileString);
    CLNet.writeLinks(ofCL)
    ofCL.close()
