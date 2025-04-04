from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from RPYVelocityEvaluator import GPUEwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, MidpointDriftIntegrator
from DiscretizedFiber import DiscretizedFiber
from FileIO import prepareOutFile, writeArray
from StericForceEvaluator import StericForceEvaluator, SegmentBasedStericForceEvaluator
import numpy as np
import chebfcns as cf
from math import exp
import sys, time, os
from warnings import warn

"""
This file runs the dynamics of a network of fibers without a background flow. 
This can be used (a) to create steady state networks with fiber turnover whose mechanics
we want to study or (b) simulate the dynamics of bundling of actin filaments. 

The first thing this script does is read the input file (SemiflexBundleInputFile.txt) and write a copy in
the folder BundlingBehavior (it will create such a folder if you do not have one). 
In the input file, there is a list of inputs to the simulation.
"""


def saveCurvaturesAndStrains(nFib,konCL,allFibers,CLNet,rl,OutputFileName,wora='a'):
    Xf = allFibers.getX();
    LinkStrains = CLNet.calcLinkStrains(allFibers.getUniformPoints(Xf), Dom);
    LinkStrainSqu = np.sum(LinkStrains**2);
    FibCurvatures = allFibers.calcCurvatures(Xf);
    writeArray('BundlingBehavior/LinkStrains'+OutputFileName,[LinkStrainSqu],wora=wora)
    writeArray('BundlingBehavior/FibCurvesF'+OutputFileName,FibCurvatures,wora=wora)

if not os.path.exists('BundlingBehavior'):
    os.makedirs('BundlingBehavior')
    
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
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYQuad,\
    RPYOversample=(not RPYQuad),NupsampleForDirect=NupsampleForDirect,nptsUniform=Nuniformsites);
    
# Automatically initialize the fat mobility
fibDiscFat=None;
RPYRadius = fibDisc._a;   
if (nonLocal and FluctuatingFibs and RPYQuad): # Fat discretization
    eps_Star = 1e-2*4/np.exp(1.5);
    fibDiscFat = ChebyshevDiscretization(Lf, eps_Star,Eb,mu,N,\
        NupsampleForDirect=NupsampleForDirect,RPYOversample=(not RPYQuad),RPYSpecialQuad=RPYQuad);
    RPYRadius = fibDiscFat._a; 
        
# Initialize the master list of fibers
if (FluctuatingFibs):
    allFibers = SemiflexiblefiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,0,\
        0,Dom,kbT,nThreads=nThr,rigidFibs=rigidFibs,fibDiscFat=fibDiscFat);
else:
    allFibers = fiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,0,0,Dom,\
        kbT,nThreads=nThr,rigidFibs=rigidFibs,fibDiscFat=fibDiscFat);
    
Ewald = None;
if (nonLocal==1):
    totnumDir = fibDisc._nptsDirect*nFib;
    xi = 3*totnumDir**(1/3)/Ld; # Ewald param
    xiHalf = xi;
    Ewald = GPUEwaldSplitter(RPYRadius,mu,xi,Dom,fibDisc._nptsDirect*nFib,xiHalf);   

# Initialize the fiber list (straight fibers)
nStericPts = int(1/eps); # Used for contact checking / pre-computations only
if (NsegForSterics > 0):
    StericEval = SegmentBasedStericForceEvaluator(nFib,fibDisc._Nx,nStericPts,fibDisc,allFibers._ptsCheb, Dom, eps*Lf,kbT,NsegForSterics,nThreads=nThr);
else: 
    StericEval = StericForceEvaluator(nFib,fibDisc._Nx,nStericPts,fibDisc,allFibers._ptsCheb, Dom, eps*Lf,kbT,nThr);
if (not Sterics):
    StericEval._DontEvalForce = True;
    
np.random.seed(seed);
fibList = [None]*nFib;
if (InFileString is None):
    allFibers.RSAFibers(fibList,Dom,StericEval,nDiameters=2);
else:
    XFile = 'BundlingBehavior/FinalLocs'+InFileString;
    allFibers.initFibList(fibList,Dom,XFile);
# allFibers.initFibList(fibList,Dom);

# Initialize the network of cross linkers
# New seed for CLs
np.random.seed(seed);
# Initialize network
print('Number uniform sites %d' %fibDisc._nptsUniform)
CLNet = DoubleEndedCrossLinkedNetwork(nFib,fibDisc._Nx,fibDisc._nptsUniform,Lf,Kspring,\
    rl,konCL,koffCL,konSecond,koffSecond,seed,Dom,fibDisc,nThreads=nThr,\
    bindingSiteWidth=bindingSiteWidth,kT=kbT,smoothForce=smForce);
if (InFileString is None):
    try:
        CLNet.updateNetwork(allFibers,Dom,100.0/min(konCL*Lf,konSecond*Lf,koffCL,koffSecond)) # just to load up CLs
    except:
        pass
else:
    CLNet.setLinksFromFile('BundlingBehavior/FinalLinks'+InFileString,'BundlingBehavior/FinalFreeLinkBound'+InFileString);
print('Number of links initially %d' %CLNet._nDoubleBoundLinks)
CLNets=[CLNet];

if (Motors):
    MotorNet = DoubleEndedCrossLinkedNetwork(nFib,fibDisc._Nx,fibDisc._nptsUniform,Lf,Kspring_M,\
        rl_M,kon_M,koff_M,konSecond_M,koffSecond_M,seed,Dom,fibDisc,nThreads=nThr,\
        bindingSiteWidth=bindingSiteWidth,kT=kbT,smoothForce=smForce,UnloadedVel=V0_M,StallForce=Fst_M);
    if (InFileString is None):
        MotorNet.updateNetwork(allFibers,Dom,100.0/min(kon_M*Lf,konSecond_M*Lf,koff_M,koffSecond_M)) # just to load up CLs
    else:
        MotorNet.setLinksFromFile('BundlingBehavior/FinalMotors'+InFileString,'BundlingBehavior/FinalFreeMotorBound'+InFileString);
    CLNets.append(MotorNet);
    print('Number of motors initially %d' %MotorNet._nDoubleBoundLinks)
        

# Initialize the temporal integrator
if (FluctuatingFibs):
    TIntegrator = MidpointDriftIntegrator(allFibers,CLNets);
else:
    TIntegrator = BackwardEuler(allFibers,CLNets);
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
if (nonLocal==1):
    TIntegrator.setMaxIters(2);
else:
    TIntegrator.setMaxIters(1);


# Prepare the output file and write initial locations
np.random.seed(seed);
# Barymat for the quarter points
LocsFileName = 'BundlingBehavior/Locs'+FileString;
allFibers.writeFiberLocations(LocsFileName,'w');
if (False and seed==1):
    ofCL = prepareOutFile('BundlingBehavior/Step'+str(0)+'Links'+FileString);
    CLNet.writeLinks(ofCL)
    ofCL.close()
    if (Motors):
        ofMot = prepareOutFile('BundlingBehavior/Step'+str(0)+'Motors'+FileString);
        MotorNet.writeLinks(ofMot)
        ofMot.close()

stopcount = int(tf/dt+1e-10);
numSaves = stopcount//saveEvery+1;

NumFibsConnected =  np.zeros((numSaves,nFib),dtype=np.int64)
AllLocalAlignment =  np.zeros((numSaves,nFib))
AllLabels = np.zeros((numSaves,nFib),dtype=np.int64);

numLinksByFib = np.zeros((numSaves,nFib),dtype=np.int64);
numLinksByFib[0,:] = CLNet.numLinksOnEachFiber();

if (Motors):
    numMotsByFib = np.zeros((numSaves,nFib),dtype=np.int64);
    numMotsByFib[0,:] = MotorNet.numLinksOnEachFiber();
    MotorSpeeds = MotorNet.MotorSpeeds(allFibers,Dom);

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
nContacts = np.zeros(numSaves);
_, _, FibContacts=StericEval.CheckContacts(allFibers._ptsCheb,Dom, excludeSelf=True);
nCont, _ = FibContacts.shape;
nContacts[0] = nCont;

# Simulate 
for iT in range(stopcount): 
    wr=0;
    if ((iT % saveEvery) == (saveEvery-1)):
        wr=1;
        mythist = time.time()
    maxX, ItsNeed[iT], _ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,outfile=LocsFileName,write=wr,\
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
        _, _, FibContacts=StericEval.CheckContacts(allFibers._ptsCheb,Dom, excludeSelf=True);
        nCont, _ = FibContacts.shape;
        print('Number of contacts %d' %nCont)
        nContacts[saveIndex]=nCont;
        if (Motors):
            print('Number of motors %d' %MotorNet._nDoubleBoundLinks)   
            numMotsByFib[saveIndex,:] = MotorNet.numLinksOnEachFiber(); 
            NewSpeeds = MotorNet.MotorSpeeds(allFibers,Dom);
            print(NewSpeeds.shape)
            MotorSpeeds=np.append(MotorSpeeds,NewSpeeds);
        if (False and seed==1):
            ofCL = prepareOutFile('BundlingBehavior/Step'+str(saveIndex)+'Links'+FileString);
            CLNet.writeLinks(ofCL)
            ofCL.close()
            if (Motors):
                ofMot = prepareOutFile('BundlingBehavior/Step'+str(saveIndex)+'Motors'+FileString);
                MotorNet.writeLinks(ofMot)
                ofMot.close()
               
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
    if (Motors):
        np.savetxt('BundlingBehavior/FinalFreeMotorBound'+FileString, MotorNet._FreeLinkBound);
        np.savetxt('BundlingBehavior/nMotorsPerFib'+FileString,numMotsByFib);  
        np.savetxt('BundlingBehavior/MotorSpeeds'+FileString,MotorSpeeds);
        ofMot = prepareOutFile('BundlingBehavior/FinalMotors'+FileString);
        MotorNet.writeLinks(ofMot)
        ofMot.close()
