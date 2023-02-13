from fiberCollection import fiberCollection, SemiflexiblefiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from RPYVelocityEvaluator import GPUEwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, MidpointDriftIntegrator
from DiscretizedFiber import DiscretizedFiber
from FileIO import prepareOutFile, writeArray
import numpy as np
import chebfcns as cf
from math import exp, pi
import sys, time
from warnings import warn

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
	InputCopyName='DynamicRheo/InputFile_'+OutFileString;
	copyInput = open(InputCopyName,'w')
	Input = open('StrainInputFile.txt','r')
	for iLine in Input:
		copyInput.write(iLine);
	copyInput.write('COMMAND LINE ARGS \n')
	copyInput.write('seed = '+str(seed)+'\n')
	copyInput.write('dt = '+str(dt)+'\n')
	copyInput.write('Omega = '+str(omHz)+'\n')
	copyInput.write('Strain = '+str(maxStrain)+'\n')
	copyInput.write('l_p = '+str(lp)+'\n')
	copyInput.close();
	Input.close()
except:
	raise ValueError('This file takes three command line arguments: the seed (for CL network),\
dt, the frequency omega to shear at, the maximum strain (on [0,1]), and l_p, in that order')


# Array of frequencies in Hz 
omega = 2*pi*omHz;
if (omHz > 0):
    T = 1/omHz;               # one period
    gam0 = maxStrain*omega    # strain amplitude 
else:
    gam0 = maxStrain;
    T = 4;     
# Set up simulation variables  
print('Omega %f: stopping time %f, saveEvery %f' %(omHz,tf,saveEvery*dt))      

# Initialize the domain
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=deltaLocal,\
    NupsampleForDirect=NupsampleForDirect,RPYSpecialQuad=RPYQuad,RPYDirectQuad=RPYDirect,RPYOversample=RPYOversample,
    UseEnergyDisc=True,nptsUniform=Nuniformsites,FPIsLocal=(nonLocal>0));

# Initialize the master list of fibers
if (FluctuatingFibs):
    allFibers = SemiflexiblefiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads=nThr);
else:
    allFibers = fiberCollection(nFib,turnovertime,fibDisc,nonLocal,mu,omega,gam0,Dom,kbT,eigValThres,nThreads=nThr,rigidFibs=rigidDetFibs);

Ewald = None;
if (nonLocal==1):
    totnumDir = fibDisc._nptsDirect*nFib;
    xi = 3*totnumDir**(1/3)/Ld; # Ewald param
    Ewald = GPUEwaldSplitter(fibDisc._a,mu,xi,Dom,fibDisc._nptsDirect*nFib);   

# Initialize the fiber list (straight fibers)
fibList = [None]*nFib;
XFile = 'BundlingBehavior/FinalLocs'+InFileString;
allFibers.initFibList(fibList,Dom,XFile);

# Initialize the network of cross linkers
# New seed for CLs
np.random.seed(seed);
# Initialize network
CLNet = DoubleEndedCrossLinkedNetwork(nFib,fibDisc._Nx,fibDisc._nptsUniform,Lf,Kspring,rl,\
    konCL,koffCL,konSecond,koffSecond,seed,Dom,fibDisc,nThreads=nThr,bindingSiteWidth=bindingSiteWidth,\
    kT=kbT,smoothForce=smForce);
CLNet.setLinksFromFile('BundlingBehavior/FinalLinks'+InFileString,'BundlingBehavior/FinalFreeLinkBound'+InFileString);
           
# Initialize the temporal integrator
TIntegrator = BackwardEuler(allFibers,CLNet);
if (FluctuatingFibs):
    TIntegrator = MidpointDriftIntegrator(allFibers,CLNet);
# Number of GMRES iterations for nonlocal solves
# 1 = block diagonal solver
# N > 1 = N-1 extra iterations of GMRES
TIntegrator.setMaxIters(gitersDeterministic);

stopcount = int(tf/dt+1e-10);
numSaves = stopcount//saveEvery+1;
Lamstress21 = np.zeros(stopcount); 
Elstress21 = np.zeros(stopcount); 
CLstress21 = np.zeros(stopcount);

LocsFileName = 'DynamicRheo/Locs'+OutFileString;
allFibers.writeFiberLocations(LocsFileName,'w');

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
saveCurvaturesAndStrains(allFibers,CLNet,OutFileString);
   
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
    maxX, _, stressArray= TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,outfile=LocsFileName,write=wr,\
        updateNet=updateNet,BrownianUpdate=RigidDiffusion,Ewald=Ewald,turnoverFibs=turnover,stress=True,fixedg=fixedg);
    Lamstress21[iT]= stressArray[0];
    Elstress21[iT] = stressArray[1];
    CLstress21[iT] = stressArray[2];
    #print('After stress %f' %Dom._g)
    #print('Max x (just to check stability): %f' %(maxX));
    if (wr==1): # save curvatures and strains
        print('MAIN Time step time %f ' %(time.time()-mythist));
        print('Max x: %f' %maxX)
        thist = time.time();
        print('Number of links %d' %CLNet._nDoubleBoundLinks)
        saveCurvaturesAndStrains(allFibers,CLNet,OutFileString);
        saveIndex = (iT+1)//saveEvery;
        numLinksByFib[saveIndex,:] = CLNet.numLinksOnEachFiber();
               
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
    np.savetxt('DynamicRheo/nLinksPerFib'+OutFileString,numLinksByFib);  
    np.savetxt('DynamicRheo/NumFibsConnectedPerFib'+OutFileString,NumFibsConnected);
    np.savetxt('DynamicRheo/LocalAlignmentPerFib'+OutFileString,AllLocalAlignment);
    np.savetxt('DynamicRheo/NumberOfBundles_Sep'+OutFileString,numBundlesSep);    
    np.savetxt('DynamicRheo/BundleOrderParams_Sep'+OutFileString,AllOrders_Sep);
    np.savetxt('DynamicRheo/NFibsPerBundle_Sep'+OutFileString,NPerBundleAll_Sep);
    np.savetxt('DynamicRheo/FinalLabels_Sep'+OutFileString,AllLabels);
    np.savetxt('DynamicRheo/AvgTangents'+OutFileString,AllaverageBundleTangents);
    allFibers.writeFiberLocations('DynamicRheo/FinalLocs'+OutFileString,'w');
    np.savetxt('DynamicRheo/FinalFreeLinkBound'+OutFileString, CLNet._FreeLinkBound);
    ofCL = prepareOutFile('DynamicRheo/FinalLinks'+OutFileString);
    CLNet.writeLinks(ofCL)
    ofCL.close()
    np.savetxt('DynamicRheo/LamStress'+OutFileString,Lamstress21);
    np.savetxt('DynamicRheo/ElStress'+OutFileString,Elstress21);
    np.savetxt('DynamicRheo/CLStress'+OutFileString,CLstress21);

