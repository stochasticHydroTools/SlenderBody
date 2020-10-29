from math import pi, ceil
from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson
from CrossLinkedNetwork import KMCCrossLinkedNetwork
from FileIO import prepareOutFile, writeArray
import numpy as np
from warnings import warn 

"""
StrainingCrossLinkedNetwork.py 
This file runs the small amplitude oscillatory shear test on 
the steady state fiber network. 
"""

def saveCurvaturesAndStrains(omega,nFib,nCL,allFibers,CLNet,HydroStr,wora='a'):
    Xf = allFibers.getX();
    LinkStrains = CLNet.calcLinkStrains(allFibers.getUniformPoints(Xf), Dom);
    FibCurvatures = allFibers.calcCurvatures(Xf);
    if (nCL > 0):
        writeArray('DynamicLinkStrains'+HydroStr+'Om'+str(omega)+'F'+str(nFib)+'C'+str(nCL)+'.txt',LinkStrains,wora=wora)
    writeArray('DynamicFibCurves'+HydroStr+'Om'+str(omega)+'F'+str(nFib)+'C'+str(nCL)+'.txt',FibCurvatures,wora=wora)

import sys
sys.path.append("/NetworkSteadyStates/")       

# Inputs for the slender body simulation
nFib = 700;             # number of fibers
nCL = 12*nFib;          # maximum # of CLs
N=16                    # number of points per fiber
Lf=2                    # length of each fiber
Ld=4                    # length of the periodic domain
xi = 0.5*(N*nFib)**(1/3)/Ld; # Ewald param
mu=1                # fluid viscosity
eps=1e-3            # slenderness ratio
Eb=0.01             # fiber bending stiffness
grav=0              # value of gravity if it exists
Kspring=1           # cross linker stiffness
rl=0.5              # cross linker rest length
konCL = 1000;       # cross linker binding rate (not applicable here, but an input to the class)
koffCL = 1e-16;     # cross linker unbinding rate (not applicable here, but an input to the class)

# Array of frequencies in Hz 
omHzs = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1, 2, 5, 10];
#iO = 1; # index of the Omega we are doing 
for iO in [9]:
    omHz = omHzs[iO];
    omega = 2*pi*omHz;
    if (iO > 0):
        loadOmega = omHzs[iO-1]; # previous omega to load
    else:
        loadOmega = 0; # will load from steady state
        
    # Set up simulation variables 
    gam0 = 0.1*omega    # strain amplitude
    T = 1.0/omHz;       # one period
    nCyc = ceil(omHz) + 3; # 3 cycles + 1 second to relax the network 
    tf = nCyc*T;
    dt = min(T/20,5e-3); # maximum stable timestep is 1e-3
    saveEvery = 1;#int(T/(4*dt)+1e-10); # measure curvature at the start and middle of each cycle
    print('Omega %f: stopping time %f, saveEvery %f' %(omHz,tf,saveEvery*dt))

    nonLocal=0; # 0 for local drag, 1 for nonlocal hydro, 4 for finite part only
    nIts = 1;   # number of iterations
    HydroStr='';
    if (nonLocal==1):
        HydroStr='HYDRO';
        nIts = 2;
        if (iO==2):
            nIts = 3;
    elif (nonLocal==4):
        HydroStr='FPONLY';
    elif (nonLocal==2):
        HydroStr='INTERH';
        nIts = 2;

    np.random.seed(1);

    # Initialize the domain and spatial database
    Dom = PeriodicShearedDomain(Ld,Ld,Ld);

    # Initialize Ewald for non-local velocities
    Ewald = EwaldSplitter(np.exp(1.5)/4*eps*Lf,mu,xi,Dom,N*nFib);

    # Initialize fiber discretization
    fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=0.1);

    # Initialize the master list of fibers
    allFibers = fiberCollection(nFib,fibDisc,nonLocal,mu,omega,gam0,Dom,nThreads=4);

    # Initialize the fiber list
    fibList = [None]*nFib;
    if (loadOmega > 0):
        print('Loading previous dynamics using Omega in Hz = %f' %loadOmega)
        XFile = 'NetworkSteadyStates/DynamicSSLocationsOm'+str(loadOmega)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt';
        XsFile = 'NetworkSteadyStates/DynamicTangentVecsOm'+str(loadOmega)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt';
    else:
        print('Loading from steady state')
        XFile = 'NetworkSteadyStates/SSLocsF'+str(nFib)+'C'+str(nCL)+'REV.txt';
        XsFile = 'NetworkSteadyStates/SSTanVecsF'+str(nFib)+'C'+str(nCL)+'REV.txt';
    allFibers.initFibList(fibList,Dom,pointsfileName=XFile,tanvecfileName=XsFile);
    allFibers.fillPointArrays();

    # Initialize the network of cross linkers
    # New seed for CLs
    CLseed = 2;
    np.random.seed(CLseed);
    CLNet = KMCCrossLinkedNetwork(nFib,N,fibDisc.getNumUniform(),Lf,nCL,Kspring,rl,konCL,koffCL,CLseed,Dom,fibDisc,nThreads=4);
    CLNet.setLinksFromFile('NetworkSteadyStates/F'+str(nFib)+'C'+str(nCL)+'.txt',Dom);
        
    # Initialize the temporal integrator
    TIntegrator = CrankNicolson(allFibers, CLNet);
    TIntegrator.setMaxIters(nIts);

    # Prepare the output file and write initial locations
    of = prepareOutFile('Straining'+HydroStr+'Om'+str(omHz)+'LocsF'+str(nFib)+'C'+str(nCL)+'REV.txt');
    allFibers.writeFiberLocations(of);
    #saveCurvaturesAndStrains(omHz,nFib,nCL,allFibers,CLNet,HydroStr,wora='w')

    stopcount = int(tf/dt+1e-10);
    Lamstress = np.zeros(stopcount); 
    Elstress = np.zeros(stopcount); 
    CLstress = np.zeros(stopcount);
    for iT in range(stopcount): 
        wr=0;
        if ((iT % saveEvery) == (saveEvery-1)):
            print('Time %1.2E' %(float(iT)*dt));
            wr=1;
        maxX = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,grav/Lf,of,write=wr);
        Lamstress[iT],Elstress[iT],CLstress[iT] = TIntegrator.computeStress(Dom,iT,dt)
        #if (iT==699):
        #    writeArray('Lambdas'+HydroStr+'.txt',TIntegrator._allFibers.getLambdas());
        if (wr==1):
            print('Max x: %f' %(maxX));
        #    saveCurvaturesAndStrains(omHz,nFib,nCL,allFibers,CLNet,HydroStr)

    writeArray('LambdaStress'+HydroStr+'Om'+str(omHz)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt',Lamstress)
    writeArray('ElasticStress'+HydroStr+'Om'+str(omHz)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt',Elstress)
    writeArray('CLStress'+HydroStr+'Om'+str(omHz)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt',CLstress)

    #ofFinalX = prepareOutFile('DynamicSSLocationsOm'+str(omHz)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt');
    #allFibers.writeFiberLocations(ofFinalX);
    #ofFinalXs = prepareOutFile('DynamicTangentVecsOm'+str(omHz)+'F'+str(nFib)+'C'+str(nCL)+'REV.txt');
    #allFibers.writeFiberTangentVectors(ofFinalXs);

    # Destruction and cleanup
    of.close();
    del Dom;
    del Ewald;
    del fibDisc;
    del allFibers;
    del TIntegrator; 



