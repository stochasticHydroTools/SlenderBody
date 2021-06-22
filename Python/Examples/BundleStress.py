from math import pi, ceil
from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import GPUEwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, CrankNicolson
from FileIO import prepareOutFile, writeArray
from DoubleEndedCrossLinkedNetwork import DoubleEndedCrossLinkedNetwork
from DiscretizedFiber import DiscretizedFiber
import numpy as np
from warnings import warn 
import time

def makeFibBundle(nFib,N,fibDisc):
    """
    Initialize the three fibers for the shear simulation for a given N and fiber length Lf. 
    fibDisc = the collocation discretization of the fiber
    """
    s=fibDisc.gets();
    # Falling fibers
    Xs = np.concatenate(([np.zeros(N)],[np.ones(N)],[np.zeros(N)]),axis=0).T;
    verts = np.zeros((nFib,2));
    verts[1:,:2] = np.loadtxt('Octagon020.txt');
    print(verts)
    fibList = [None]*9;
    for iFib in range(nFib):
        X = np.concatenate(([np.zeros(N)]+verts[iFib,0],[s],[np.zeros(N)]+verts[iFib,1]),axis=0).T;
        fibList[iFib] = DiscretizedFiber(fibDisc,X,Xs);
    return fibList;

# Inputs for the slender body simulation
nFib = 9;
Lf=1                    # length of each fiber
Ld=3                    # length of the periodic domain
xi = 2; # Ewald param
mu=0.1                # fluid viscosity
eps=4e-3            # slenderness ratio
grav=0              # value of gravity if it exists
saveEvery=50;
nIts = 1;   # number of iterations
rl = 0.05;

N = 16;
Eb = 0.07;
dt = 1e-4;#[2.5e-3,1e-3,5e-4,2.5e-4]; # Max stable = 5e-3
omHz = 0;
tf = 1#5/omHz;
omega = 2*pi*omHz;
gam0 = 0.1;

for nonLocal in [0,3,4]:
    # Initialize the domain and spatial database
    Dom = PeriodicShearedDomain(Ld,Ld,Ld);

    # Initialize fiber discretization
    NupsampleForDirect=128;
    fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=0.1,nptsUniform=40,NupsampleForDirect=NupsampleForDirect);

    fibList = makeFibBundle(nFib,N,fibDisc);
    # Initialize the master list of fibers
    allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom);
    #allFibers.initFibList(fibList,Dom,pointsfileName='StiffLocs'+str(seed+1)+'.txt',tanvecfileName='StiffTanVecs'+str(seed+1)+'.txt');
    allFibers.initFibList(fibList,Dom);
    allFibers.fillPointArrays();

    Ewald = GPUEwaldSplitter(allFibers.getaRPY()*eps*Lf,mu,xi*1.4*(fibDisc.getNumDirect()/N)**(1/3),Dom,NupsampleForDirect*nFib);

    # Initialize the temporal integrator
    TIntegrator = CrankNicolson(allFibers);
    if (nonLocal==3):
        nIts = 2; 
    TIntegrator.setMaxIters(nIts);
    TIntegrator.setLargeTol(0.1);

    # Prepare the output file and write initial locations
    of = prepareOutFile('BundleTest/XLooseBundle'+str(nonLocal)+'.txt');
    allFibers.writeFiberLocations(of);
    stopcount = int(tf/dt+1e-10);
    Lamstress = np.zeros(stopcount);
    Elstress = np.zeros(stopcount);
    nLinks =np.zeros(stopcount)
    for iT in range(stopcount): 
        maxX, _, StressArray = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,grav/Lf,write=1,outfile=of,stress=True);
        Lamstress[iT]=StressArray[0];
        Elstress[iT]=StressArray[1];
        
    np.savetxt('BundleTest/StresssLooseBundle'+str(nonLocal)+'.txt',Lamstress+Elstress);
    del allFibers;
    del fibDisc;
    del Dom;
    del TIntegrator;

