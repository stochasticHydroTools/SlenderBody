from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson
from CrossLinkedNetwork import CrossLinkedNetwork
from DiscretizedFiber import DiscretizedFiber
from FileIO import prepareOutFile
import numpy as np

"""
This file performs the three sheared fiber test in Section 5.1.2
"""

def makeThreeSheared(Lf,N,fibDisc):
    """
    Initialize the three fibers for the shear simulation for a given N and fiber length Lf. 
    fibDisc = the collocation discretization of the fiber
    """
    s=fibDisc.gets();
    # Falling fibers
    Xs13 = np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T;
    Xs2 = np.concatenate(([np.zeros(N)],[np.ones(N)],[np.zeros(N)]),axis=0).T;
    Xs13 = np.reshape(Xs13,3*N);
    Xs2 = np.reshape(Xs2,3*N);
    X1 = np.concatenate(([s-1],[np.zeros(N)-0.6],[np.zeros(N)-0.04]),axis=0).T;
    X2 = np.concatenate(([np.zeros(N)],[s-1],[np.zeros(N)]),axis=0).T;
    X3 = np.concatenate(([s-1],[np.zeros(N)+0.6],[np.zeros(N)+0.06]),axis=0).T;
    fibList = [None]*3;
    fibList[0] = DiscretizedFiber(fibDisc,np.reshape(X1,3*N),Xs13);
    fibList[1] = DiscretizedFiber(fibDisc,np.reshape(X2,3*N),Xs2);
    fibList[2] = DiscretizedFiber(fibDisc,np.reshape(X3,3*N),Xs13);
    return fibList;

# Inputs 
nFib=3          # number of fibers
N=32            # number of points per fiber
Lf=2            # length of each fiber
nonLocal=1      # doing nonlocal solves? 0 = local drag, 1 = nonlocal hydro. See fiberCollection.py for full list of values. 
Ld=2.4          # length of the periodic domain
xi=6            # Ewald parameter
mu=1            # fluid viscosity
eps=1e-3        # slenderness ratio
Eb=1e-2         # fiber bending stiffness
dt=0.01          # timestep
omega=0         # frequency of oscillations
gam0=1          # base rate of strain
tf=1          # final time
gravity = 0.0   # gravity on the fibers
giters = 1;

Ns = [32];
dts = [0.01];
for N in Ns:
    gits = [1];
    if (N==16):
        gits=[1,2,4];
    for giters in gits:
        for dt in dts:
            # Initialize the domain
            Dom = PeriodicShearedDomain(Ld,Ld,Ld);

            # Initialize fiber discretization
            fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=0.1,NupsampleForDirect=N,rigid=True);

            # Initialize the master list of fibers
            allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom);
            fibList = makeThreeSheared(Lf,N,fibDisc);
            allFibers.initFibList(fibList,Dom);
            allFibers.fillPointArrays();
            
            # Initialize Ewald for non-local velocities
            Ewald = EwaldSplitter(allFibers.getaRPY()*eps*Lf,mu,xi,Dom,N*nFib);

            # Initialize the temporal integrator
            TIntegrator = CrankNicolson(allFibers);
            # Number of GMRES iterations for nonlocal solves
            # 1 = block diagonal solver
            # N > 1 = N-1 extra iterations of GMRES
            TIntegrator.setMaxIters(giters);

            # Prepare the output file and write initial locations
            of = prepareOutFile('New2LocationsN' +str(N)+'G'+str(giters)+str(dt)+'.txt');
            allFibers.writeFiberLocations(of);

            # Time loop
            stopcount = int(tf/dt+1e-10);
            for iT in range(stopcount): 
                print('Time %f' %(float(iT)*dt));
                maxX = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,gravity/Lf,of);
                print(maxX)

            # Destruction and cleanup
            of.close();
            del Dom;
            del Ewald;
            del fibDisc;
            del allFibers;
            del TIntegrator;

