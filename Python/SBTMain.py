from math import pi
from fiberCollection import fiberCollection
from FiberDiscretization import FiberDiscretization
from EwaldSplitter import EwaldSplitter
from Domain import Domain
from TemporalIntegrator import TemporalIntegrator, BackwardEuler, CrankNicolsonLMM
from FibInitializer import makeCurvedFiber, makeFallingFibers
from CrossLinkedNetwork import CrossLinkedNetwork
import numpy as np

def prepareOutFile(name):
    outFile = name;
    of = open(outFile,'w');
    of.close();
    of = open(outFile,'a');
    return of;


# Seed random number generator (for reproducibility)
np.random.seed(0);

# Read the input file
infile = open('InputFile.txt','r');
lines = infile.readlines();
for line in lines:
    exec(line)
infile.close();

# Initialize the domain
Dom = Domain(Ld,Ld,Ld);

# Initialize Ewald for non-local velocities
Ewald = EwaldSplitter(xi,np.sqrt(1.5)*eps*Lf,mu);

# Initialize fiber discretization
fibDisc = FiberDiscretization(Lf,eps,ellipsoidal,Eb,mu,N);

# Initialize the master list of fibers
allFibers = fiberCollection(nFib,fibDisc,nonLocal,mu,omega,gam0);

# Initialize the fiber list
# For the single relaxing fiber test
#fibList = makeCurvedFiber(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
#fibList = makeFallingFibers(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
fibList = [None]*nFib;
allFibers.initFibList(fibList,Dom);

# Initialize the network of cross linkers
CLNet = CrossLinkedNetwork(nFib,N,Lf,grav,nCL,Kspring,rl);

# Initialize the temporal integrator
if (solver==2):
    TIntegrator = CrankNicolsonLMM(allFibers, CLNet);
elif (solver==1):
    TIntegrator = BackwardEuler(allFibers, CLNet);
else:
    raise ValueError('Invalid solver. Either 2 for Crank Nicolson or 1 for backward Euler')

# Prepare the output file and write initial locations
of = prepareOutFile('Locations.txt');
allFibers.writeFiberLocations(of);

# Compute the endtime
stopcount = int(tf/dt+1e-10);
if (stopcount - tf/dt) > 1e-10:
    print("Warning: the stop time is not an integer number of dt\'s")

# Time loop
for iT in range(stopcount): 
    print 'Time %f' %(iT*dt);
    TIntegrator.updateAllFibers(iT,dt,Dom,Ewald,of);

# Destruction and cleanup
of.close();

