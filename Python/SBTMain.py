from math import pi
from fiberCollection import fiberCollection
from FibCollocationDiscretization import FibCollocationDiscretization, ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter
from Domain import Domain, PeriodicShearedDomain
from TemporalIntegrator import TemporalIntegrator, BackwardEuler, CrankNicolsonLMM
from FibInitializer import makeCurvedFiber, makeFallingFibers
from CrossLinkedNetwork import CrossLinkedNetwork, KMCCrossLinkedNetwork
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

# Initialize the domain and spatial database
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize Ewald for non-local velocities
Ewald = EwaldSplitter(np.sqrt(1.5)*eps*Lf,mu,xi,Dom);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,ellipsoidal,Eb,mu,N);

# Initialize the master list of fibers
allFibers = fiberCollection(nFib,fibDisc,nonLocal,mu,omega,gam0,Dom);

# Initialize the fiber list
# For the single relaxing fiber test
#fibList = makeCurvedFiber(Lf,N,fibDisc);
#fibList = makeFallingFibers(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
fibList = [None]*nFib;
allFibers.initFibList(fibList,Dom);
allFibers.fillPointArrays();

# Initialize the network of cross linkers
CLNet = KMCCrossLinkedNetwork(nFib,N,fibDisc.getNumUniform(),Lf,grav,nCL,Kspring,rl,konCL,koffCL);

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
    print('Time %f' %(float(iT)*dt));
    TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,of);

# Destruction and cleanup
of.close();
del Dom;
del Ewald;
del fibDisc;
del allFibers;
del CLNet;
del TIntegrator;

