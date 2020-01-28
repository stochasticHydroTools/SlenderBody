from math import pi
from fiberCollection import fiberCollection
from FiberDiscretization import FiberDiscretization
from DiscretizedFiber import DiscretizedFiber
from Domain import Domain
from FibInitializer import makeCurvedFiber, makeFallingFibers
import chebfcns as cf
import numpy as np

def prepareOutFile(name):
    outFile = name;
    of = open(outFile,'w');
    of.close();
    of = open(outFile,'a');
    return of;


# Seed random number generator (for reproducibility)
np.random.seed(0);
fixedpointtol=1e-6;

# Read the input file
infile = open('InputFile.txt','r');
lines = infile.readlines();
for line in lines:
    exec(line)
infile.close();

# Initialize the domain
Dom = Domain(Ld,Ld,Ld);

# Initialize fiber discretization
fibDisc = FiberDiscretization(Lf,eps,ellipsoidal,Eb,mu,N);

# Initialize the master list of fibers
allFibers = fiberCollection(nFib,fibDisc,nonLocal,Dom,xi,mu,omega,gam0)
# Initialize the fiber list
# For the single relaxing fiber test
#fibList = makeCurvedFiber(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
#fibList = makeFallingFibers(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
fibList = [None]*nFib;
allFibers.initFibList(fibList);

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
    tval = (iT+(abs(solver)-1)/2.0)*dt; # the argument for the background flow t
    iters=0;
    itmax = maxiters;
    anyNot = 1;
    if (iT < abs(solver)): # for now only doing a max of 10 iterations the first two steps
        itmax=10;#float("inf");
    # Points and U0 do not depend on lambda - fill those arrays outside loop
    allFibers.setg(tval);
    allFibers.fillPointBkgrndVelocityArrays(solver,tval); # also makes kD trees
    fext = np.zeros(nFib*N*3); # temporary
    # Fixed point iteration to solve for the lambdas
    while (iters < itmax and anyNot):
        # Compute non-local velocity (this includes finite part integral)
        LambdaCombo = LMMlam and (iT > abs(solver)-1) and iters==0; # LMM for lambda?
        allFibers.fillForceArrays(solver,LambdaCombo,np.reshape(fext,(N*nFib,3)));
        uNonLoc = allFibers.nonLocalBkgrndVelocity();
        allFibers.linSolveAllFibers(dt,solver,iters,uNonLoc,fext);
        anyNot = allFibers.AnyNotConverged(fixedpointtol);
        iters+=1;
    # Once the fixed point iteration has converged, update all the fiber tangent vectors
    allFibers.updateAllFibers(dt,solver,exacinex);
    allFibers.writeFiberLocations(of);

# Destruction and cleanup
of.close();

