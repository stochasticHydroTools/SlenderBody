from math import pi
from Fiber import Fiber
from Fluid import Fluid
from ExForces import ExForces
from FibInitializer import makeCurvedFiber, makeFallingFibers, makeRandomFibers
import chebfcns as cf
import numpy as np

def prepareOutFile(name, fibList):
    outFile = name;
    of = open(outFile,'w');
    of.close();
    of = open(outFile,'a');
    # Write the initial locations
    for fib in fibList:
        fib.writeLocs(of);
    return of;


# Seed random number generator (for reproducibility)
np.random.seed(0);

# Read the input file
infile = open('InputFile.txt','r');
lines = infile.readlines();
for line in lines:
    exec(line)
infile.close();

# Initialize the fiber list
if (nFib==1):
    fibList = makeCurvedFiber(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
elif (nFib==4):
    fibList = makeFallingFibers(nFib,Lf,eps,ellipsoidal,Eb,mu,N);
else:
    fibList = makeRandomFibers(nFib,Lf,eps,ellipsoidal,Eb,mu,N,Ld,Ld,Ld);

# Initialize the fluid
flu = Fluid(nFib,N,nonLocal,Ld,Ld,Ld,xi,mu,omega,gam0,Lf,eps);

# Initialize the external forcing (if any)
exForce = ExForces(nFib,N,Lf,grav,nCL,Kspring,rl,Ld,Ld,Ld);

# Prepare the output file and write initial locations
of = prepareOutFile('Locations.txt',fibList);

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
    notconvbyFib = np.ones(nFib);
    notconv = 1;
    if (iT < abs(solver)): # for now only doing a max of 10 iterations the first two steps
        itmax=10;#float("inf");
    # Compute the external forcing (not dependent on lambda so outside loop)
    fext = exForce.totalExForce(fibList,flu.getg(tval),solver);
    # Fixed point iteration to solve for the lambdas
    while (iters < itmax and notconv):
        # Compute non-local velocity (this includes finite part integral)
        glam = LMMlam and (iT > abs(solver)-1) and iters==0; # whether to use LMM for lambda
        unL = flu.nLvel(fibList,solver,glam,np.reshape(fext,(N*nFib,3)),tval);
        # Use it to update fibers
        for jFib in range(len(fibList)):
            fib = fibList[jFib];
            fib.updateX(dt,solver,iters,unL[jFib*3*N:(jFib+1)*3*N],fext[jFib*3*N:(jFib+1)*3*N]);
            notconvbyFib[jFib] = fib.notConverged(1e-6);
        notconv = (sum(notconvbyFib) > 0);
        iters+=1;
    # Once the fixed point iteration has converged, update all the fiber tangent vectors
    for fib in fibList:
        fib.updateXs(dt,solver,exacinex);
        fib.writeLocs(of);
of.close();

