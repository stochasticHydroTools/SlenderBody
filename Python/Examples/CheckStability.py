from math import pi
from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import BackwardEuler, CrankNicolson
import numpy as np
import sys
from FileIO import prepareOutFile

"""
This file runs a stability test. 1000 fibers in shear flow. 
"""

# Inputs for the slender body simulation
nFib=1000		    # number of fibers
N=16		    # number of points per fiber
Lf=2		    # length of each fiber
nonLocal=1	    # doing nonlocal solves? See fiberCollection.py for list of possible values. 
deltaLocal=0;
#Ld=4		    # length of the periodic domain
mu=1		    # fluid viscosity
eps=1e-2;	    # slenderness ratio
dt=5e-2		    # timestep
omega=2*pi	    # frequency of oscillations
gam0=0.2*pi	    # base rate of strain 
tf=5		    # final time
if (eps < 5e-3):
    Ldlist = [2.4];
else:
    Ldlist = [3];
RPYSQ=True; # Special quadrature
RPYOversamp = not (RPYSQ);  # Oversampled quadrature
if (RPYOversamp):           # Number of points for oversampled quad
    NDirect = int(1/eps);   # If doing self with oversampled, use 1/epsilon pts.
else:
    NDirect = 128;  # If not doing self, use fixed number of points.

Iters=[];
for Ld in Ldlist:
    print('Ld = %f' %Ld)
    Lits = [];
    for bendpow in [-2, -1, 0, 1, 2]: # loop over bending moduli
        Eb = 10.0**bendpow;
        print('Bending modulus %f' %Eb);
        giters = 1; # begin with 1 iteration
        done=0;

        while (done==0):
            print('Iterations %d' %giters)
            try: 
                np.random.seed(1);
                #if (N == 32 and eps > 5e-3):
                #    np.random.seed(2);
                # Initialize the domain and spatial database
                Dom = PeriodicShearedDomain(Ld,Ld,Ld);

                # Initialize fiber discretization
                fibDisc =  ChebyshevDiscretization(Lf,eps,Eb,mu,N,RPYSpecialQuad=RPYSQ,\
                    RPYDirectQuad=False,RPYOversample=RPYOversamp,NupsampleForDirect=NDirect,FPIsLocal=True);
                
                # Initialize Ewald for non-local velocities
                totnumDir = fibDisc._nptsDirect*nFib;
                xi = 3*totnumDir**(1/3)/Ld; # Ewald param
                Ewald = GPUEwaldSplitter(fibDisc._a,mu,xi,Dom,fibDisc._nptsDirect*nFib);

                # Initialize the master list of fibers
                allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,0,nThreads=16);

                # Initialize the fiber list
                fibList = [None]*nFib;
                allFibers.initFibList(fibList,Dom);
                allFibers.fillPointArrays();
                allFibers.writeFiberLocations('LocsForStability.txt','w');

                # Initialize the temporal integrator
                TIntegrator = BackwardEuler(allFibers);
                TIntegrator.setMaxIters(giters)

                # Compute the endtime and do the time loop
                stopcount = int(tf/dt+1e-10);
                for iT in range(stopcount): 
                    #print('Time %1.2E' %(float(iT)*dt));
                    maxX, _, _, _ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,write=0);
                    print('iT %d, Max x: %f' %(iT, maxX));
                    xlimit = 2*Ld;
                    if (Ld < 3):
                        xlimit = 6;
                    elif (Ld < 4):
                        xlimit = 7;
                    if (maxX > xlimit):
                        print('Max x: %f' %maxX);
                        raise ValueError('Unstable with Eb = %f and giters = %d' %(Eb,giters))

                # Destruction and cleanup
                done=1;
                Lits.append(giters)
                #of.close();
                del Dom;
                del Ewald;
                del fibDisc;
                del allFibers;
                del TIntegrator; 
    
            except: # unstable! Do more iterations
                giters+=1;
                if (giters > 10):
                    done = 1;
                    print('Not stable with 10 iterations, giving up.')
                
        print('Number of iterations for stability with N = %d and Ld=%f' %(N,Ld)) 
        print(Lits)          

