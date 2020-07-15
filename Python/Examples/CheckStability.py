from math import pi
from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson
import numpy as np

"""
This file runs the stability test in Section 5.2. 
"""

# Inputs for the slender body simulation
nFib=1000		# number of fibers
N=16		    # number of points per fiber
Lf=2		    # length of each fiber
nonLocal=1	    # doing nonlocal solves?
Ld=5.0		    # length of the periodic domain
xi = 0.5*(N*nFib)**(1/3)/Ld; # Ewald param
mu=1		    # fluid viscosity
eps=1e-3	    # slenderness ratio
dt=5e-2		    # timestep
omega=2*pi	    # frequency of oscillations
gam0=0.2*pi	    # base rate of strain 
tf=5		    # final time
gravity=0.0     # no uniform force

Iters=[];
for bendpow in range(-2,3): # loop over bending moduli
    Eb = 10.0**bendpow;
    print('Bending modulus %f' %Eb);
    giters = 1; # begin with 1 iteration
    done=0;

    while (done==0):
        print('Iterations %d' %giters)
        try: 
            np.random.seed(1);
            # Initialize the domain and spatial database
            Dom = PeriodicShearedDomain(Ld,Ld,Ld);

            # Initialize Ewald for non-local velocities
            Ewald = EwaldSplitter(np.sqrt(1.5)*eps*Lf,mu,xi,Dom,N*nFib);

            # Initialize fiber discretization
            fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N);

            # Initialize the master list of fibers
            allFibers = fiberCollection(nFib,fibDisc,nonLocal,mu,omega,gam0,Dom,nThreads=4);

            # Initialize the fiber list
            fibList = [None]*nFib;
            allFibers.initFibList(fibList,Dom);
            allFibers.fillPointArrays();

            # Initialize the temporal integrator
            TIntegrator = CrankNicolson(allFibers);
            TIntegrator.setMaxIters(giters)

            # Compute the endtime and do the time loop
            stopcount = int(tf/dt+1e-10);
            for iT in range(stopcount): 
                print('Time %1.2E' %(float(iT)*dt));
                maxX = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,gravity,write=0);
                print('Max x: %f' %maxX);
                if (maxX > Ld*Ld):
                    raise ValueError('Unstable with Eb = %f' %(Eb))

            # Destruction and cleanup
            done=1;
            Iters.append(giters)
            of.close();
            del Dom;
            del Ewald;
            del fibDisc;
            del allFibers;
            del CLNet;
            del TIntegrator; 
    
        except: # unstable! Do more iterations
            giters+=1;

print('Number of iterations for stability')        
print(Iters)

