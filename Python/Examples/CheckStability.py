from math import pi
from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter, GPUEwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson, BackwardEuler
import numpy as np
import sys
from FileIO import prepareOutFile

"""
This file runs the stability test in Section 5.2. 
"""

# Inputs for the slender body simulation
nFib=1000		    # number of fibers
#N=16		    # number of points per fiber
Lf=2		    # length of each fiber
nonLocal=3	    # doing nonlocal solves? See fiberCollection.py for list of possible values. 
deltaLocal=0.1;
#Ld=4		    # length of the periodic domain
mu=1		    # fluid viscosity
eps=float(sys.argv[2]);	    # slenderness ratio
dt=5e-2		    # timestep
omega=2*pi	    # frequency of oscillations
gam0=0.2*pi	    # base rate of strain 
tf=5		    # final time
gravity=0.0     # no uniform force

TemporalOrder = int(sys.argv[1]);
Nin = int(sys.argv[3]);
print('Temporal order: %d' %TemporalOrder)
print('Epsilon: %f' %eps)
print('N: %d' %Nin)
Ldlist = [3];
if (eps < 1e-2):
    Ldlist.append(2.4);
for ImpFP in [1]:
    print('Implicit Fp = %d' %ImpFP)
    for RPYMob in [True]:
        print('RPY Mob = %d' %RPYMob)
        for N in [Nin]:
            print('N = %d' %N)
            Iters=[];
            for Ld in Ldlist:
                print('Ld = %f' %Ld)
                Lits = [];
                xi = 2.4*(N*nFib)**(1/3)/Ld; # Ewald param
                for bendpow in range(2,3): # loop over bending moduli
                    Eb = 10.0**bendpow;
                    print('Bending modulus %f' %Eb);
                    giters = 1; # begin with 1 iteration
                    done=0;

                    while (done==0):
                        print('Iterations %d' %giters)
                        #try: 
                        np.random.seed(1);
                        # Initialize the domain and spatial database
                        Dom = PeriodicShearedDomain(Ld,Ld,Ld);

                        # Initialize fiber discretization
                        fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=deltaLocal,trueRPYMobility=RPYMob,NupsampleForDirect=128);
                        
                        # Initialize Ewald for non-local velocities
                        Ewald = GPUEwaldSplitter(np.exp(1.5)/4*eps*Lf,mu,xi,Dom,fibDisc._nptsDirect*nFib);

                        # Initialize the master list of fibers
                        allFibers = fiberCollection(nFib,10,fibDisc,nonLocal,mu,omega,gam0,Dom,nThreads=4);

                        # Initialize the fiber list
                        fibList = [None]*nFib;
                        allFibers.initFibList(fibList,Dom);
                        allFibers.fillPointArrays();
                        #of = prepareOutFile('PyLocs.txt');
                        #allFibers.writeFiberLocations(of);

                        # Initialize the temporal integrator
                        if (TemporalOrder==1):
                            TIntegrator = BackwardEuler(allFibers,FPimp=ImpFP);
                        else:
                            TIntegrator = CrankNicolson(allFibers,FPimp=ImpFP);
                        TIntegrator.setMaxIters(giters)
                        TIntegrator.setLargeTol(1e-6);

                        # Compute the endtime and do the time loop
                        stopcount = int(tf/dt+1e-10);
                        for iT in range(stopcount): 
                            #print('Time %1.2E' %(float(iT)*dt));
                            maxX, _, _ = TIntegrator.updateAllFibers(iT,dt,stopcount,Dom,Ewald,gravity,write=0);
                            #print('Max x: %f' %maxX);
                            xlimit = 2*Ld;
                            if (Ld < 3):
                                xlimit = 6;
                            elif (Ld < 4):
                                xlimit = 7;
                            if (maxX > xlimit):
                                print('Max x: %f' %maxX);
                                raise ValueError('Unstable with Eb = %f' %(Eb))

                        # Destruction and cleanup
                        done=1;
                        Lits.append(giters)
                        #of.close();
                        del Dom;
                        del Ewald;
                        del fibDisc;
                        del allFibers;
                        del TIntegrator; 
                        
                        #except: # unstable! Do more iterations
                        #    giters+=1;
                        #    if (giters > 10):
                        #        done = 1;
                        #        print('Not stable with 10 iterations, giving up.')
                
                print('Number of iterations for stability with N = %d and Ld=%f, RPYMob=%d, ImpFP=%d' %(N,Ld,RPYMob,ImpFP)) 
                print(Lits)          
                Iters.append(Lits.copy());

            print('Number of iterations for stability with N = %d' %N)        
            print(Iters.copy())

