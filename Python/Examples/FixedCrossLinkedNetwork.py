from math import pi
from fiberCollection import fiberCollection
from FibCollocationDiscretization import ChebyshevDiscretization
from RPYVelocityEvaluator import EwaldSplitter
from Domain import PeriodicShearedDomain
from TemporalIntegrator import CrankNicolson
from CrossLinkedNetwork import KMCCrossLinkedNetwork
from FileIO import prepareOutFile, writeArray
import numpy as np

"""
FixedCrossLinkedNetwork.py 
This file runs the cross linked network to steady state. 
Chebyshev point locations, CL strains, and fiber curvatures
are printed to a file. 
"""

def saveCurvaturesAndStrains(nFib,nCL,allFibers,CLNet,rl,wora='a'):
    Xf = allFibers.getX();
    LinkStrains = CLNet.calcLinkStrains(allFibers.getUniformPoints(Xf), Dom);
    FibCurvatures = allFibers.calcCurvatures(Xf);
    if (nCL > 0):
        writeArray('SSLinkStrainsF'+str(nFib)+'C'+str(nCL)+'rl'+str(rl)+'.txt',LinkStrains,wora=wora)
    writeArray('SSFibCurvesF'+str(nFib)+'C'+str(nCL)+'rl'+str(rl)+'.txt',FibCurvatures,wora=wora)

import sys
sys.path.append("/NetworkSteadyStates/")    

# Inputs for the slender body simulation
nFib = 100;                     # number of fibers
nCL = 12*nFib;                  # maximum # of CLs
N=16                            # number of points per fiber
Lf=2                            # length of each fiber
Ld=4                            # length of the periodic domain
nonLocal=0;                     # 0 for local drag, 1 for nonlocal hydro
xi = 0.5*(N*nFib)**(1/3)/Ld;    # Ewald param
mu=1                            # fluid viscosity
eps=1e-3                        # slenderness ratio
Eb=0.01                         # fiber bending stiffness
omega=0                         # frequency of oscillations
gam0=0                          # base rate of strain
tf = 1000;                      # stop time
dt = 5e-3;                      # timestep
saveEvery = 200;                # every second
grav=0                          # value of gravity if it exists
Kspring=5                      # cross linker stiffness
rl=0.2                          # cross linker rest length

# We don't have dynamic cross linking, what we do is bind links 
# at t = 0 by calling the dynamic method with high binding rate and 
# zero unbinding rate
konCL = 1000;                   # cross linker binding rate
koffCL = 1e-16;                 # cross linker unbinding rate

np.random.seed(1);

# Initialize the domain and spatial database
Dom = PeriodicShearedDomain(Ld,Ld,Ld);

# Initialize Ewald for non-local velocities
Ewald = EwaldSplitter(np.exp(1.5)/4*eps*Lf,mu,xi,Dom,N*nFib);

# Initialize fiber discretization
fibDisc = ChebyshevDiscretization(Lf,eps,Eb,mu,N,deltaLocal=0.1);

# Initialize the master list of fibers
allFibers = fiberCollection(nFib,fibDisc,nonLocal,mu,omega,gam0,Dom,nThreads=4);

# Initialize the fiber list (straight fibers)
fibList = [None]*nFib;
#print('Loading from steady state')
#XFile = 'SSLocsF'+str(nFib)+'C'+str(nCL)+'REV.txt';
#XsFile = 'SSTanVecsF'+str(nFib)+'C'+str(nCL)+'REV.txt';
allFibers.initFibList(fibList,Dom);#,pointsfileName=XFile,tanvecfileName=XsFile);
allFibers.fillPointArrays();

# Initialize the network of cross linkers (set links from file)
# New seed for CLs
CLseed = 2;
np.random.seed(CLseed);
CLNet = KMCCrossLinkedNetwork(nFib,N,fibDisc.getNumUniform(),Lf,nCL,Kspring,rl,konCL,koffCL,CLseed,Dom,fibDisc,nThreads=4);
CLNet.updateNetwork(allFibers,Dom,0.01)
ofCL = prepareOutFile('F'+str(nFib)+'C'+str(nCL)+'rl'+str(rl)+'.txt');
CLNet.writeLinks(ofCL,allFibers.getUniformPoints(allFibers._ptsCheb),Dom)
#CLNet.setLinksFromFile('F'+str(nFib)+'C'+str(nCL)+'.txt',Dom);
    
# Initialize the temporal integrator
TIntegrator = CrankNicolson(allFibers, CLNet);
TIntegrator.setMaxIters(1); # only 1 iteration since we do local drag

# Prepare the output file and write initial locations
of = prepareOutFile('PermanentLinkLocsF'+str(nFib)+'C'+str(nCL)+'rl'+str(rl)+'.txt');
allFibers.writeFiberLocations(of);
saveCurvaturesAndStrains(nFib,nCL,allFibers,CLNet,rl,wora='w')

# Run to steady state
stopSS = int(tf/dt+1e-10);
for iT in range(stopSS): 
    wr=0;
    if ((iT % saveEvery) == (saveEvery-1)):
        print('Time %1.2E' %(float(iT)*dt));
        wr=1;
    maxX = TIntegrator.updateAllFibers(iT,dt,stopSS,Dom,Ewald,grav/Lf,of,write=wr);
    if (wr==1): # save curvatures and strains
        saveCurvaturesAndStrains(nFib,nCL,allFibers,CLNet,rl)
        print('Max x (just to check stability): %f' %(maxX));

#ofFinalX = prepareOutFile('SSLocsF'+str(nFib)+'C'+str(nCL)+'REV2.txt');
#allFibers.writeFiberLocations(ofFinalX);
#ofFinalXs = prepareOutFile('SSTanVecsF'+str(nFib)+'C'+str(nCL)+'REV2.txt');
#allFibers.writeFiberTangentVectors(ofFinalXs);
 
# Destruction and cleanup
of.close();
del Dom;
del Ewald;
del fibDisc;
del allFibers;
del TIntegrator; 



