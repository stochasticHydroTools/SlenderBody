# Inputs for straining test
nFib=200        # number of fibers
N=12            # number of tangent vectors per fiber
Lf=1            # length of each fiber
nonLocal=0;     # doing nonlocal solves? 0 = intra-fiber hydro, 1 = full nonlocal hydro
Ld=2            # length of the periodic domain
mu=0.1          # fluid viscosity
eps=4e-3/Lf     # slenderness ratio
lp = 2*Lf;      # persistence length
kbT = 4.1e-3;   # thermal energy
Eb = kbT*lp     # fiber bending stiffness
dt = 1e-4;      # timestep
omHz = 100    # frequency of oscillations in Hz
maxStrain = 0.05               # maximum strain
numStressPerCycle = 100;       # number of times per cycle we write the stress to file
tf=10/omHz                     # final time (simulate for 10 cycles)
saveEvery = max(1,int(2e-2/dt+1e-10)); # How often do we save? 
seed = 1
nThr = 16;                     # Number of threads for OpenMP
RPYQuad = True;                # Doing special quadrature for mobility
RPYDirect = False;             # Doing direct quadrature for mobility
RPYOversample = False;         # Doing oversampled quadrature for mobility
NupsampleForDirect = 100;      # Number of oversampled points
FluctuatingFibs = False;        # Are there semiflexible bending fluctuations?
RigidDiffusion = False;        # Are the fibers diffusing as rigid bodies every time step?
rigidFibs = False;             # Are the fibers actually rigid and straight the whole time?
smForce = False;               # Are we smoothing out the cross linking force or using a delta function?
deltaLocal = 1;                # Regularization if doing local drag SBT. 
Nuniformsites = 40;            # Number of uniform points for CL binding. 
Kspring = 10                   # cross linker stiffness
rl = 0.05                      # cross linker rest length 
konCL = 5;                     # first end binding rate
konSecond = 50;                # second end binding rate
koffCL = 1;                    # free end unbinding rate
koffSecond = 1;                # bound CL unbinding rate
turnovertime = 2.5;            # mean fiber lifetime
bindingSiteWidth = 20e-3;      # Set to 0 for no limit, 20 nm otherwise
bunddist = 1/4                 # Distance separating 2 links to form a bundle
OutFileString = 'FirstStrainingSimulation.txt'
InFileString = 'FirstBundlingSimulation.txt'
updateNet = True;              # Updating the network at each time step?
turnover = True;               # Turning over the fibers? Turnover time above


