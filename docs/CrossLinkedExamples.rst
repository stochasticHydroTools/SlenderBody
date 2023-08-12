Cross-linked actin networks
========================================
There are three example files that simulate
cross linked actin networks:

    1) FixedDynamicLinkNetwork.py -- simulates a network without measuring stress. This is used
       to generate steady states to perform stress relaxation and small amplitude oscillatory shear
       tests. The input file is SemiflexBundleInputFile.txt. 
    2) MeasureG0.py -- stress relaxation test; requires an input network. The input file is StrainInputFile.txt.
    3) StrainDynamicNetwork.py -- small amplitude oscillatory shear test; also requires an input
       network. The input file is StrainInputFile.txt.
       
The ideal workflow is to first run FixedDynamicLinkNetwork.py, which generates a steady state network. 
This network can then be used to measure stress (MeasureG0.py and StrainDynamicNetwork.py). 

The input files (SemiflexBundleInputFile.txt and StrainInputFile.txt) contain the following variables:
    - nFib = number of fibers
    - N = number of tangent vectors per fiber
    - Lf = length of each fiber
    - nonLocal = nature of hydrodynamics. Setting to 0 will give intra-fiber hydrodynamics, while 1 
      will give hydrodynamics between all fibers.
    - Ld = length of periodic domain
    - mu = fluid viscosity
    - eps = fiber aspect ratio
    - lp = fiber persistence length
    - kbT = thermal energy
    - Eb = fiber bending stiffness
    - dt = time step size
    - omega = frequency of box oscillations
    - gam0 = base rate of strain (the maximum strain is $\gamma_0/\omega$)
    - tf = final time
    - saveEvery = how often (in integer number of time steps) to save statistics
    - seed = random seed (simulation will only be reproducible if you use 1 thread however)
    - nThr = number of OpenMP threads for CPU parallel calculations
    - RPYQuad = whether to use special quadrature for the self-fiber mobility. This is only an 
      option in simulations without thermal fluctuations, or simulations without nonlocal
      hydrodynamics (nonLocal=0). If you try to run nonLocal=1 with RPYQuad=True, the code 
      will stop with an error. The reason for this is we do not know how to consistently 
      generate fluctuations in this case. 
    - RPYDirect = whether to instead use direct quadrature (just direct RPY sums over the $N_x=N+1$
      Chebyshev points). This mobility works for all flucutation and hydrodynamic options, but is 
      not accurate for slender fibers.
    - RPYOversample = whether to use oversampled RPY quadrature for the self mobility. This mobility
      is the preferred one for fluctuations, as it is guaranteed SPD, and works for all flucutation 
      and hydrodynamic options. It becomes expensive if you try to take the aspect ratio down, because
      you'll have to take the number of oversampled points down. 
    - NupsampleForDirect = number of oversampling points if using RPYOversample=True. If RPYOversample=False,
      this variable still matters, as it sets the number of points used for nonlocal integrals. If RPYOversample=True,
      then is sets the number of points for the self integral as well. For accuracy, we have found $N_u = 0.4/\epsilon$
      it a good number, where $\epsilon$ is the fiber aspect ratio. 
    - deltaLocal = regularization for local drag SBT. If you set all of RPYQuad, RPYDirect, and RPYOversample to false,
      the default mobility is slender body theory with regularized local drag coefficients. This coefficient is the 
      dimensionless strength of the regularization (the lengthscale on which the fiber radius function decays ellipsoidally). 
      Set to 1 for ellipsoidal fibers. If you set one of RPYQuad, RPYDirect, and RPYOversample to true, this variable has 
      no effect. 
    - FluctuatingFibs = whether the fibers have semiflexible bending fluctuations
    - RigidDiffusion = whether the fibers diffuse as rigid bodies. This can be useful if you want to check if 
      semiflexible bending fluctuations are important, as you can update the bending modes deterministically and set 
      this option to true to thermally force only rotation and translation.
    - rigidFibs = whether the fibers are actually rigid bodies. 
    - smForce = whether to smooth out the cross linking force on the fibers. 
    - Sterics = whether we include excluded-volume interactions to keep the fibers apart. At present, steric
      interactions are only implemented in FixedDynamicLinkNetwork.py (because we have not included the 
      steric forces in the stress calculation). 
    - NsegForSterics = number of segments we use to estimate whether fibers are touching. If set to 0, the code
      will use a (slightly) less efficient sphere-based sterics algorithm.
    - Nuniformsites = number of uniform sites we sample the fibers on for cross linking.
    - Kspring = cross linker spring stiffness
    - rl = cross linker rest length
    - updateNet = whether the network is static (false) or dynamic (true)
    - konCL = cross linker on rate (for a single end)
    - konSecond = on rate for the second end of a cross linker if the first end is already bound.
    - koffCL = cross linker off rate (for a CL with one bound end)
    - koffSecond = cross linker off rate (for one end when both ends are bound)
    - turnovertime = mean fiber lifetime
    - turnover = whether the fibers do turnover (which models (de)polymerization)
    - bindingSiteWidth = width of binding sites, which limits the number of bound CLs. Set to zero for no 
      limit on the number of CLs bound to each site. 
    - bunddist = distance separating two links to form a bundle
    - (Out)FileString = the string associated with the output file that will write the network configuration. 
    - (In)FileString = the string associated with the input file that initializes the network (required for MeasureG0.py 
      and StrainDynamicNetwork.py)
 
The output files will all have the form (OUTPUTNAME)(OutFileString).txt. The following are the different OUTPUTNAMEs:
    - AvgTangents = the average tangent vectors for all fibers
    - BundleOrderParams = the order parameter for each of the bundles
    - FibCurvesF = curvatures of each fiber
    - ItsNeeded = number of GMRES iterations needed at each time step (zero
      for simulations that don't have hydrodynamics)
    - LinkStrains = the total link strain 
    - LocalAlignmentPerFib = the alignment parameter for the 
      group of fibers connected by at least one link to each fiber
    - Locs = the fiber locations (to make a movie)
    - nContacts = number of contacts between fibers (to check how many
      overlaps there are)
    - NFibsPerBundle_Sep = number of fibers in each bundle
    - NumberOfBundles = the number of bundles per time step. The sum of this 
      array is the number of entries in the arrays BundleOrderParams, and 
      NFibsPerBundle_Sep
    - nLinksPerFib = number of links attached to each fiber
    - NumFibsConnectedPerFib = number of fibers connected to each fiber 
      by at least one link
    - A copy of the input file
    - LamStress, ElStress, CLStress = the stress in the suspension 
      due to the inextensibility forces, elastic (bending) forces, and CL forces.
      There is also an extra array called DriftStress which is used
      to write the additional stress due to thermal fluctuations. This is not
      operational yet

It also contains these files to initialize the next simulation: 
    - FinalFreeLinkBound = the number of single ends bound to each site 
    - FinalLabels_Sep = the labels of each of the fibers (to tell what bundle
      they are in)
    - FinalLinks = a list of the bound CLs and their offsets
    - FinalLocs = the final locations of the fibers
