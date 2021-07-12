# Examples 
This folder contains the main python files to run the examples
in the papers: [1] "An integral-based spectral method for inextensible slender fibers 
in Stokes flow." [Arxiv link](https://arxiv.org/abs/2007.11728) and [2] "Simulations of dynamically cross-linked actin networks: morphology, rheology, and hydrodynamic interactions" [bioarxiv link](bioRxiv.org:2021.07.07.451453)

# Tests from [1]:
1) ThreeShearedFibs.py corresponds to the 3 fibers being sheared in Section 5.1.2.
2) CheckStability.py gives the stability test of Section 5.2 (for a specific domain length Ld)
3) FixedCrossLinkedNetwork.py runs a cross linked network to steady state (t=2000), as we do in
Section 6.3.2
4) StrainingCrossLinkedNetwork.py is the actual dynamic simulation (Section 6.3.2) that measures the 
stress for a specific omega. We load in the steady state dynamics from previous simulations in the 
folder NetworkSteadyStates. 
5) HexagonalRPYTest.py reproduces the results in Appendix B for the same set of particles viewed in
a rectangular and parallelepiped way. 

# Input files for straining network
Because we start a simulation for one frequency (e.g., omega = 1 Hz) using the locations from the 
next smallest frequency (e.g., from omega = 0.5 Hz), input files are required for some of the simulations. 

All of the input files are contained in [NetworkSteadyStates.](https://github.com/stochasticHydroTools/SlenderBody/tree/master/Python/Examples/NetworkSteadyStates)
The general idea is that each frequency in the [list omHzs](https://github.com/stochasticHydroTools/SlenderBody/blob/4cf402e21404ad8b9589af1de5b652adfbb1f72e/Python/Examples/StrainingCrossLinkedNetwork.py#L45)
is associated with an input file from the previous frequency. For example, the input files DynamicSSLocationsOm0.5F700C8400.txt
and DynamicTangentVecsOm0.5F700C8400.txt are used to load the steady state locations and tangent vectors from omega = 0.5 Hz that 
initialize the configuration for omega = 1 Hz. When omega = 0.01 Hz (the smallest frequency we simulate), the associated input 
files are SSLocationsF700C8400E0.01.txt and TangentVecsF700C8400E0.01.txt, which are the configurations coming from steady state 
equilibration. The input file handles are passed to a fiberCollection object 
in the code that begins on [line 86 of StrainingCrossLinkedNetwork.py](https://github.com/stochasticHydroTools/SlenderBody/blob/4cf402e21404ad8b9589af1de5b652adfbb1f72e/Python/Examples/StrainingCrossLinkedNetwork.py#L86).

We also include an input file to [initialize the cross-linked network object.](https://github.com/stochasticHydroTools/SlenderBody/blob/4cf402e21404ad8b9589af1de5b652adfbb1f72e/Python/Examples/StrainingCrossLinkedNetwork.py#L101)
This file, NetworkSteadyStates/F700C8400.txt, has a list of connections and periodic shifts at g=0 for the 8400 links. 

# Tests from [2]:
The results in [2] can be broadly grouped into three categories
* Steady state networks with no shear. These tests can be run by executing the file FixedDynamicLinkNetwork.py, which takes an integer command line argument with a random seed. For example, "FixedDynamicLinkNetwork.py 1" runs the network evolution with the parameters listed in 'FixedNetworkInputFile.txt.' By specifying the OutputFileName, the simulation will output statistics prefaced by this name, and a set of dynamic steady state locations, fiber tangent vectors, and links that can be used to initialize the next set of two simulations. 
* Stress relaxation tests, where a shear is applied for one quarter of a period and then the network relaxes. These tests can be run by executing  "MeasureG0.py," which takes three command line arguments: the seed, frequency of strain, and maximum strain. So for instance, "MeasureG0.py 1 2 0.1" would shear the cell to a 10\% strain with frequency 2 Hz, using a random seed of 1. The parameters are set in the file "StrainInputFile.txt." The name inFile (on line 38 of the input file) should match the OutputFileName of the steady state network generated above. 
* Measuring elastic and viscous moduli. Here we use the file "StrainDynamicNetwork.py," with the same three command line arguments and input file as for the stress relaxation tests. 

The simulations will output files prefixed with the user-specified outFile string (specified on line 37 of StrainInputFile.txt). The statistics are output every ``savedt'' seconds, where the interval savedt is specified on line 35 of the input file. Outputs include: 
* numLinksByFib = number of links attached to each fiber, 
* NumFibsConnectedPerFib = number of fibers connected to each fiber by at least one link
* LocalAlignmentPerFib = the alignment parameter for the group of fibers connected by at least one link to each fiber
* NumberOfBundles = raw number of bundles in the system
* BundleOrderParams = the order parameter for each of the bundles
* NFibsPerBundle = number of fibers in each bundle
* FinalLabels = tags for the fibers for each time step. These tags give which bundle each fiber belongs to and are used for visualization
* AvgBundleTangents = the average tangent vector in each bundle 
* AvgTangentVectors = the average tangent vectors for all fibers
* LamStress, ElStress, CLStress = the stress in the suspension due to the inextensibility forces, elastic (bending) forces, and CL forces
