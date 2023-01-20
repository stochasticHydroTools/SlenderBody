# Examples 
This folder contains the main python files to run the examples
in the papers: 
* [1] "An integral-based spectral method for inextensible slender fibers in
Stokes flow," by O. Maxian, A. Mogilner, and A. Donev, Jan. 2021.
See [arxiv](https://arxiv.org/abs/2007.11728) for text and 
[Phys. Rev. Fluids](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.014102) for published
version.
* [2] "Simulations of dynamically cross-linked actin networks: morphology, rheology, and hydrodynamic interactions," 
by O. Maxian, R. P. Peláez, A. Mogilner, and A. Donev, Dec. 2021. 
See [bioarxiv](https://www.biorxiv.org/content/10.1101/2021.07.07.451453v3) for text and 
[PLoS Comp. Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009240)
for published version.
* [3] "Interplay between Brownian motion and cross-linking kinetics controls bundling dynamics in actin networks," 
by O. Maxian, A. Donev, and A. Mogilner. Biophysical Journal, April 2022. 
See [bioarxiv](https://www.biorxiv.org/content/10.1101/2021.09.17.460819v2) for text and [BJ](https://www.cell.com/biophysj/fulltext/S0006-3495(22)00154-0) for 
published version.
* [6] "Semiflexible bending fluctuations in inextensible slender filaments in Stokes flow: towards a spectral discretization," by O. Maxian, B. Sprinkle, and A. Donev, Jan. 2022. See [arxiv](link) for text. 


# Tests from [1]:
1) ThreeShearedFibs.py corresponds to the 3 fibers being sheared in Section 5.1.2.
2) CheckStability.py gives the stability test of Section 5.2 (for a specific domain length Ld)
3) HexagonalRPYTest.py reproduces the results in Appendix B for the same set of particles viewed in
a rectangular and parallelepiped way. 
4) *We have removed files associated with the cross linked networks of [1], as they were not compatible with the updates made
in [2] and [3]. For instructions on running cross linked networks, see below.

# Simulations from [2] and [3]:
The results in [2] and [3] can be broadly grouped into three categories
* Steady state networks with no shear (this includes ALL simulations in [3]). These tests can be run by executing the file FixedDynamicLinkNetwork.py, which takes an integer command line argument with a random seed. For example, "FixedDynamicLinkNetwork.py 1" runs the network evolution with the parameters listed in 'FixedNetworkInputFile.txt.' By specifying the OutputFileName, the simulation will output statistics prefaced by this name, and a set of dynamic steady state locations, fiber tangent vectors, and links that can be used to initialize the next set of two simulations.
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
