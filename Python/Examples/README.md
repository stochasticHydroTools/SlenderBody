# Examples 
This folder contains the main python files to run the examples
in the paper "An integral-based spectral method for inextensible slender fibers 
in Stokes flow." [Arxiv](https://arxiv.org/abs/2007.11728)

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

All of the input files are contained in [Python/Examples/NetworkSteadyStates](https://github.com/stochasticHydroTools/SlenderBody/tree/master/Python/Examples/NetworkSteadyStates)
The general idea is that each frequency in the [list omHzs](https://github.com/stochasticHydroTools/SlenderBody/blob/4cf402e21404ad8b9589af1de5b652adfbb1f72e/Python/Examples/StrainingCrossLinkedNetwork.py#L45)
is associated with an input file from the previous frequency. For example, the input files DynamicSSLocationsOm0.5F700C8400.txt
and DynamicTangentVecsOm0.5F700C8400.txt are used to load the steady state locations and tangent vectors from omega = 0.5 Hz that 
initialize the configuration for omega = 1 Hz. When omega = 0.01 Hz (the smallest frequency we simulate), the associated input 
files are SSLocationsF700C8400E0.01.txt and TangentVecsF700C8400E0.01.txt. The input file handles are passed to a fiberCollection object 
in the code that begins on [line 86 of StrainingCrossLinkedNetwork.py](https://github.com/stochasticHydroTools/SlenderBody/blob/4cf402e21404ad8b9589af1de5b652adfbb1f72e/Python/Examples/StrainingCrossLinkedNetwork.py#L86)

We also include an input file to [initialize the cross-linked network object](https://github.com/stochasticHydroTools/SlenderBody/blob/4cf402e21404ad8b9589af1de5b652adfbb1f72e/Python/Examples/StrainingCrossLinkedNetwork.py#L101)
this file, NetworkSteadyStates/F700C8400.txt, has a list of connections and periodic shifts at g=0 for the 8400 links. 