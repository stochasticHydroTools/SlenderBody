This folder contains the main python files to run the examples
in the paper "An integral-based spectral method for inextensible slender fibers 
in Stokes flow." 

1) ThreeShearedFibs.py corresponds to the 3 fibers being sheared in Section 5.1.2.
2) CheckStability.py gives the stability test of Section 5.2 (for a specific domain length Ld)
3) FixedCrossLinkedNetwork.py runs a cross linked network to steady state (t=2000), as we do in
Section 6.3.2
4) StrainingCrossLinkedNetwork.py is the actual dynamic simulation (Section 6.3.2) that measures the 
stress for a specific omega. We load in the steady state dynamics from previous simulations in the 
folder NetworkSteadyStates. 