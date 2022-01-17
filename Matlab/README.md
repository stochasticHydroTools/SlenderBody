# Matlab codes for slender fibers
These are standalone Matlab codes for slender body hydrodynamics. These matlab codes are used primarily
in [4] (see main repo page for references), but can also be used for some of the examples in [1]. 
In particular, this repository contains the codes necessary to run the dynamic examples in [4]:

# Main programs
1) SBTRelaxingFib.m simulates a relaxing fiber with nonzero twist modulus (Section 5.2 in [4])
2) WhirlingFibInstability.m simulates a fiber spinning about its axis (to generate the twirling-whirling
instability; Section 5.3 in [4])
3) SBTGravity.m simulates four falling fibers in free space. This is Section 
5.1.1 in the paper "An integral-based spectral method for inextensible slender fibers 
in Stokes flow." [Arxiv link](https://arxiv.org/abs/2007.11728). 
4) SBTThreeFibs.m simulates three fibers being sheared in a perioid domain. This is 
Section 5.1.2 in [1], and is equivalent to the [python version](https://github.com/stochasticHydroTools/SlenderBody/blob/master/Python/Examples/ThreeShearedFibs.py)
up to errors in the solvers (GMRES for first 2 iterations, nonlocal hydro tolerances, etc.). Also, the matlab code
only does direct upsampled quadrature.

