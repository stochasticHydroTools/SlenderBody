# Matlab codes for slender fibers
These are standalone Matlab codes for slender body hydrodynamics. 

# Main programs
1) SBTGravity.m simulates four falling fibers in free space. This is Section 
5.1.1 in the paper "An integral-based spectral method for inextensible slender fibers 
in Stokes flow." [Arxiv link](https://arxiv.org/abs/2007.11728). 
2) SBTThreeFibs.m simulates three fibers being sheared in a perioid domain. This is 
Section 5.1.2 in the paper, but we caution that results will be different (although to 
lower tolerances than spatio-temporal errors) due to slight differences in nonlocal hydro 
calculations in Matlab vs. Python. Use the [python version](https://github.com/stochasticHydroTools/SlenderBody/blob/master/Python/Examples/ThreeShearedFibs.py)
to exactly reproduce results in the paper. 
3) SBTCLs. simulates a cross-linked mesh with solid-like stress behavior. 

# Matlab functions
See the functions-solvers folder, which is only partially documented, for other codes (not main). 
