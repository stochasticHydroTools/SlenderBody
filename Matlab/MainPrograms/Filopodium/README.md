# MATLAB Code for filopodium simulations
These are standalone Matlab codes for the simulations of filopodia that appear in
"Helical motors and formins synergize to compact chiral filopodial bundles: a theoretical perspective," by O. Maxian and A. Mogilner, Oct. 2023.
See [biorxiv](https://www.biorxiv.org/content/10.1101/2023.07.24.550422) for text.

The simulations can be run and analyzed via the following files
1) FiberBundle.m - the main driver file where parameters are specified
2) FiloExtFT.m - file that computes the external (motor and cross linking) forces and torques on the filaments at each time step
3) RecomputeDiscMats.m - file that recomputes the discretization matrices (for polymerizing fibers) at each time step
4) TemporalIntegrator_Fil - file that computes the velocity on each filament from the forces and torques
5) AnalyzeTrajector.m and PloyTraj.m - for analysis of the statistics and plotting the filaments

Files related to supercoiling actin filaments are:
1) StabilityThetConst.m - performs linear stability analysis on the model equations
2) TwistFiber.m - simulation of (potentially supercoiling) single actin filament with fixed twist density over time
