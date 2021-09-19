# SlenderBody
Slender-body hydrodynamics

This repository contains the Python/C++ codes for the publications:
* [1] "An integral-based spectral method for inextensible slender fibers in
Stokes flow," by Ondrej Maxian, Alex Mogilner, and Aleksandar Donev, Jan. 2021.
See [arxiv](https://arxiv.org/abs/2007.11728) for text and [Phys. Rev. Fluids](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.014102) for published
version
* [2] "Simulations of dynamically cross-linked actin networks: morphology, rheology, and hydrodynamic interactions" by O. Maxian, R. P. Peláez, A. Mogilner, and A. Donev, submitted to PLOS Comp. Bio., June 2021. See [bioarxiv](https://www.biorxiv.org/content/10.1101/2021.07.07.451453v1) for text
* [3] "Interplay between Brownian motion and cross-linking kinetics controls bundling dynamics in actin networks" by O. Maxian, A. Donev, and A. Mogilner. submitted to Biophysical Journal, September 2021. See [bioarxiv](https://www.biorxiv.org/content/10.1101/2021.09.17.460819v1) for text

Organization is as follows:
* Python: directory with python codes
* Python/cppmodules: directory with C++ modules linked to python using pybind11 (compile with included makefile)
* Python/Examples: python scripts to reproduce the tests and examples in the paper. See the README there for more information. 
* Python/ModifiedFiles: contains files that replace the corresponding ones in the external libraries. See the README there for more information. 
* Matlab: directory with matlab codes (these are only partially documented)

# External dependencies:
* [FINUFFT](https://github.com/flatironinstitute/finufft)
* [krypy](https://github.com/andrenarchy/krypy)
* [scipy](https://github.com/scipy/scipy)
* LaPack (for C++ functions)
* [PyBind11](https://github.com/pybind/pybind11) (to link python and C++)
* [numba](https://github.com/numba/numba) (to accelerate and parallelize native python)
* [UAMMD](https://github.com/RaulPPelaez/UAMMD) (for Ewald splitting on the GPU)

For nearly singular SBT integrals, we use a modified version of the quadrature scheme of Ludvig af Klinteberg and 
Alex Barnett. Their original code is [here](https://github.com/ludvigak/linequad); we have made some modifications 
to switch their Legendre discretization to a Chebyshev one in [Python/cppmodules/SpecialQuadratures.cpp](https://github.com/stochasticHydroTools/SlenderBody/blob/master/Python/cppmodules/SpecialQuadratures.cpp)
The code here is independent of the linequad code of af Klinteberg and Barnett. 

# Instructions for running code 
1) Download FINUFFT and krypy and follow instructions in Python/ModifiedFiles
to properly modify them
2) Compile FINUFFT using instructions [here](https://finufft.readthedocs.io/en/latest/install.html)
3) Install lapack and pybind11 to compile C++ code
4) Compile the C++ modules using the provided makefile. If you install using pip, the 
--includes flag in the makefile will find the pybind11 path on its own. 
5) Install numba to compile the parallelized python code in Python/FiberUpdateNumba.py
6) Run the python scripts in Python/Examples. For example, 
```
python3 ThreeShearedFibs.py
```
will run the example in Section 5.1.2 of [1] 

# Parallelization
There are three portions of our code that are parallelized. We first note that the number of OpenMP threads
(environment variable) MUST be set to one to obtain good performance. In particular, you must use 
```
export OMP_NUM_THREADS=1
```
in linux prior to running our code.
The parallelization is then implemented in python in the following three ways:
1) The nonlocal velocity calculations (Ewald splitting) and near fiber corrections, are parallelized \
within C++ using OpenMP. The number of threads in these calculations can be set by passing an integer \
to the constructor of fiberCollection.py. An example of this is on [line 49 of Python/Examples/CheckStability.py](https://github.com/stochasticHydroTools/SlenderBody/blob/990fc394a7c0341d38b3bc809a52991353e88f2e/Python/Examples/CheckStability.py#L49). 
2) The force and stress calculations for cross-linking are parallelized within C++ using OpenMP. \
The number of threads in these calculations can be set by passing an integer to the contructor of \
CrossLinkedNetwork.py (and objects which inherit from it). See [Python/Examples/FixedCrossLinkedNetwork.py, line 78](https://github.com/stochasticHydroTools/SlenderBody/blob/77224b963c0e5b4d6344b8d7b644acca0f3a0fa9/Python/Examples/FixedCrossLinkedNetwork.py#L78), for an example. 
3) The linear solves on all fibers are parallelized using numba. The number of numba threads can be set \
on the command line in linux using (for example, to obtain 4 threads)
```
export NUMBA_NUM_THREADS=4
```
