# SlenderBody
Slender-body hydrodynamics

This repository contains the Python/C++ and Matlab codes for the publications:
* [1] "An integral-based spectral method for inextensible slender fibers in
Stokes flow," by Ondrej Maxian, Alex Mogilner, and Aleksandar Donev, Jan. 2021.
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
* [4] "The hydrodynamics of a twisting, bending, inextensible fiber in Stokes flow," 
by O. Maxian, B. Sprinkle, C.S. Peskin, and A. Donev. Submitted to Phys. Rev. Fluids, Jan. 2022. 
See [arxiv](https://arxiv.org/abs/2201.04187) for text. 

Organization is as follows:
* Python: directory with python codes
* Python/cppmodules: directory with C++ modules linked to python using pybind11 (compile with included makefile)
* Python/Examples: python scripts to reproduce the tests and examples in the paper. See the README there for more information. 
* Python/Dependencies: contains all dependencies for the python codes. Some are submodules, while some are modified copies of the libraries. 
* Matlab: directory with matlab codes. These are the only codes used in [4]. 

# External dependencies:
* [FINUFFT](https://github.com/flatironinstitute/finufft). I have modified some of the v 1.0 code and included the raw code in Python/Dependencies. 
* [krypy](https://github.com/andrenarchy/krypy).  I have modified some of this and included the raw code in Python/Dependencies.
* [scipy](https://github.com/scipy/scipy)
* LaPack (for C++ functions)
* [PyBind11](https://github.com/pybind/pybind11) (to link python and C++)
* [numba](https://github.com/numba/numba) (to accelerate and parallelize native python)
* [UAMMD](https://github.com/RaulPPelaez/UAMMD) (for Ewald splitting on the GPU). This is used as a submodule in Python/Dependencies. 

For nearly singular SBT integrals, we use a modified version of the quadrature scheme of Ludvig af Klinteberg and 
Alex Barnett. Their original code is [here](https://github.com/ludvigak/linequad); we have made some modifications 
to switch their Legendre discretization to a Chebyshev one in [Python/cppmodules/SpecialQuadratures.cpp](https://github.com/stochasticHydroTools/SlenderBody/blob/master/Python/cppmodules/SpecialQuadratures.cpp)
The code here is independent of the linequad code of af Klinteberg and Barnett. 

# Instructions for running code 
1) Clone this repo using 
```
git clone --recursive https://github.com/stochasticHydroTools/SlenderBody
```
This will also clone the submodules in the dependency folder. 
2) cd into the Python and run make. This will compile all the dependencies and the C++ modulues
3) Common mistakes:
* Make sure you include the directory containing the dependencies in your python path. This can be done by adding
```
export PYTHONPATH=${PYTHONPATH}:/HomeDir/SlenderBody/Python/cppmodules
export PYTHONPATH=${PYTHONPATH}:/HomeDir/SlenderBody/Python/Dependencies/BatchedNBodyRPY
export PYTHONPATH=${PYTHONPATH}:/HomeDir/SlenderBody/Python/Dependencies/UAMMD_PSE_Python
export PYTHONPATH=${PYTHONPATH}:/HomeDir/SlenderBody/Python/Dependencies/NeighborSearch
export PYTHONPATH=${PYTHONPATH}:/HomeDir/SlenderBody/Python/Dependencies
```
to your .bashrc file, for example.
* When compiling UAMMD in the dependencies, the compiler make complain depending on what version of nvcc you use.
If it complains that there is no file called cub, cd into SlenderBody/Python/Dependencies/UAMMD_PSE_Python/uammd/src/third_party
and change the folder called "cub_bak" to "cub." This should fix that issue. See also the [UAMMD webpage](https://github.com/RaulPPelaez/UAMMD) 
for more compilation instructions. 
* You will need to add the location of the fortran module SlenderBody/Fortran/MinHeapModule.f90 to your LD_LIBRARY_PATH. Specifically, the makefile
in cppmodules will compile the fortran code. You need to make sure your system knows where that .so file is to load it in. For me this is
```
export LD_LIBRARY_PATH=/home/ondrejmaxian/lib
```
4) Run the python scripts in Python/Examples. For example, 
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
