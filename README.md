# SlenderBody
Slender-body hydrodynamics

This repository contains the Python/C++ codes for
"An integral-based spectral method for inextensible slender fibers in
Stokes flow," by Ondrej Maxian, Alex Mogilner, and Aleksandar Donev. 
See [arxiv](https://arxiv.org/abs/2007.11728) for text

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
will run the example in Section 5.1.2 of the paper. 

# Parallelization
There are three portions of our code that are parallelized. We first note that the number of OpenMP threads
(environment variable) MUST be set to one to obtain good performance. In particular, you must use 
```
export OMP_NUM_THREADS=1
```
in linux prior to running our code.
The parallelization is then implemented in python in the following three ways:
* The nonlocal velocity calculations (Ewald splitting) and near fiber corrections, are parallelized \
within C++ using OpenMP. The number of threads in these calculations can be set by passing an integer \
to the constructor of fiberCollection.py. An example of this is on line 49 of Python/Examples/CheckStability.py. 
* The force and stress calculations for cross-linking are parallelized within C++ using OpenMP. \
The number of threads in these calculations can be set by passing an integer to the contructor of \
CrossLinkedNetwork.py (and objects which inherit from it). See Python/Examples/FixedCrossLinkedNetwork.py, \
line 78, for an example. 
* The linear solves on all fibers are parallelized using numba. The number of numba threads can be set \
on the command line in linux using (for example, to obtain 4 threads)
```
export NUMBA_NUM_THREADS=4
```