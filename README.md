# SlenderBody
Slender-body hydrodynamics

This repository contains the Python/C++ codes for
"An integral-based spectral method for inextensible slender fibers in
Stokes flow," by Ondrej Maxian, Alex Mogilner, and Aleksandar Donev. 

Organization is as follows:
Python: directory with python codes
Python/cppmodules: directory with C++ modules linked to python using pybind11
Python/Examples: python scripts to reproduce 
the tests and examples in the paper. See the README there for more information. 
Python/ModifiedFiles: contains files that replace the corresponding ones in the
external libraries. See the README there for more information. 

External dependencies (Python):
[FINUFFT](https://github.com/flatironinstitute/finufft)
[krypy](https://github.com/andrenarchy/krypy)
LaPack (for C++ functions)
PyBind11 (to link python and C++)

To run our codes:
1) Download FINUFFT and krypy and follow instructions in Python/ModifiedFiles
to properly modify them
2) Compile the C++ modules using the provided makefile (for linux, assumes PyBind11 installed)
3) Run the python scripts in Python/Examples
