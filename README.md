# SlenderBody
Slender-body hydrodynamics

This repository contains the Python/C++ codes for
"An integral-based spectral method for inextensible slender fibers in
Stokes flow," by Ondrej Maxian, Alex Mogilner, and Aleksandar Donev. 

Organization is as follows:
* Python: directory with python codes
* Python/cppmodules: directory with C++ modules linked to python using pybind11 (compile with included makefile)
* Python/Examples: python scripts to reproduce the tests and examples in the paper. See the README there for more information. 
* Python/ModifiedFiles: contains files that replace the corresponding ones in the external libraries. See the README there for more information. 

# External dependencies:
* [FINUFFT](https://github.com/flatironinstitute/finufft)
* [krypy](https://github.com/andrenarchy/krypy)
* [scipy](https://github.com/scipy/scipy)
* LaPack (for C++ functions)
* [PyBind11](https://github.com/pybind/pybind11) (to link python and C++)

# Instructions for running code 
1) Download FINUFFT and krypy and follow instructions in Python/ModifiedFiles
to properly modify them
2) Compile FINUFFT using instructions [here](https://finufft.readthedocs.io/en/latest/install.html)
3) Install lapack and pybind11 to compile C++ code
4) Compile the C++ modules using the provided makefile. If you install using pip, the 
--includes flag in the makefile will find the pybind11 path on its own. 
5) Run the python scripts in Python/Examples. For example, 
```
python3 ThreeShearedFibs.py
```
will run the example in Section 5.1.2 of the paper. 
