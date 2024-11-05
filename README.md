# SPHARCLE
Slender Particle Hydrodynamics And Reacting Cytoskeletal Linking Elements.

See the [documentation](https://slenderbody.readthedocs.io/en/latest/) on readthedocs for installation instructions, as well as how to run and modify the examples. 

This repository contains the Python/C++ and Matlab codes for the publications:
1. (Leading reference where all theory is summarized) "A simulation platform for slender, semiflexible, and inextensible fibers with Brownian hydrodynamics and steric repulsion," by O. Maxian and A. Donev, Nov. 2024. See [arxiv](https://arxiv.org/abs/2408.15913) for text. 
2. "An integral-based spectral method for inextensible slender fibers in
Stokes flow," by O. Maxian, A. Mogilner, and A. Donev, Jan. 2021.
See [arxiv](https://arxiv.org/abs/2007.11728) for text and 
[Phys. Rev. Fluids](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.6.014102) for published
version.
3. "Simulations of dynamically cross-linked actin networks: morphology, rheology, and hydrodynamic interactions," 
by O. Maxian, R. P. Peláez, A. Mogilner, and A. Donev, Dec. 2021. 
See [biorxiv](https://www.biorxiv.org/content/10.1101/2021.07.07.451453) for text and 
[PLoS Comp. Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009240)
for published version.
4. "Interplay between Brownian motion and cross-linking kinetics controls bundling dynamics in actin networks," 
by O. Maxian, A. Donev, and A. Mogilner, April 2022. 
See [biorxiv](https://www.biorxiv.org/content/10.1101/2021.09.17.460819) for text and [Biophys. J.](https://www.cell.com/biophysj/fulltext/S0006-3495(22)00154-0) for 
published version.
5. "The hydrodynamics of a twisting, bending, inextensible fiber in Stokes flow," 
by O. Maxian, B. Sprinkle, C.S. Peskin, and A. Donev, July 2022. 
See [arxiv](https://arxiv.org/abs/2201.04187) for text and [Phys. Rev. Fluids](https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.7.074101) for published
version.
6. "Slender body theories for rotating filaments," by O. Maxian and A. Donev, Nov. 2022. See [J. Fluid Mech.](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/slender-body-theories-for-rotating-filaments/0A9E1AB691DC4AFDB57C6471928745AE) for published version, which is open access. This is primarily a theoretical study which has some numerical calculations. The relevant files are [here](https://github.com/stochasticHydroTools/SlenderBody/tree/master/Matlab/NumericalSBT/SingleLayer), but we recommend using the [more accurate quadrature scheme](https://github.com/dmalhotra/CSBQ) by Malhotra and Barnett for slender body boundary integral calculations.
7. "Bending fluctuations in semiflexible, inextensible, slender filaments in Stokes flow: towards a spectral discretization," by O. Maxian, B. Sprinkle, and A. Donev, Jan. 2023. See [arxiv](https://arxiv.org/abs/2301.11123) for text and [J. Chem. Phys](https://pubs.aip.org/aip/jcp/article/158/15/154114/2884532/Bending-fluctuations-in-semiflexible-inextensible) for published version. 
8. "Helical motors and formins synergize to compact chiral filopodial bundles: a theoretical perspective," by O. Maxian and A. Mogilner, Nov. 2023. See [biorxiv](https://www.biorxiv.org/content/10.1101/2023.07.24.550422) for text and [Eur. J. Cell Biol](https://www.sciencedirect.com/science/article/pii/S0171933523000985) for published version. The codes for this publication are [in this subfolder](https://github.com/stochasticHydroTools/SlenderBody/tree/master/Matlab/MainPrograms/Filopodium). 

Organization is as follows:
* Python: directory with python codes
* Python/cppmodules: directory with C++ modules linked to python using pybind11 (compile with included makefile)
* Python/Examples: python scripts to reproduce the tests and examples [1-3] and [6]. See the README there for more information. 
* Python/Dependencies: contains all dependencies for the python codes. Some are submodules, while some are modified copies of the libraries. 
* Matlab: directory with matlab codes. These are the only codes used in [4], [5], and [7]. 

# External dependencies:
* [krypy](https://github.com/andrenarchy/krypy).  I have modified some of this and included the raw code in Python/Dependencies.
* [scipy](https://github.com/scipy/scipy)
* LaPack (for C++ functions)
* [PyBind11](https://github.com/pybind/pybind11) (to link python and C++)
* [UAMMD](https://github.com/RaulPPelaez/UAMMD) (for Ewald splitting on the GPU). This is used as a submodule in Python/Dependencies. 
* [FINUFFT](https://github.com/flatironinstitute/finufft). I have modified some of the v 1.0 code and included the raw code in Python/Dependencies. Note that this dependency is not strictly necessary; it is only needed if you wish to run nonlocal hydrodynamics and do not have a GPU.

# Instructions for running code 
1) Clone this repo using 
```
git clone --recursive https://github.com/stochasticHydroTools/SlenderBody
```
This will also clone the submodules in the dependency folder. Then run 
```
cd SlenderBody
bash install.sh
```
in the main directory. This will compile all of the dependencies and C++ modulues. It will also
add the appropriate directories (where the python modulues are compiled) to your PYTHONPATH.  

If the compilation process fails this script will show an error. In that case fix the compilation issue and run ```bash install.sh``` again.

3) Common mistakes:
* When compiling UAMMD in the dependencies, the compiler may complain depending on what version of nvcc you use.
If it complains that there is no file called cub, cd into SlenderBody/Python/Dependencies/UAMMD_PSE_Python/uammd/src/third_party
and change the folder called "cub_bak" to "cub." This should fix that issue. See also the [UAMMD webpage](https://github.com/RaulPPelaez/UAMMD) 
for more compilation instructions. 

4) Run the python scripts in Python/Examples. For example, 
```
python3 ThreeShearedFibs.py
```
will run the example in Section 5.1.2 of [1] 

# Parallelization
The number of OpenMP threads (environment variable) MUST be set to one to obtain good performance. 
In particular, you must use 
```
export OMP_NUM_THREADS=1
```
in linux prior to running our code.
The parallelization is then implemented in the C++ class FiberCollectionC.cpp which takes as an inpute the number of threads. The number of threads in these calculations can be set by passing an integer to the constructor of fiberCollection.py. An example of this is on [line 49 of Python/Examples/CheckStability.py](https://github.com/stochasticHydroTools/SlenderBody/blob/990fc394a7c0341d38b3bc809a52991353e88f2e/Python/Examples/CheckStability.py#L49). We also have a GPU version. 

# Uninstalling 
1) Run ```make clean``` inside the Python folder.  

2) The install.sh script modifies the .bashrc file (which are marked as written by the script). Simply remove these lines.  
