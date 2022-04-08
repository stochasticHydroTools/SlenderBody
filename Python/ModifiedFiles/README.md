This folder contains the files that were modified from external libraries. There are 
two main modifications. 

1) The file [linsys.py](https://github.com/andrenarchy/krypy/blob/master/krypy/linsys.py) in 
the krypy library was modified to remove an extraneous mobility application. To make use of this, 
download krypy and replace linsys.py with the version in this folder.

2) FINUFFT python wrappers [finufftpy](https://github.com/flatironinstitute/finufft/tree/master/finufftpy)
were modified to pass the number of threads. OUTDATED: to make use of this download FINUFFT, replace the 
python version with the version in this folder, 
and compile using the FINUFFT makefile: make python(3).

3) [UAMMD PSE routine ](https://github.com/RaulPPelaez/UAMMD/tree/v2.x/src/Integrator/BDHI/PSE) modified to 
pass the shear strain as an argument. To use this, download UAMMD and replace the PSE folder in src/Integrator/BDHI 
with the PSE folder here.
