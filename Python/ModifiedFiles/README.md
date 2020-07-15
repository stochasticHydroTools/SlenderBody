This folder contains the files that were modified from external libraries. There are 
two main modifications. 

1) The file [linsys.py](https://github.com/andrenarchy/krypy/blob/master/krypy/linsys.py) in 
the krypy library was modified to remove an extraneous mobility application

2) FINUFFT python wrappers [finufftpy](https://github.com/flatironinstitute/finufft/tree/master/finufftpy)
were modified to pass the number of threads 

To properly use the external dependencies, download krypy and replace linsys.py with the version in 
this folder. Likewise, download FINUFFT, replace the python version with the version in this folder, 
and compile using the FINUFFT makefile: make python(3). 