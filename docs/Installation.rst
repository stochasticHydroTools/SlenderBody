Installing the software
===================================================

**Instructions for running code**
   1) Clone the repo using
    
      .. code-block:: bash
       
       git clone --recursive https://github.com/stochasticHydroTools/SlenderBody
       
      This will also clone the submodules in the dependency folder. 
       
   2) Run
   
      .. code-block:: bash
      
        cd SlenderBody
        bash install.sh
      
      
      in the main directory. This will compile all of the dependencies and C++ modulues. It will also
      add the appropriate directories (where the python modulues are compiled) to your PYTHONPATH.  

      If the compilation process fails this script will show an error. In that case fix the compilation issue and run 
      
      .. code-block:: bash
        
        bash install.sh
      
      again.

   3) Common mistakes:
        - `UAMMD <https://github.com/RaulPPelaez/UAMMD>`_, which we use for hydrodynamic interactions, requires a GPU
          and a system with a working CUDA environment. You can still run the software without a GPU, but you cannot
          compute hydrodynamics between fibers.
        - When compiling UAMMD in the dependencies, the compiler may complain depending on what version of nvcc you use.
          If it complains that there is no file called cub, cd into SlenderBody/Python/Dependencies/UAMMD_PSE_Python/uammd/src/third_party
          and change the folder called "cub_bak" to "cub." This should fix that issue. See also the `UAMMD webpage <https://github.com/RaulPPelaez/UAMMD>`_
          for more compilation instructions. 
        - Sometimes there is an issue with linking the fortran library (Heap) to C++ and then Python. If the commands in the makefile (in cppmodules)
          produce an error, this sequence typically works
          
          .. code-block:: bash
          
            gfortran -shared -O3 ../../Fortran/MinHeapModule.f90 -o /($SLENDERROOT)/Fortran/libFortranHeap.so -fPIC
            icpc -O3 -Wall -shared EndedCrossLinkers.cpp -o EndedCrossLinkedNetwork.so     -L/($SLENDERROOT)/Fortran/ -lFortranHeap -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`
          
          where (SLENDERROOT) is the location of installation. Make sure that ($SLENDERROOT)/Fortran is also added to your LD_LIBRARY_PATH. 
        - The file `VectorMethods.cpp <https://github.com/stochasticHydroTools/SlenderBody/blob/b0982fcad18dddf88f04dc82c41dadb47c5f25c9/Python/cppmodules/VectorMethods.cpp#L3>`_ is set up to use LAPACKE and CBLAS. If you compile using Intel's MKL, then simply comment lines 3 and 4 and uncomment line 18.
              

   4) Run the python scripts in Python/Examples. For example,
    
       .. code-block:: bash 
        
        python3 ThreeShearedFibs.py
       
      will run ThreeShearedFibs.py (see Examples page for documentation).

**Parallelization**

The number of OpenMP threads (environment variable) MUST be set to one to obtain good performance. 
In particular, you must use 

.. code-block:: bash
    
    export OMP_NUM_THREADS=1

in linux prior to running our code.
The parallelization is then implemented in the C++ class FiberCollectionC.cpp which takes as an inpute the number of threads. The number of threads in these calculations can be set by passing an integer to the constructor of fiberCollection.py.

**Uninstalling**
    1) Run 
       
       .. code-block:: bash
        
        make clean
        
       inside the Python folder.  

    2) The install.sh script modifies the .bashrc file (which are marked as written by the script). Simply remove these lines.  
