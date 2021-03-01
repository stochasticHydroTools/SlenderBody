all: RPYKernels FiberCollection DomainCpp CrossLinkForces EndedCrossLinking

DomainCpp:
	c++ -O3 -Wall -shared Domain.cpp -o DomainCpp.so -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke  `python3 -m pybind11 --includes`

RPYKernels: 
	c++ -O3 -Wall -shared pyRPYKernels.cpp -o RPYKernels.so -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`

FiberCollection: 
	c++ -O3 -Wall -shared pyFiberCollection.cpp -o FiberCollection.so -llapack -lblas -llapacke -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` 

CrossLinkForces: 
	c++ -O3 -Wall -shared pyCrossLinkForces.cpp -o CrossLinkForces.so -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`

EndedCrossLinking:
	gfortran -shared -O3 ../../Fortran/MinHeapModule.f90 -o /home/ondrejmaxian/lib/libFortranHeap.so -fPIC
	c++ -O3 -Wall -shared pyEndedCrossLinkers.cpp -o EndedCrossLinkedNetwork.so \
	    -L/home/ondrejmaxian/lib/ -lFortranHeap -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`

clean:
	rm *.so
