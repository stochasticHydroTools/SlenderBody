all: RPYKernelEvaluator FiberCollection DomainC CrossLinkForceEvaluator EndedCrossLinking

DomainC:
	c++ -O3 -Wall -shared DomainC.cpp -o DomainC.so -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke  `python3 -m pybind11 --includes`

RPYKernelEvaluator: 
	c++ -O3 -Wall -shared RPYKernelEvaluator.cpp -o RPYKernelEvaluator.so -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`

FiberCollection: 
	c++ -O3 -Wall -shared FiberCollection.cpp -o FiberCollection.so -llapack -lblas -llapacke -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` 

CrossLinkForceEvaluator: 
	c++ -O3 -Wall -shared CrossLinkForceEvaluator.cpp -o CrossLinkForceEvaluator.so -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`

EndedCrossLinking:
	gfortran -shared -O3 ../../Fortran/MinHeapModule.f90 -o /home/ondrejmaxian/Documents/SLENDER_FIBERS/SlenderBody/Fortran/libFortranHeap.so -fPIC
	c++ -O3 -Wall -shared EndedCrossLinkers.cpp -o EndedCrossLinkedNetwork.so \
	    -L/home/ondrejmaxian/Documents/SLENDER_FIBERS/SlenderBody/Fortran/ -lFortranHeap -std=c++11 -fPIC -fopenmp -llapack -lblas -llapacke `python3 -m pybind11 --includes`

clean:
	rm *.so
