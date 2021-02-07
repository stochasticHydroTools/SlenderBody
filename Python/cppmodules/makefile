all: RPYKernels ManyFiberMethods DomainCpp CrossLinkForces EndedCrossLinking

DomainCpp:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` Domain.cpp -o DomainCpp.so

RPYKernels: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyRPYKernels.cpp -o RPYKernels.so

ManyFiberMethods: 
	c++ -O3 -Wall -shared pyManyFiberMethods.cpp -o ManyFiberMethods.so -llapack -lblas -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` 

CrossLinkForces: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyCrossLinkForces.cpp -o CrossLinkForces.so

EndedCrossLinking:
	gfortran -shared -O3 ../../Fortran/MinHeapModule.f90 -o /home/ondrejmaxian/lib/libFortranHeap.so -fPIC
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyEndedCrossLinkers.cpp -o EndedCrossLinkedNetwork.so \
	    -L/home/ondrejmaxian/lib/ -lFortranHeap

clean:
	rm *.so
