all: RPYKernels ManyFiberMethods DomainCpp

DomainCpp:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` Domain.cpp -o DomainCpp.so

RPYKernels: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyRPYKernels.cpp -o RPYKernels.so

ManyFiberMethods: 
	c++ -O3 -Wall -llapack -lblas -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyManyFiberMethods.cpp -o ManyFiberMethods.so

CrossLinking: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyCrossLinkers.cpp -o CrossLinking.so

clean:
	rm *.so
