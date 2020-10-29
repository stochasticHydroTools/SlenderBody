all: RPYKernels ManyFiberMethods DomainCpp CrossLinking

DomainCpp:
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` Domain.cpp -o DomainCpp.so

RPYKernels: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyRPYKernels.cpp -o RPYKernels.so

ManyFiberMethods: 
	c++ -O3 -Wall -shared pyManyFiberMethods.cpp -o ManyFiberMethods.so -llapack -lblas -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` 

CrossLinking: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC -fopenmp `python3 -m pybind11 --includes` pyCrossLinkers.cpp -o CrossLinking.so

clean:
	rm *.so
