all: RPYKernelEvaluator.so FiberCollection.so DomainC.so CrossLinkForceEvaluator.so EndedCrossLinking.so

CXX=g++
PYTHON3=python3

SLENDER_ROOT=../../
LDLIBS=-llapack -lblas -llapacke
OPENMP=-fopenmp


%.so:%.cpp makefile
	$(CXX) -O3 -Wall -shared $< -o $@ -std=c++11 -fPIC $(OPENMP) $(LDLIBS)  `$(PYTHON3) -m pybind11 --includes`

EndedCrossLinking.so: FortranHeap.o EndedCrossLinkers.cpp 
	$(CXX) -O3 -Wall -shared  -o $@ $^ -std=c++11 -fPIC $(OPENMP) $(LDLIBS) `$(PYTHON3) -m pybind11 --includes`

FortranHeap.o: $(SLENDER_ROOT)/Fortran/MinHeapModule.f90 makefile
	$(CXX) -x f95 -O3 -c  $< -o $@ -fPIC

clean:
	rm -f *.so FortranHeap.o
