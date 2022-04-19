all: RPYKernelEvaluator.so FiberCollection.so DomainC.so CrossLinkForceEvaluator.so EndedCrossLinkedNetwork.so

CXX=g++
PYTHON3=python3

LDLIBS=-llapacke -lcblas
OPENMP=-fopenmp
PYBINDINCLUDE:=`$(PYTHON3) -m pybind11 --includes`

CXXFLAGS=-Wall -O3 -std=c++11 -fPIC $(OPENMP) $(LDLIBS)  $(PYBINDINCLUDE)

SLENDER_ROOT=../..
%.so:%.cpp makefile
	$(CXX) -shared $< -o $@ $(CXXFLAGS)

EndedCrossLinkedNetwork.so: 
	gfortran -shared -O3 $(SLENDER_ROOT)/Fortran/MinHeapModule.f90 -o $(SLENDER_ROOT)/Fortran/libFortranHeap.so -fPIC
	c++ -shared EndedCrossLinkers.cpp -o EndedCrossLinkedNetwork.so \
	    -L/home/ondrejmaxian/Documents/SLENDER_FIBERS/SlenderBody/Fortran/ -lFortranHeap $(CXXFLAGS)

clean:
	rm -f *.so
