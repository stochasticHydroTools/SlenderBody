all: RPYKernelEvaluator.so FiberCollectionC.so DomainC.so CrossLinkForceEvaluator.so  StericForceEvaluatorC.so

CXX=icpx
PYTHON3=python3

LDLIBS= -qmkl  -lgfortran
OPENMP=-qopenmp
PYBINDINCLUDE:=`$(PYTHON3) -m pybind11 --includes`

CXXFLAGS=-Wall -O3 -std=c++11 -fPIC $(OPENMP) $(LDLIBS)  $(PYBINDINCLUDE)

SLENDER_ROOT=../../
%.so:%.cpp makefile
	$(CXX) -shared $< -o $@ $(CXXFLAGS)

EndedCrossLinkedNetwork.so: FortranHeap.o EndedCrossLinkers.cpp 
	$(CXX)  -shared  -o $@ $^ $(CXXFLAGS)

FortranHeap.o: $(SLENDER_ROOT)/Fortran/MinHeapModule.f90 makefile
	$(CXX) -x f95 -O3 -c  $< -o $@ -fPIC

clean:
	rm -f *.so FortranHeap.o


