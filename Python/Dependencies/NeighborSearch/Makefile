#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information 
LOG_LEVEL=0

CXX=g++
#If you define nvcc the code will compile in hybrid CPU/GPU mode, if this variable is empty
# the code will be compiled in CPU only mode
NVCC=
CUDA_ROOT=/usr/local/cuda
#Python is required for the python_wrappers example
PYTHON=python3
UAMMD_ROOT=../UAMMD_PSE_Python/uammd/
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS:=`python3 -m pybind11 --includes`
CXXSTD=14
OLEVEL=4
CXXFLAGS:= -std=c++$(CXXSTD) -ffast-math -march=native -O$(OLEVEL) -fPIC -g -DPYTHON_LIBRARY_MODE
ifneq ($(NVCC),)
UAMMD_INCLUDES=-I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
INCLUDEFLAGS:=-I$(CUDA_ROOT)/include $(UAMMD_INCLUDES) $(INCLUDEFLAGS) 
NVCCFLAGS=-std=c++$(CXXSTD) -O$(OLEVEL) $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler "$(CXXFLAGS)"
endif

LDFLAGS= -ltbb

PYTHON_MODULE_NAME=NeighborSearch
PYTHON_LIBRARY_NAME=$(PYTHON_MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)
all: $(PYTHON_LIBRARY_NAME)

$(PYTHON_LIBRARY_NAME): nlist_py.cu Makefile
ifneq ($(NVCC),)
	$(NVCC)  $(NVCCFLAGS) $(INCLUDEFLAGS) $< -o $(PYTHON_LIBRARY_NAME) $(LDFLAGS) -shared
else
	$(CXX)  $(CXXFLAGS) $(INCLUDEFLAGS) -x c++ $< -o $(PYTHON_LIBRARY_NAME) $(LDFLAGS) -shared
endif

clean:
	rm -f $(PYTHON_LIBRARY_NAME)
