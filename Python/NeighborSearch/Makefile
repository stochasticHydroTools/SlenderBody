#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information 
LOG_LEVEL=0

NVCC=nvcc
CUDA_ROOT=/usr/local/cuda
#Python is required for the python_wrappers example
PYTHON=python3
UAMMD_ROOT=../UAMMD_PSE_Python/uammd/
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
INCLUDEFLAGS=-I$(CUDA_ROOT)/include -I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/utils -I$(UAMMD_ROOT)/src/third_party `python3 -m pybind11 --includes`
NVCCFLAGS=-std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION) --expt-extended-lambda --expt-relaxed-constexpr

PYTHON_MODULE_NAME=NeighborSearch
PYTHON_LIBRARY_NAME=$(PYTHON_MODULE_NAME)$(shell $(PYTHON)-config --extension-suffix)
all: $(PYTHON_LIBRARY_NAME)

$(PYTHON_LIBRARY_NAME): nlist_py.cu Makefile
	$(NVCC)  $(NVCCFLAGS) $< -o $(PYTHON_LIBRARY_NAME) -ltbb -w -shared -I pybind11/include -Xcompiler "-fPIC -w" `$(PYTHON)-config --includes` 

clean:
	rm -f $(PYTHON_LIBRARY_NAME)
