all: EwaldUtils SpecQuadUtils CLUtils FiberUtils

EwaldUtils: 
	c++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` EwaldUtils.cpp -o EwaldUtils.so

SpecQuadUtils:
	c++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` SpecQuadUtils.cpp -o SpecQuadUtils.so

CLUtils:
	c++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` CLUtils.cpp -o CLUtils.so

FiberUtils:
	c++ -O3 -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` FiberUtils.cpp -o FiberUtils.so

clean:
	rm *.so
