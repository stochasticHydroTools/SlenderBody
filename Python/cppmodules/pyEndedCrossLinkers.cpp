#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "EndedCrossLinkers.cpp"

/**
    Python wrappers for cross linkers
    This file (prefixed by py) is just a list of interfaces 
    that call the C++ functions in CrossLinkers.cpp
**/

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 


PYBIND11_MODULE(EndedCrossLinkedNetwork, m) {
    py::class_<EndedCrossLinkedNetwork>(m, "EndedCrossLinkedNetwork")
        .def(py::init<int, vec, double>())
        .def("updateNetwork", &EndedCrossLinkedNetwork::updateNetwork)
        .def("getNBoundEnds", &EndedCrossLinkedNetwork::getNBoundEnds)
        .def("getLinkHeadsOrTails",&EndedCrossLinkedNetwork::getLinkHeadsOrTails);
        /*
        .def("setNumChildren", &EndedCrossLinkedNetwork::pysetNumChildren)
        .def("getFirstGen", &EndedCrossLinkedNetwork::getFirstGen)
        .def("__repr__",
        [](const EndedCrossLinkedNetwork &a) {
            return "<EndedCrossLinkedNetwork named '" + a.getName() + "'>";
        }
    );*/
}

int main(){
 return 0;
}


