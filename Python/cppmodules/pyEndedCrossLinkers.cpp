#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "EndedCrossLinkers.cpp"

/**
    Python wrapper for ended CL class
**/

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 


PYBIND11_MODULE(EndedCrossLinkedNetwork, m) {
    py::class_<EndedCrossLinkedNetwork>(m, "EndedCrossLinkedNetwork")
        .def(py::init<int, vec, vec3, vec, double>())
        .def("updateNetwork", &EndedCrossLinkedNetwork::updateNetwork)
        .def("getNBoundEnds", &EndedCrossLinkedNetwork::getNBoundEnds)
        .def("getLinkHeadsOrTails",&EndedCrossLinkedNetwork::getLinkHeadsOrTails)
        .def("getLinkShifts", &EndedCrossLinkedNetwork::getLinkShifts)
        .def("setLinks",&EndedCrossLinkedNetwork::setLinks)
        .def("deleteLinksFromSites", &EndedCrossLinkedNetwork::deleteLinksFromSites);
}
