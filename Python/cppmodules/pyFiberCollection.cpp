#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "FiberCollection.cpp"

/**
    Python wrapper for ended CL class
**/

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 


PYBIND11_MODULE(FiberCollection, m) {
    py::class_<FiberCollection>(m, "FiberCollection")
    
        .def(py::init<double,vec3, double, double, double,int,int, int, double, int>())
        .def("initSpecQuadParams", &FiberCollection::initSpecQuadParams)
        .def("initNodesandWeights", &FiberCollection::initNodesandWeights)
        .def("initResamplingMatrices",&FiberCollection::initResamplingMatrices)
        .def("init_FinitePartMatrix", &FiberCollection::init_FinitePartMatrix)
        .def("initMatricesForPreconditioner", &FiberCollection::initMatricesForPreconditioner)
        .def("RPYFiberKernel",&FiberCollection::RPYFiberKernel)
        .def("SubtractAllRPY",&FiberCollection::SubtractAllRPY)
        .def("FinitePartVelocity",&FiberCollection::FinitePartVelocity)
        .def("RodriguesRotations",&FiberCollection::RodriguesRotations)
        .def("applyPreconditioner",&FiberCollection::applyPreconditioner)
        .def("CorrectNonLocalVelocity",&FiberCollection::CorrectNonLocalVelocity)
        .def("evalBendForces",&FiberCollection::evalBendForces)
        .def("evalLocalVelocities", &FiberCollection::evalLocalVelocities);
}
