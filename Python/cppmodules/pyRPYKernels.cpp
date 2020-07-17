#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "RPYKernels.cpp"

/**
    pyRPYKernels.cpp
    Python wrappers for Ewald and total RPY calculations. 
    Here we only deal with the near field (far field uses FINUFFT). 
**/

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

py::array py_RPYNearPairs(npInt PyPairPts, npDoub pyPoints, npDoub pyForces, double xi, 
                          double g, double rcut, int nThreads)
{
  /**
  Python wrapper for the near field kernel evaluations, done in pairs. 
  @param PyPairPts = 2D integer numpy array of pairs of points, where each row represents a
  pair of indices that interact
  @param pyPoints = 2D numpy array of the Chebyshev points 
  @param pyForces = 2D numpy array of the forces at the Chebyshev points
  @param xi =  Ewald parameter
  @param g = strain in the coordinate system
  @param rcut = cutoff distance for near field Ewald
  @param nThreads = number of threads for the calculation
  @return 2D numpy array of the near field velocities 
  **/
    
  // allocate std::vector (to pass to the C++ function)
  intvec PairPts(PyPairPts.size());
  vec Points(pyPoints.size());
  vec Forces(pyForces.size());

  // copy py::array -> std::vector
  std::memcpy(PairPts.data(),PyPairPts.data(),PyPairPts.size()*sizeof(int));
  std::memcpy(Points.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
  std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));

  // call pure C++ function
  int Npairs = PyPairPts.shape()[0];
  vec result = RPYNKerPairs(Npairs,PairPts,Points,Forces,xi,g,rcut,nThreads);

  ssize_t              ndim    = 2;
  std::vector<ssize_t> shape   = { pyPoints.shape()[0] , 3 };
  std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    result.data(),                           /* data as contiguous array  */
    sizeof(double),                          /* size of one scalar        */
    py::format_descriptor<double>::format(), /* data type                 */
    ndim,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
}

// Python wrapper for a single near field calculation            
py::array py_RPYNearKernel(vec3 &rvec,const vec3 &force, double xi)
{
  /**
  Python wrapper for a single near field evaluation. 
  @param rvec = displacement vector between the points
  @param force = the force at the point
  @param xi = Ewald parameter
  **/
  
  vec3 unear;
  RPYNKer(rvec,force,xi,unear);

  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyArray = py::array_t<double>(3);
  auto result_buffer = pyArray.request();
  double *result_ptr    = (double *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,unear.data(),unear.size()*sizeof(double));
  
  return pyArray;
}                                  
    
// PYTHON MODULE DECLARATION
PYBIND11_MODULE(RPYKernels, m) {
    m.doc() = "The C++ version of the functions for Ewald RPY"; // optional module docstring
    m.def("EvaluateRPYNearPairs", &py_RPYNearPairs, "RPY near sum done in pairs");
    m.def("RPYNearKernel", &py_RPYNearKernel, "Single evaluation of near kernel");
    m.def("initRPYVars", &initRPYVars, "Initialize global variables");
}
