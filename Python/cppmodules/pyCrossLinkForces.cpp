#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "CrossLinkForces.cpp"

/**
    Python wrappers for cross linkers
    This file (prefixed by py) is just a list of interfaces 
    that call the C++ functions in CrossLinkForces.cpp
**/

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

      
void py_initCLForcingVariables(npDoub pyUniformNodes, npDoub pyChebNodes, npDoub pyChebWts, double sigin, double Kin, double rl, int nThreads)
{
  /**
    Python wrapper for node and weight initialization in C++. 
    @param pyUniformNodes  = uniform nodes on the fiber (1D numpy array)
    @param pyChebNodes = Chebyshev nodes on the fiber (1D numpy array)
    @param pyChebWts = Clenshaw-Curtis weights for direct quadrature on the fiber (1D numpy array)
    @param sigin = standard deviation sigma for the smeared delta function
    @param Kin = spring constant for CLs
    @param rl = rest length for CLs
    @param nThreads = number of threads for parallel processing
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec uniformNodes(pyUniformNodes.size());
  vec chebNodes(pyChebNodes.size());
  vec chebWts(pyChebWts.size());

  // copy py::array -> std::vector
  std::memcpy(uniformNodes.data(),pyUniformNodes.data(),pyUniformNodes.size()*sizeof(double));
  std::memcpy(chebNodes.data(),pyChebNodes.data(),pyChebNodes.size()*sizeof(double));
  std::memcpy(chebWts.data(),pyChebWts.data(),pyChebWts.size()*sizeof(double));

  // call pure C++ function
  initCLForcingVariables(uniformNodes,chebNodes,chebWts,sigin, Kin, rl,nThreads);
} 


py::array py_calcCLForces(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints)
{
  /**
    Python wrapper to evaluate the cross linking forces. 
    @param iPts = 1D list (pybind turns into a vector) of first point in the pair of linked uniform points 
    @param jPts = 1D list (to vector) of second point in the pair of linked uniform points 
    @param pyShifts = 2D numpy array of shifts in the links due to periodicity
    @param pyUnipoints = uniform points on the fibers for the force calculation
    @param pyChebPoints = Chebyshev fiber points for the force calculation
    @return the force densities at all the points due to cross-linking
    
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec Shifts(pyShifts.size());
  vec uniPoints(pyUnipoints.size());
  vec ChebPoints(pyChebPoints.size());
 
  // copy py::array -> std::vector
  std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
  std::memcpy(uniPoints.data(),pyUnipoints.data(),pyUnipoints.size()*sizeof(double));
  std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));

  // call pure C++ function
  vec CLForceDensities(pyChebPoints.shape()[0]*3,0.0);
  calcCLForces(iPts, jPts, Shifts, uniPoints, ChebPoints, CLForceDensities);

  // return 1-D NumPy array
  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyArray = py::array_t<double>(pyChebPoints.shape()[0]*3);
  auto result_buffer = pyArray.request();
  double *result_ptr    = (double *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,CLForceDensities.data(),CLForceDensities.size()*sizeof(double));
  
  return pyArray;
} 

py::array py_calcCLForces2(npInt pyiFibs, npInt pyjFibs, npDoub pyiSstars, npDoub pyjSstars, npDoub pyShifts, npDoub pyChebPoints, 
    npDoub pychebCoefficients, double Lfib)
{
  /**
    Python wrapper to evaluate the cross linking forces. 
    @param iPts = 1D list (pybind turns into a vector) of first point in the pair of linked uniform points 
    @param jPts = 1D list (to vector) of second point in the pair of linked uniform points 
    @param pyShifts = 2D numpy array of shifts in the links due to periodicity
    @param pyUnipoints = uniform points on the fibers for the force calculation
    @param pyChebPoints = Chebyshev fiber points for the force calculation
    @return the force densities at all the points due to cross-linking
    
  **/
  
  // allocate std::vector (to pass to the C++ function)
  intvec iFibs(pyiFibs.size());
  intvec jFibs(pyjFibs.size());
  vec iSstars(pyiSstars.size());
  vec jSstars(pyjSstars.size());
  vec Shifts(pyShifts.size());
  vec ChebPoints(pyChebPoints.size());
  vec ChebCoefficients(pychebCoefficients.size());
 
  // copy py::array -> std::vector
  std::memcpy(iFibs.data(),pyiFibs.data(),pyiFibs.size()*sizeof(int));
  std::memcpy(jFibs.data(),pyjFibs.data(),pyjFibs.size()*sizeof(int));
 
  // copy py::array -> std::vector
  std::memcpy(iSstars.data(),pyiSstars.data(),pyiSstars.size()*sizeof(double));
  std::memcpy(jSstars.data(),pyjSstars.data(),pyjSstars.size()*sizeof(double));
  std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
  std::memcpy(ChebCoefficients.data(),pychebCoefficients.data(),pychebCoefficients.size()*sizeof(double));
  std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));

  // call pure C++ function
  vec CLForceDensities(pyChebPoints.shape()[0]*3,0.0);
  calcCLForces2(iFibs, jFibs, iSstars, jSstars, Shifts, ChebPoints, ChebCoefficients,CLForceDensities, Lfib);

  // return 1-D NumPy array
  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyArray = py::array_t<double>(pyChebPoints.shape()[0]*3);
  auto result_buffer = pyArray.request();
  double *result_ptr    = (double *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,CLForceDensities.data(),CLForceDensities.size()*sizeof(double));
  
  return pyArray;
} 



py::array py_calcCLStress(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints)
{
  /**
    Python wrapper to evaluate the cross linking forces. 
    @param iPts = 1D list (pybind turns into a vector) of first point in the pair of linked uniform points 
    @param jPts = 1D list (to vector) of second point in the pair of linked uniform points 
    @param pyShifts = 2D numpy array of shifts in the links due to periodicity
    @param pyUnipoints = uniform points on the fibers for the force calculation
    @param pyChebPoints = Chebyshev fiber points for the force calculation
    @return stress due to cross-linkers
    
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec Shifts(pyShifts.size());
  vec uniPoints(pyUnipoints.size());
  vec ChebPoints(pyChebPoints.size());
 
  // copy py::array -> std::vector
  std::memcpy(Shifts.data(),pyShifts.data(),pyShifts.size()*sizeof(double));
  std::memcpy(uniPoints.data(),pyUnipoints.data(),pyUnipoints.size()*sizeof(double));
  std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));

  // call pure C++ function
  vec stress = calcCLStress(iPts, jPts, Shifts, uniPoints, ChebPoints);

  ssize_t              ndim    = 2;
  std::vector<ssize_t> shape   = { 3, 3 };
  std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    stress.data(),                       /* data as contiguous array  */
    sizeof(double),                          /* size of one scalar        */
    py::format_descriptor<double>::format(), /* data type                 */
    ndim,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
  
} 

// PYTHON MODULE DECLARATION
// Module for python
PYBIND11_MODULE(CrossLinkForces, m) {
    
    m.doc() = "The C++ functions for cross linking"; // optional module docstring
    m.def("initCLForcingVariables",&py_initCLForcingVariables,"Initialize global variables for CL force calculation");
    m.def("calcCLForces", &py_calcCLForces, "Forces due to the cross linkers");
    m.def("calcCLForces2", &py_calcCLForces2, "Forces due to the cross linkers");
    m.def("calcCLStress", &py_calcCLStress, "Stress due to the cross linkers");
}
