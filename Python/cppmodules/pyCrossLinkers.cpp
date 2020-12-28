#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "CrossLinkers.cpp"

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

      
void py_initCLForcingVariables(npDoub pyUniformNodes, npDoub pyChebNodes, npDoub pyChebWts, double sigin, double Kin, double rl, int nCL)
{
  /**
    Python wrapper for node and weight initialization in C++. 
    @param pyUniformNodes  = uniform nodes on the fiber (1D numpy array)
    @param pyChebNodes = Chebyshev nodes on the fiber (1D numpy array)
    @param pyChebWts = Clenshaw-Curtis weights for direct quadrature on the fiber (1D numpy array)
    @param sigin = standard deviation sigma for the smeared delta function
    @param Kin = spring constant for CLs
    @param rl = rest length for CLs
    @param nCL = maximum number of cross linkers
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
  initCLForcingVariables(uniformNodes,chebNodes,chebWts,sigin, Kin, rl, nCL);
} 


py::array py_calcCLForces(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints, int nThreads)
{
  /**
    Python wrapper to evaluate the cross linking forces. 
    @param iPts = 1D list (pybind turns into a vector) of first point in the pair of linked uniform points 
    @param jPts = 1D list (to vector) of second point in the pair of linked uniform points 
    @param pyShifts = 2D numpy array of shifts in the links due to periodicity
    @param pyUnipoints = uniform points on the fibers for the force calculation
    @param pyChebPoints = Chebyshev fiber points for the force calculation
    @param nThreads = number of threads for parallel processing
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
  calcCLForces(iPts, jPts, Shifts, uniPoints, ChebPoints, CLForceDensities, nThreads);

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
    npDoub pychebCoefficients, double Lfib, int nThreads)
{
  /**
    Python wrapper to evaluate the cross linking forces. 
    @param iPts = 1D list (pybind turns into a vector) of first point in the pair of linked uniform points 
    @param jPts = 1D list (to vector) of second point in the pair of linked uniform points 
    @param pyShifts = 2D numpy array of shifts in the links due to periodicity
    @param pyUnipoints = uniform points on the fibers for the force calculation
    @param pyChebPoints = Chebyshev fiber points for the force calculation
    @param nThreads = number of threads for parallel processing
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
  calcCLForces2(iFibs, jFibs, iSstars, jSstars, Shifts, ChebPoints, ChebCoefficients,CLForceDensities, Lfib,nThreads);

  // return 1-D NumPy array
  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyArray = py::array_t<double>(pyChebPoints.shape()[0]*3);
  auto result_buffer = pyArray.request();
  double *result_ptr    = (double *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,CLForceDensities.data(),CLForceDensities.size()*sizeof(double));
  
  return pyArray;
} 



py::array py_calcCLStress(const intvec &iPts, const intvec &jPts, npDoub pyShifts, npDoub pyUnipoints, npDoub pyChebPoints, int nThreads)
{
  /**
    Python wrapper to evaluate the cross linking forces. 
    @param iPts = 1D list (pybind turns into a vector) of first point in the pair of linked uniform points 
    @param jPts = 1D list (to vector) of second point in the pair of linked uniform points 
    @param pyShifts = 2D numpy array of shifts in the links due to periodicity
    @param pyUnipoints = uniform points on the fibers for the force calculation
    @param pyChebPoints = Chebyshev fiber points for the force calculation
    @param nThreads = number of threads for parallel processing
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
  vec stress = calcCLStress(iPts, jPts, Shifts, uniPoints, ChebPoints, nThreads);

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

py::array py_calcKOns(npInt pynewLinks, npDoub pyUniPoints, double g)
{
  /**
    Python wrapper to evaluate the binding rates of a set of potential links. 
    @param pynewLinks = 2D numpy integer array of possible new binding locations for links. Each row is 
    a pair of points to be linked. 
    @param pyuniPoints = 2D numpy array of uniform points 
    @param g = strain in coordinate system
    @return the binding rates for each link
    
  **/
  
  // allocate std::vector (to pass to the C++ function)
  intvec newLinks(pynewLinks.size());
  vec uniPoints(pyUniPoints.size());
 
  // copy py::array -> std::vector
  std::memcpy(newLinks.data(),pynewLinks.data(),pynewLinks.size()*sizeof(int));
  std::memcpy(uniPoints.data(),pyUniPoints.data(),pyUniPoints.size()*sizeof(double));

  // call pure C++ function
  int nNewLinks = pynewLinks.shape()[0];
  vec BindingRates(nNewLinks,0.0);
  for (int iL =0 ; iL < nNewLinks; iL++){
      BindingRates[iL]=calcKonOne(newLinks[2*iL],newLinks[2*iL+1],uniPoints,g);
  }

  // return 1-D NumPy array
  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyArray = py::array_t<double>(nNewLinks);
  auto result_buffer = pyArray.request();
  double *result_ptr    = (double *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,BindingRates.data(),BindingRates.size()*sizeof(double));
  
  return pyArray;
} 

py::array py_newEventsList(npDoub pyrates, npInt pyiPts, npInt pyjPts, npInt pynowBound, npInt pyadded, int nLinks,
    npDoub pyUniPoints, double g, double tstep){
  /**
    Python wrapper for method that determines the events that occur in a given timestep. 
    @param pyrates = 1d numpy array of rates of each event happening (units 1/time)
    @param pyiPts = 1D numpy array of one endpoint index for each link (uniform point index)
    @param pyjPts = 1D numpy array of other endpoint index
    @param pynowBound = 1D numpy array of bound state of each event/link
    @param pyadded = 1D numpy array of whether a link exists at each uniform point (only 1 link allowed)
    @param nLinks = number of links going into this
    @param pyuniPoints = 2D numpy array of uniform points on the fibers
    @param g = strain in the coordinate system
    @param tstep = timestep
    @return 1D numpy array of the events that will happen during the timestep. 
    Each "event" is an index in iPts and jPts that says that pair will bind
    **/
  
  // allocate std::vector (to pass to the C++ function)
  vec rates(pyrates.size());
  intvec iPts(pyiPts.size());
  intvec jPts(pyjPts.size());
  intvec nowBound(pynowBound.size());
  intvec added(pyadded.size());
  vec uniPoints(pyUniPoints.size());
 
  // copy py::array -> std::vector
  std::memcpy(rates.data(),pyrates.data(),pyrates.size()*sizeof(double));
  std::memcpy(iPts.data(),pyiPts.data(),pyiPts.size()*sizeof(int));
  std::memcpy(jPts.data(),pyjPts.data(),pyjPts.size()*sizeof(int));
  std::memcpy(nowBound.data(),pynowBound.data(),pynowBound.size()*sizeof(int));
  std::memcpy(added.data(),pyadded.data(),pyadded.size()*sizeof(int));
  std::memcpy(uniPoints.data(),pyUniPoints.data(),pyUniPoints.size()*sizeof(double));

  // call pure C++ function
  intvec events = newEventsList(rates, iPts, jPts, nowBound, added, nLinks,uniPoints, g, tstep);
  
  // return 1-D NumPy array
  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyEvents = py::array_t<int>(events.size());
  auto result_buffer = pyEvents.request();
  int *result_ptr    = (int *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,events.data(),events.size()*sizeof(int));
  
  return pyEvents;
} 
    
// PYTHON MODULE DECLARATION
// Module for python
PYBIND11_MODULE(CrossLinking, m) {
    
    m.doc() = "The C++ functions for cross linking"; // optional module docstring
    m.def("seedRandomCLs", &seedRandomCLs, "Seed random number generator");
    m.def("initDynamicCLVariables",&initDynamicCLVariables, "Initialize global variables for dynamic CLs");
    m.def("initCLForcingVariables",&py_initCLForcingVariables,"Initialize global variables for CL force calculation");
    m.def("calcCLForces", &py_calcCLForces, "Forces due to the cross linkers");
    m.def("calcCLForces2", &py_calcCLForces2, "Forces due to the cross linkers");
    m.def("calcCLStress", &py_calcCLStress, "Stress due to the cross linkers");
    m.def("calcKons", &py_calcKOns, "Rate binding constants");
    m.def("newEvents", &py_newEventsList, "Vector of new events");
}
