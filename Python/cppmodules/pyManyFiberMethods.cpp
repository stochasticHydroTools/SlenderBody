#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "ManyFiberMethods.cpp"

/**
    Python wrappers for many fiber quadratures. 
    This file (prefixed by py) is just a list of interfaces 
    that call the C++ functions in ManyFiberMethods.cpp
**/

// ----------------
// Python interface
// ----------------

namespace py = pybind11;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> npInt; 
typedef py::array_t<double, py::array::c_style | py::array::forcecast> npDoub; 

      
void py_initNodesandWeights(npDoub pyNormalNodes, npDoub pyNormalWts, npDoub pyUpNodes, npDoub pyUpWeights)
{
  /**
    Python wrapper for node and weight initialization in C++. 
    @param pyNormalNodes = Chebyshev nodes on the fiber (1D numpy array)
    @param pyNormalWeights = Clenshaw-Curtis weights for direct quadrature on the fiber (1D numpy array)
    @param pyUpNodes  = upsampled Chebyshev nodes on the fiber (1D numpy array)
    @param pyNormalWeights = upsampled Clenshaw-Curtis weights for upsampled direct quadrature on the fiber (1D numpy array)
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec normalNodes(pyNormalNodes.size());
  vec normalWts(pyNormalWts.size());
  vec upNodes(pyUpNodes.size());
  vec upWeights(pyUpWeights.size());

  // copy py::array -> std::vector
  std::memcpy(normalNodes.data(),pyNormalNodes.data(),pyNormalNodes.size()*sizeof(double));
  std::memcpy(normalWts.data(),pyNormalWts.data(),pyNormalWts.size()*sizeof(double));
  std::memcpy(upNodes.data(),pyUpNodes.data(),pyUpNodes.size()*sizeof(double));
  std::memcpy(upWeights.data(),pyUpWeights.data(),pyUpWeights.size()*sizeof(double));

  // call pure C++ function
  initNodesandWeights(normalNodes,normalWts,upNodes,upWeights);
} 

// Python wrapper for node and weight initialization          
void py_initResamplingMatrices(npDoub pyRsUpsamp, npDoub pyRs2Panel, npDoub pyValtoCoef)
{
  /**
    Python wrapper for initializion of resampling and upsampling matrices in C++
    @param pyRsUpsample = upsampling matrix from the N point grid to Nupsample point grid (2D numpy aray)
    @param pyRs2Panel = upsampling matrix from the N point grid to 2 panels of Nupsample point grid (2D numpy aray)
    @param pyValtoCoef  = matrix that takes values to Chebyshev series coefficients on the N point Chebyshev grid (2D numpy array)
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec upsampMat(pyRsUpsamp.size());
  vec upsamp2PanMat(pyRs2Panel.size());
  vec ValtoCoeffMat(pyValtoCoef.size());

  // copy py::array -> std::vector
  std::memcpy(upsampMat.data(),pyRsUpsamp.data(),pyRsUpsamp.size()*sizeof(double));
  std::memcpy(upsamp2PanMat.data(),pyRs2Panel.data(),pyRs2Panel.size()*sizeof(double));
  std::memcpy(ValtoCoeffMat.data(),pyValtoCoef.data(),pyValtoCoef.size()*sizeof(double));
  

  // call pure C++ function
  initResamplingMatrices(upsampMat,upsamp2PanMat, ValtoCoeffMat);
} 

void py_initFinitePartMatrix(npDoub pyFPMatrix, npDoub pyDiffMatrix)
{    
  /**
    Python wrapper to initialize variables for finite part integration in C++. 
    @param pyFPMatrix = matrix that maps g function coefficients to velocity values on the N point Cheb grid (2D numpy array)
    @param pyDiffMatrix = Chebyshev differentiation matrix on the N point Chebyshev grid (2D numpy aray)
  **/
  
    // allocate std::vector (to pass to the C++ function)
    vec FinitePartMatrix(pyFPMatrix.size());
    vec DiffMatrix(pyDiffMatrix.size());
    
    // copy py::array -> std::vector
    std::memcpy(FinitePartMatrix.data(),pyFPMatrix.data(),pyFPMatrix.size()*sizeof(double));
    std::memcpy(DiffMatrix.data(),pyDiffMatrix.data(),pyDiffMatrix.size()*sizeof(double));
  
    // call pure C++ function
    initFinitePartMatrix(FinitePartMatrix, DiffMatrix);
}

py::array py_RPYFiberKernel(npDoub pyPoints, npDoub pyForces)
{
  /**
    Python wrapper to evaluate free space RPY kernel over a fiber (necessary to subtract from the Ewald results), since
    a single fiber is handled by SBT, not RPY. 
    @param pyPoints = Npts * 3 numpy array (2D) of points on the fiber 
    @param pyForces = Npts * 3 numpy array (2D) of forces (not force densities) on the fiber
    @return 2D numpy array of the velocity at the fiber points
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec Points(pyPoints.size());
  vec Forces(pyForces.size());

  // copy py::array -> std::vector
  std::memcpy(Points.data(),pyPoints.data(),pyPoints.size()*sizeof(double));
  std::memcpy(Forces.data(),pyForces.data(),pyForces.size()*sizeof(double));

  // call pure C++ function
  vec velocities(pyPoints.shape()[0]*3,0.0);
  RPYFiberKernel(Points,Points,Forces,velocities);

  ssize_t              ndim    = 2;
  std::vector<ssize_t> shape   = { pyPoints.shape()[0] , 3 };
  std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    velocities.data(),                       /* data as contiguous array  */
    sizeof(double),                          /* size of one scalar        */
    py::format_descriptor<double>::format(), /* data type                 */
    ndim,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
} 

// Python wrapper for velocity corrections     
py::array py_CorrectNonLocalVelocity(npDoub pyChebPoints, npDoub pyUniformPoints, npDoub pyForceDens, 
     npDoub pyFinitePartVels, double g,  npInt pyNumbyFib, intvec &TargetNums, int nThreads)
{
  /**
    Python wrapper to correct the velocity from Ewald via special quadrature (or upsampled quadrature)
    @param pyChebPoints = Npts * 3 numpy array (2D) of Chebyshev points on ALL fibers
    @param pyUniformPoints = Npts * 3 numpy array (2D) of uniform points on ALL fibers
    @param pyForceDens = Npts * 3 numpy array (2D) of force densities on ALL fibers
    @param pyFinitePartVels = Npts * 3 numpy array (2D) of velocities due to the finite part integral on 
        ALL fibers (necessary when the fibers are close togther and the centerline velocity is used)
    @param g = strain in coordinate system
    @param pyNumbyFib = Npts (1D) numpy array of the number of targets that require correction on each fiber
    @param TargetNums = python LIST (not array) of the target indices that need correction in sequential order
    @param nThreads = number of threads to use in parallel processing
    @return velocities = Npts * 3 numpy array (2D) of corrections to the Ewald velocity. 
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec ChebPoints(pyChebPoints.size());
  vec UniformPoints(pyUniformPoints.size());
  vec ForceDensities(pyForceDens.size());
  vec CenterLineVels(pyFinitePartVels.size());
  intvec NumbyFib(pyNumbyFib.size());
  //intvec TargetNums(pytargetNums.size());

  // copy py::array -> std::vector
  std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
  std::memcpy(UniformPoints.data(),pyUniformPoints.data(),pyUniformPoints.size()*sizeof(double));
  std::memcpy(ForceDensities.data(),pyForceDens.data(),pyForceDens.size()*sizeof(double));
  std::memcpy(CenterLineVels.data(),pyFinitePartVels.data(),pyFinitePartVels.size()*sizeof(double));
  std::memcpy(NumbyFib.data(),pyNumbyFib.data(),pyNumbyFib.size()*sizeof(int));
  //std::memcpy(TargetNums.data(),pytargetNums.data(),pytargetNums.size()*sizeof(int));

  // call pure C++ function
  vec velocities(pyChebPoints.shape()[0]*3,0.0);
  CorrectNonLocalVelocity(ChebPoints,UniformPoints,ForceDensities,CenterLineVels, g,NumbyFib,TargetNums, velocities,nThreads);

  ssize_t              ndim    = 2;
  std::vector<ssize_t> shape   = { pyChebPoints.shape()[0] , 3 };
  std::vector<ssize_t> strides = { sizeof(double)*3 , sizeof(double) };

  // return 2-D NumPy array
  return py::array(py::buffer_info(
    velocities.data(),                       /* data as contiguous array  */
    sizeof(double),                          /* size of one scalar        */
    py::format_descriptor<double>::format(), /* data type                 */
    ndim,                                    /* number of dimensions      */
    shape,                                   /* shape of the matrix       */
    strides                                  /* strides for each axis     */
  ));
} 

// Python wrapper for velocity corrections     
py::array py_FinitePartVelocity(npDoub pyChebPoints, npDoub pyForceDens, npDoub pyXs)
{
  /**
    Python wrapper to compute the finite part velocity on all fibers 
    @param pyChebPoints = Npts * 3 numpy array (2D) of Chebyshev points on ALL fibers
    @param pyForceDens = Npts * 3 numpy array (2D) of force densities on ALL fibers
    @param pyXs = Npts * 3 numpy array (2D) of tangent vectors on ALL fibers
    @return velocities = Npts * 3 numpy array (1D, row stacked "C" order) of finite part velocities
  **/
  
  // allocate std::vector (to pass to the C++ function)
  vec ChebPoints(pyChebPoints.size());
  vec ForceDensities(pyForceDens.size());
  vec XsVectors(pyXs.size());
  // copy py::array -> std::vector
  std::memcpy(ChebPoints.data(),pyChebPoints.data(),pyChebPoints.size()*sizeof(double));
  std::memcpy(ForceDensities.data(),pyForceDens.data(),pyForceDens.size()*sizeof(double));
  std::memcpy(XsVectors.data(),pyXs.data(),pyXs.size()*sizeof(double));

  // call pure C++ function
  vec velocities(pyChebPoints.shape()[0]*3,0.0);
  FinitePartVelocity(ChebPoints,ForceDensities,XsVectors, velocities);
  
  // allocate py::array (to pass the result of the C++ function to Python)
  auto pyArray = py::array_t<double>(pyChebPoints.shape()[0]*3);
  auto result_buffer = pyArray.request();
  double *result_ptr    = (double *) result_buffer.ptr;
  // copy std::vector -> py::array
  std::memcpy(result_ptr,velocities.data(),velocities.size()*sizeof(double));
  
  return pyArray;
}                            
    
// PYTHON MODULE DECLARATION
PYBIND11_MODULE(ManyFiberMethods, m) {
    m.doc() = "C++ module of functions for many fiber quadratures, etc."; // optional module docstring
    m.def("initFiberVars", &initFiberVars, "Initialize global C++ variables for each fiber");
    m.def("initSpecQuadParams", &initSpecQuadParams, "Initialize global C++ special quadrature parameters");
    m.def("initNodesandWeights", &py_initNodesandWeights, "Initialize nodes and weights for special quad");
    m.def("initResamplingMatrices", &py_initResamplingMatrices, "Initialize C++ resampling matrices");
    m.def("initFinitePartMatrix", &py_initFinitePartMatrix, "Initialize finite part matrix for C++");
    m.def("RPYFiberKernel", &py_RPYFiberKernel, "Evaluate the free space RPY kernel over a fiber");
    m.def("CorrectNonLocalVelocity", &py_CorrectNonLocalVelocity, "Evaluate corrections to the nonlocal velocity");
    m.def("FinitePartVelocity", &py_FinitePartVelocity, "Compute the finite part density");
}
