/* Raul P. Pelaez 2021
   Neighbour list with a python interface. By default the resulting python module is named "uammd".
   import it from python and call help(uammd) for more information.

   Once updateList is called, the neighbour list is provided via a member called "pairList".
   pairList contains a list of indices of particles that are neighbours.
   It is a 2D array stored contigously, so that the particle with index pairList[2*i] is a neighbour of the particle with index pairList[2*i+1].
   The total number of pairs is encoded in the length of this list ( len(pairList)/2 from python).

   The order of the pairs is arbitrary and each pair is included only once.
   
   Additionally, the NperFiber argument allows to exclude particles that are in the same "group". 
   See the function isPairExcluded to customize this behavior.

   USAGE EXAMPLE:
   import uammd;
   numberParticles = 13840
   lx = ly = lz = 32.0
   rcut = 1
   NperFiber = 1 # Number of particles in each fiber, 1 is the minimum and excludes self interactions
   nlist = uammd.NList()
   np.random.seed(1234)
   precision = np.float32
   positions = np.array(np.random.rand(numberParticles, 3), precision)
   positions[:, 0] *= lx
   positions[:, 1] *= ly
   positions[:, 2] *= lz
   useGPU = False
   nlist.updateList(pos=positions, Lx=lx, Ly=ly, Lz=lz, numberParticles=numberParticles, NperFiber=NperFiber, rcut=rcut, useGPU=useGPU)
   nl = nlist.pairList.reshape((npairs, 2))
   # Each element of nl contains a pair of neighbour particles
   print(nl[0,0], "is a neighbour of ", nl[0,1])
   
   

 */
#include<iostream>
#include<tbb/concurrent_vector.h>
#include<tbb/parallel_for.h>
#include<tbb/global_control.h>
#include<thread>
#include<vector>
#ifndef __CUDACC__
#include"vector.h"
using namespace utils;
#else
#include"uammd.cuh"
// #include"Interactor/NeighbourList/CellList/CellListBase.cuh"
// #include"Interactor/NeighbourList/CellList/NeighbourContainer.cuh"
using namespace uammd;
#endif

//These helper functions allow to grow an std or concurrent tbb container using the same interface
template<class GrowableContainer>
struct ContainerAdaptor{
  template<class T> static void append(GrowableContainer & gc, std::initializer_list<T> arg){    
    gc.insert(gc.end(), arg);
  }
};
template<class VecType>
struct ContainerAdaptor<tbb::concurrent_vector<VecType>>{
  template<class T> static void append(tbb::concurrent_vector<VecType> & gc, std::initializer_list<T> arg){
    gc.grow_by(arg);
  }
};

template<class GrowableContainer, class T> inline void append_to_container(GrowableContainer & gc, std::initializer_list<T> arg){
  ContainerAdaptor<GrowableContainer>::append(gc, arg);
}

//Apply minimum image convention
inline real3 apply_mic(real3 pos, real3 boxSize){
  return pos - floorf((pos/boxSize + real(0.5)))*boxSize;
}


//Given a number between 0 and 27 this function will return a 3d coordinate betweem [-1 -1 -1] and [1 1 1]
inline int3 linearIndexTo3D_3x3(int icell){
  int3 cell;
  cell.x = icell%3 - 1;
  cell.y = (icell/3)%3 - 1;
  cell.z = (icell/9) - 1;
  return cell;    
}

//Folds a 3D index to the range given by ncells.
//Only allows for cell coordinates that are at most "ncells" apart from the main range.
inline int3 pbc_cell(int3 cell, int3 ncells){
  if(cell.x < 0) cell.x += ncells.x;
  if(cell.x >= ncells.x) cell.x -= ncells.x;
  if(cell.y < 0) cell.y += ncells.y;
  if(cell.y >= ncells.y) cell.y -= ncells.y;
  if(cell.z < 0) cell.z += ncells.z;
  if(cell.z >= ncells.z) cell.z -= ncells.z;
  return cell;
}

//Head and list cell list algorithm
class HeadAndList{
  //Initializes and cleans the cell list
  void initialize(int numberParticles, real3 i_boxSize, real cutOff){
    this->boxSize = i_boxSize;
    this->cells = make_int3(boxSize/cutOff);
    //The algorithm expects at least 3 cells per direction
    cells.x = std::max(cells.x, 3);
    cells.y = std::max(cells.y, 3);
    cells.z = std::max(cells.z, 3);
    head.resize(cells.x*cells.y*cells.z);
    list.resize(numberParticles+1); //+1 to index particles from 1 to N later on
    std::fill(head.begin(), head.end(), 0);
  }
public:
  std::vector<int> head, list;
  int3 cells; //Number of cells in each direction
  real3 boxSize; //Current box size

  //Update the list with the positions in pos
  void updateList(const real3* pos, real3 i_boxSize, real cutOff, int N){
    initialize(N, i_boxSize, cutOff);
    for(int i = 0; i<N; i++){ //The head-and-list algorithm
      int icell = getCellIndex(getCell(pos[i]));
      list[i+1] = head[icell];
      head[icell] = i+1;
    }
  }
  
  //Get the cell of a certain position, the position is folded to the main box using mic
  inline int3 getCell(real3 pos){
    auto posInBox = apply_mic(pos, this->boxSize);
    return make_int3((posInBox/this->boxSize + 0.5)*make_real3(this->cells));
  }

  //Get the index of a 3D cell in the head array
  inline int getCellIndex(int3 ic){
    return ic.x + (ic.y + ic.z*cells.y)*cells.x;
  }

  //Returns the index of a neighbouring cell. The neighbour index goes from 0 to 27
  inline int getNeighbourCellIndex(int ic, int3 cell_i){
    const int3 cell_j = pbc_cell(cell_i + linearIndexTo3D_3x3(ic), cells);
    const int jcell = getCellIndex(cell_j);
    return jcell;
  }
  
};

//This rule allows prevent certain pais of particles form becoming neighbours
inline bool isPairExcluded(int i, int j, int NperFiber){
  if(i>j) return true;
  const int fiber_i = i/NperFiber;
  const int fiber_j = j/NperFiber;
  if(fiber_i == fiber_j) return true;
  return false;
}

inline real areParticlesClose(real3 ri, real3 rj, real3 L, real rcut2){
  const auto rij = apply_mic(ri-rj, L);
  const real r2 = dot(rij, rij);
  return r2<rcut2;
}

//This class provides a neighbour list (as defined in the header comments) using a CPU only algorithm.
class NListCPU{
  HeadAndList cellList;
  real rcut;
  int NperFiber;
  size_t maximumPairs = 100;
  real3 L;
  int numberParticles;
public:
  std::vector<int> pairList;

  NListCPU(){
    growPairListIfNecessary();
  }

  //Updates the neighbour list with the provided parameters
  void update(const real3* h_pos, real Lx, real Ly, real Lz, int numberParticles, int NperFiber, real rcut, int nThreads){
    //Store current parameters
    this->NperFiber = NperFiber;
    this->L = {Lx, Ly, Lz};
    this->numberParticles = numberParticles;
    this->rcut = rcut;    
    cellList.updateList(h_pos, L, rcut, numberParticles);
    //Create the pair list using the cell list
    //Uncomment these lines to force a certain number of threads
    using gc = tbb::global_control;
    std::unique_ptr<gc> control;
    if(nThreads>0) control = std::make_unique<gc>(gc::max_allowed_parallelism, nThreads);
    createPairListTraverseCells(h_pos);
  }
private:

  //Reserve more space for the pair list if it has grown beyond the previous maximum expected
  void growPairListIfNecessary(){
    if(pairList.size() > 2*maximumPairs or pairList.size() == 0){
      //Increase the maximum number of pairs
      maximumPairs = pairList.size()/2 + 1000;
      //Advice the vector of the new maximum expected size
      pairList.reserve(2*maximumPairs);
    }
  }

  /*Given a particle index, "i", and a cell index (from the cell list) appends to the list any neighbours of "i" located in that cell.
  Arguments:
     i: Particle index
     pi: Position of particle i
     pos: Array with all positions
     jcell: Index of a cell
     gc: The pair list, any neighbours found will be appended to this list (increasing its size)
  */
  template<class GrowableContainer>
  inline void processCell(int i, real3 pi, const real3* pos, int jcell, GrowableContainer &gc){
    const real rc2 = this->rcut*this->rcut;
    int currentLink = cellList.head[jcell];
    while(currentLink!=0){
      const int j = currentLink-1;
      if( not isPairExcluded(i, j, this->NperFiber) and areParticlesClose(pi, pos[j], this->L, rc2)){
	append_to_container(gc, {i,j});
      }
      currentLink = cellList.list[currentLink];
    }
  }

  //Resets the neighbour list and constructs it again in parallel from the cell list.
  //Assigns a thread per particle
  void createPairListTraverseParticles(const real3* pos){
    //This container can be dynamically grown in the parallel environment
    static tbb::concurrent_vector<int> plt;
    //Advise the parallel container on the maximum number of elements
    plt.reserve(2*maximumPairs);   
    //The list is going to be grown in parallel
    plt.resize(0);
    //Launch a worker for each particle
    tbb::parallel_for(tbb::blocked_range<size_t>(0, numberParticles),
		      [&](const auto &block){
			for(size_t i = block.begin(); i!=block.end(); ++i){
			  //For a given particle, traverse its 27 neighbour cells
			  const real3 pi = pos[i];
			  const int3 cell_i = cellList.getCell(pi);
			  //The order of traversal is x,y,z, the same as the order in memory
			  //Doing it this way (instead of three nested loops) saves two loops and increases the chances of unrolling
			  constexpr int numberNeighbourCells = 27;
			  for(int ic = 0; ic<numberNeighbourCells; ic++){
			    const int jcell = cellList.getNeighbourCellIndex(ic, cell_i);
			    //Append neighbours in cell "jcell" to the list
			    processCell(i, pi, pos, jcell, plt);
			  }
			}
		      }
		      );
    //Gather the concurrent container into a contigous std::vector
    pairList.assign(plt.begin(), plt.end());
    growPairListIfNecessary();
  }

  //Resets the neighbour list and constructs it again in parallel from the cell list.
  //Assigns a thread per cell in the cell list, so that each threads serially processes each particle in a cell
  void createPairListTraverseCells(const real3* pos){
    //This container can be dynamically grown in the parallel environment
    static tbb::concurrent_vector<int> plt;
    //Advise the parallel container on the maximum number of elements
    plt.reserve(2*maximumPairs);   
    //The list is going to be grown in parallel
    plt.resize(0);
    int ncells = cellList.head.size();
    //Launch a worker for each cell
    tbb::parallel_for(tbb::blocked_range<size_t>(0, ncells),
		      [&](const auto &block){
			for(size_t icell = block.begin(); icell!=block.end(); ++icell){
			  //For a given cell, process all the particles in it
			  int currentLink = cellList.head[icell];
			  while(currentLink){
			    const int i = currentLink -1;
			    const real3 pi = pos[i];
			    const int3 cell_i = cellList.getCell(pi);
			    constexpr int numberNeighbourCells = 27;
			    //The order of traversal is x,y,z, the same as the order in memory
			    //Doing it this way (instead of three nested loops) saves two loops and increases the chances of unrolling
			    for(int ic = 0; ic<numberNeighbourCells; ic++){
			      const int jcell = cellList.getNeighbourCellIndex(ic, cell_i);
			      //Append neighbours in cell "jcell" to the list
			      processCell(i, pi, pos, jcell, plt);
			    }
			    currentLink = cellList.list[currentLink]; 
			  }
			}
		      }
		      );
    //Gather the concurrent container into a contigous std::vector
    pairList.assign(plt.begin(), plt.end());
    growPairListIfNecessary();
  }
  
  //This serial version is left here for performance measurements. Using the tbb version above with 1 thread is something like 15% slower than this
  
  void createPairListTraverseParticlesSerial(const real3* pos){
    pairList.resize(0);
    for(int i = 0; i < numberParticles; i++){
      const real3 pi = pos[i];
      const int3 cell_i = cellList.getCell(pos[i]);
      constexpr int numberNeighbourCells = 27;
      for(int ic = 0; ic<numberNeighbourCells; ic++){
	const int jcell = cellList.getNeighbourCellIndex(ic, cell_i);
	processCell(i, pi, pos, jcell, pairList);
      }
    }
    growPairListIfNecessary();
  }
   

};



// #ifdef GPU_MODE
// //The operator () of this object returns its input as a real4
// struct ToReal4{
//   template<class vectype>
//   __host__ __device__ real4 operator()(vectype i){
//     auto pr4 = make_real4(i);
//     return pr4;
//   }
// };

// class NListGPU{
//   template<class T> using gpu_container = thrust::device_vector<T>;
//   CellListBase d_nl;
//   gpu_container<real4> pos;
//   gpu_container<real3> tmp;
//   gpu_container<int> errorStatus;  
//   cudaStream_t st;
//   int stride;
//   int NperFiber;
//   real rcut;
//   Grid createUpdateGrid(Box box, real3 cutOff){
//     real3 L = box.boxSize;
//     constexpr real inf = std::numeric_limits<real>::max();
//     //If the box is non periodic L and cellDim are free parameters
//     //If the box is infinite then periodicity is irrelevan
//     constexpr int maximumNumberOfCells = 64;
//     if(L.x >= inf) L.x = maximumNumberOfCells*cutOff.x;
//     if(L.y >= inf) L.y = maximumNumberOfCells*cutOff.y;
//     if(L.z >= inf) L.z = maximumNumberOfCells*cutOff.z;
//     Box updateBox(L);
//     updateBox.setPeriodicity(box.isPeriodicX() and L.x < inf, box.isPeriodicY() and L.y<inf, box.isPeriodicZ() and L.z<inf);
//     Grid a_grid = Grid(updateBox, cutOff);
//     int3 cellDim = a_grid.cellDim;
//     if(cellDim.x <= 3) cellDim.x = 1;
//     if(cellDim.y <= 3) cellDim.y = 1;
//     if(cellDim.z <= 3) cellDim.z = 1;
//     a_grid = Grid(updateBox, cellDim);
//     return a_grid;
//   }
// public:
    
//   NListGPU(){
//       this->stride = 4;
//       CudaSafeCall(cudaStreamCreate(&st));
//       errorStatus.resize(1);
//     }
  
//   ~NListGPU(){
//       cudaDeviceSynchronize();
//       cudaStreamDestroy(st);
//     }

//   void update(real3* h_pos, real Lx, real Ly, real Lz, int numberParticles, int NperFiber, real rcut){
//     this->NperFiber = NperFiber;
//     Box box({Lx, Ly, Lz});
//     this->rcut = rcut;
//     pos.resize(numberParticles);
//     tmp.resize(numberParticles);
//     thrust::copy((real3*)h_pos.data(), (real3*)h_pos.data() + numberParticles, tmp.begin());
//     thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), pos.begin(), ToReal4());
//     auto p_ptr = thrust::raw_pointer_cast(pos.data());
//     Grid grid = createUpdateGrid(box, {rcut, rcut, rcut});
//     d_nl.update(p_ptr, numberParticles, grid, st);
//     downloadList();
//   }
  
//   void downloadList(){
//     static thrust::device_vector<int> d_list;
//     static thrust::device_vector<int> d_nneigh;
//     auto listDataGPU = d_nl.getCellList();
//     int numberParticles = pos.size();
//     auto box = listDataGPU.grid.box;
//     const real rc2 = rcut*rcut;
//     bool tooManyNeighbours;
//     auto err_ptr = thrust::raw_pointer_cast(errorStatus.data());
//     int maxNeighboursPerParticle = this->stride;
//     auto cit = thrust::make_counting_iterator<int>(0);
//     int Nexclude = this->NperFiber;
//     do{
//       errorStatus[0] = 0;
//       d_list.resize(numberParticles*maxNeighboursPerParticle);
//       d_nneigh.resize(numberParticles);
//       auto d_list_ptr = thrust::raw_pointer_cast(d_list.data());
//       auto d_nneigh_ptr = thrust::raw_pointer_cast(d_nneigh.data());
//       thrust::for_each(thrust::cuda::par.on(0),
// 		       cit, cit + numberParticles,
// 		       [=] __device__ (int tid){
// 			 auto nc = CellList_ns::NeighbourContainer(listDataGPU);
// 			 const int ori = nc.getGroupIndexes()[tid];
// 			 nc.set(tid);
// 			 const real3 pi = make_real3(nc.getSortedPositions()[tid]);
// 			 auto it = nc.begin();
// 			 int nneigh = 0;
// 			 const int fiber_i = ori/Nexclude;
// 			 while(it){
// 			   auto neighbour = *it++;
// 			   const int j = neighbour.getGroupIndex();
// 			   const int fiber_j = j/Nexclude;
// 			   if(fiber_i != fiber_j){
// 			     const real3 pj = make_real3(neighbour.getPos());
// 			     const auto rij = box.apply_pbc(pi-pj);
// 			     const real r2 = dot(rij, rij);
// 			     if(r2 <  rc2){
// 			       nneigh++;
// 			       if(nneigh >= maxNeighboursPerParticle){
// 				 err_ptr[0] = 1;
// 				 d_nneigh_ptr[ori] = 0;
// 				 return;
// 			       }
// 			       d_list_ptr[maxNeighboursPerParticle*ori + nneigh - 1] = j;
// 			     }
// 			   }
// 			 }
// 			 d_nneigh_ptr[ori] = nneigh;
// 		       });
//       tooManyNeighbours = errorStatus[0] != 0;
//       if(tooManyNeighbours){
// 	maxNeighboursPerParticle += 4;
// 	System::log<System::WARNING>("Increasing max neighbours to %d", maxNeighboursPerParticle);
//       }
//     }while(tooManyNeighbours);
//     this->stride = maxNeighboursPerParticle;
//     h_nl.stride = this->stride;
//     h_nl.nneigh.resize(numberParticles);
//     h_nl.list.resize(h_nl.stride*numberParticles);
//     thrust::copy(d_list.begin(), d_list.end(), h_nl.list.begin());
//     thrust::copy(d_nneigh.begin(), d_nneigh.end(), h_nl.nneigh.begin());
//     CudaCheckError();
//   }
// };
// #else
using NListGPU = NListCPU;
// #endif

#ifdef PYTHON_LIBRARY_MODE
#include<pybind11/pybind11.h> //Basic interfacing utilities
#include<pybind11/numpy.h>    //Utilities to work with numpy arrays
namespace py = pybind11;

//This shuts up compiler warnings when compiling as a shared library
#pragma GCC visibility push(hidden)
//This class interfaces the CPU and GPU lists with python
class PyNeighbourList{
  NListGPU gpuList;
  NListCPU cpuList;
  int _nThr;
public:
  py::array_t<int> pairList;
  
  PyNeighbourList(int nThr):_nThr(nThr){}  
  
  void updateList(py::array_t<real> &h_pos, real Lx, real Ly, real Lz, int numberParticles, int NperFiber, real rcut, bool useGPU){
    if(numberParticles <= 0 or rcut <= 0){
      std::cerr<<"ERROR: Invalid parameters"<<std::endl;
      return;
    }
    if(not useGPU){
      cpuList.update((real3*)h_pos.data(), Lx, Ly, Lz, numberParticles, NperFiber, rcut, _nThr);
      this->pairList = py::array_t<int>(cpuList.pairList.size(), cpuList.pairList.data());
    }
    else{
      gpuList.update((real3*)h_pos.data(), Lx, Ly, Lz, numberParticles, NperFiber, rcut, _nThr);
      this->pairList = py::array_t<int>(gpuList.pairList.size(), gpuList.pairList.data());
    }
  }  

};
#pragma GCC visibility pop

using namespace pybind11::literals;
PYBIND11_MODULE(NeighborSearch, m) {
  m.doc() = "UAMMD NieghbourList CPU interface";
  //Lets expose the UAMMD class defined above under the name "LJ"
  py::class_<PyNeighbourList>(m, "NList").
    def(py::init<int>(),
	"Initialize the list, optionally a maximum numbe rof threads can be provided. By default all available cores will be used.",
	"nThreads"_a = -1).
    def("updateList", &PyNeighbourList::updateList, "Update list with the provided positions and parameters",
	"pos"_a,
	"Lx"_a = std::numeric_limits<real>::infinity(),
	"Ly"_a = std::numeric_limits<real>::infinity(),
	"Lz"_a = std::numeric_limits<real>::infinity(),
	"numberParticles"_a,
	"NperFiber"_a = 1,
	"rcut"_a,
	"useGPU"_a = true).
    def_readonly("pairList", &PyNeighbourList::pairList, "List of neighbour particle pairs");
}

#endif
