#include"uammd.cuh"
#include"omp.h"
#include"Interactor/NeighbourList/CellList/CellListBase.cuh"
#include"Interactor/NeighbourList/CellList/NeighbourContainer.cuh"
using namespace uammd;
template<class T> using gpu_container = thrust::device_vector<T>;

struct NlistCPU{
  //Neighbour j of particle i (of a total of nneigh[i] neighbours): list[stride*i + j]
  std::vector<int> nneigh;
  int stride;
  std::vector<int> list;
};

#include<pybind11/pybind11.h> //Basic interfacing utilities
#include<pybind11/numpy.h>    //Utilities to work with numpy arrays
namespace py = pybind11;


//The operator () of this object returns its input as a real4
struct ToReal4{
  template<class vectype>
  __host__ __device__ real4 operator()(vectype i){
    auto pr4 = make_real4(i);
    return pr4;
  }
};

Grid createUpdateGrid(Box box, real3 cutOff){
  real3 L = box.boxSize;
  constexpr real inf = std::numeric_limits<real>::max();
  //If the box is non periodic L and cellDim are free parameters
  //If the box is infinite then periodicity is irrelevan
  constexpr int maximumNumberOfCells = 64;
  if(L.x >= inf) L.x = maximumNumberOfCells*cutOff.x;
  if(L.y >= inf) L.y = maximumNumberOfCells*cutOff.y;
  if(L.z >= inf) L.z = maximumNumberOfCells*cutOff.z;
  Box updateBox(L);
  updateBox.setPeriodicity(box.isPeriodicX() and L.x < inf, box.isPeriodicY() and L.y<inf, box.isPeriodicZ() and L.z<inf);
  Grid a_grid = Grid(updateBox, cutOff);
  int3 cellDim = a_grid.cellDim;
  if(cellDim.x <= 3) cellDim.x = 1;
  if(cellDim.y <= 3) cellDim.y = 1;
  if(cellDim.z <= 3) cellDim.z = 1;
  a_grid = Grid(updateBox, cellDim);
  return a_grid;
}


class NListUAMMD{
  CellListBase d_nl;
  NlistCPU h_nl;
  thrust::device_vector<real4> pos;
  thrust::device_vector<real3> tmp;
  thrust::device_vector<int> errorStatus;
  
  cudaStream_t st;
  real rcut;
  int NperFiber;

public:
  py::array_t<int> nneigh;
  int stride;
  py::array_t<int> list;

  
  NListUAMMD(){
    this->stride = 4;
  }
  
  ~NListUAMMD(){
      cudaDeviceSynchronize();
      cudaStreamDestroy(st);
  }
  
  void updateList(py::array_t<real> h_pos, real Lx, real Ly, real Lz, int numberParticles,
     int NperFiber, real rcut){
    if(numberParticles < 0 or rcut < 0){
      std::cerr<<"ERROR: Invalid parameters"<<std::endl;
      return;
    }

	CudaSafeCall(cudaStreamCreate(&st));
	errorStatus.resize(1);
    updateListGPU(h_pos, Lx, Ly, Lz, numberParticles, NperFiber, rcut);
    h_nl.list.erase(std::remove(h_nl.list.begin(),h_nl.list.end(),-1),h_nl.list.end());
    this->nneigh = py::array_t<int>(h_nl.nneigh.size(), h_nl.nneigh.data());
    this->stride = h_nl.stride;
    this->list = py::array_t<int>(h_nl.list.size(), h_nl.list.data());

  }

  void updateListGPU(py::array_t<real> &h_pos, real Lx, real Ly, real Lz, int numberParticles, int NperFiber, real rcut){
    this->NperFiber = NperFiber;
    Box box({Lx, Ly, Lz});
    this->rcut = rcut;
    pos.resize(numberParticles);
    tmp.resize(numberParticles);
    thrust::copy((real3*)h_pos.data(), (real3*)h_pos.data() + numberParticles, tmp.begin());
    thrust::transform(thrust::cuda::par.on(st), tmp.begin(), tmp.end(), pos.begin(), ToReal4());
    auto p_ptr = thrust::raw_pointer_cast(pos.data());
    Grid grid = createUpdateGrid(box, {rcut, rcut, rcut});
    d_nl.update(p_ptr, numberParticles, grid, st);
    downloadList();
  }
  
  void downloadList(){
    static thrust::device_vector<int> d_list;
    static thrust::device_vector<int> d_nneigh;
    auto listDataGPU = d_nl.getCellList();
    int numberParticles = pos.size();
    auto box = listDataGPU.grid.box;
    const real rc2 = rcut*rcut;
    bool tooManyNeighbours;
    auto err_ptr = thrust::raw_pointer_cast(errorStatus.data());
    int maxNeighboursPerParticle = this->stride;
    auto cit = thrust::make_counting_iterator<int>(0);
    int Nexclude = this->NperFiber;
    do{
      errorStatus[0] = 0;
      d_list.resize(numberParticles*maxNeighboursPerParticle);
      d_nneigh.resize(numberParticles);
      auto d_list_ptr = thrust::raw_pointer_cast(d_list.data());
      auto d_nneigh_ptr = thrust::raw_pointer_cast(d_nneigh.data());
      thrust::for_each(thrust::cuda::par.on(0),
		       cit, cit + numberParticles,
		       [=] __device__ (int tid){
			 auto nc = CellList_ns::NeighbourContainer(listDataGPU);
			 const int ori = nc.getGroupIndexes()[tid];
			 nc.set(tid);
			 for (int i =0; i < maxNeighboursPerParticle; i++){
			    d_list_ptr[maxNeighboursPerParticle*ori+i] = -1;   
			 }
			 const real3 pi = make_real3(nc.getSortedPositions()[tid]);
			 auto it = nc.begin();
			 int nneigh = 0;
			 const int fiber_i = ori/Nexclude;
			 while(it){
			   auto neighbour = *it++;
			   const int j = neighbour.getGroupIndex();
			   const int fiber_j = j/Nexclude;
			   if(fiber_i != fiber_j && j > ori){
			     const real3 pj = make_real3(neighbour.getPos());
			     const auto rij = box.apply_pbc(pi-pj);
			     const real r2 = dot(rij, rij);
			     if(r2 <  rc2){
			       nneigh++;
			       if(nneigh >= maxNeighboursPerParticle){
				 err_ptr[0] = 1;
				 d_nneigh_ptr[ori] = 0;
				 return;
			       }
			       d_list_ptr[maxNeighboursPerParticle*ori + nneigh - 1] = j;
			     }
			   }
			 }
			 d_nneigh_ptr[ori] = nneigh;
		       });
      tooManyNeighbours = errorStatus[0] != 0;
      if(tooManyNeighbours){
	maxNeighboursPerParticle += 4;
	System::log<System::WARNING>("Increasing max neighbours to %d", maxNeighboursPerParticle);
      }
    }while(tooManyNeighbours);
    this->stride = maxNeighboursPerParticle;
    h_nl.stride = this->stride;
    h_nl.nneigh.resize(numberParticles);
    h_nl.list.resize(h_nl.stride*numberParticles);
    thrust::copy(d_list.begin(), d_list.end(), h_nl.list.begin());
    thrust::copy(d_nneigh.begin(), d_nneigh.end(), h_nl.nneigh.begin());
    CudaCheckError();
  }
};

using namespace pybind11::literals;
PYBIND11_MODULE(NeighborSearchGPU, m) {
  m.doc() = "UAMMD NieghbourList CPU interface";
  //Lets expose the UAMMD class defined above under the name "LJ"
  py::class_<NListUAMMD>(m, "NList").
    def(py::init()).
    def("updateList", &NListUAMMD::updateList, "Update list with the provided positions and parameters",
	"pos"_a,
	"Lx"_a = std::numeric_limits<real>::infinity(),
	"Ly"_a = std::numeric_limits<real>::infinity(),
	"Lz"_a = std::numeric_limits<real>::infinity(),
	"numberParticles"_a,
	"NperFiber"_a = 1,
	"rcut"_a).
    def_readonly("nneigh", &NListUAMMD::nneigh, "Number of neighbours per particle").
    def_readonly("stride", &NListUAMMD::stride, "Offset between particles in the list").
    def_readonly("list", &NListUAMMD::list, "List of neighbours per particle, neighbour j of particle i (of a total nneigh[i]) is at list[i*stride + j]");     
    }

