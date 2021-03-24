#include"uammd.cuh"
#include"omp.h"
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


struct HeadAndList{
  
  std::vector<int> head, list;
  int3 cells;
  real3 boxSize;
    inline int3 indexToCell3D(int icell, int3 ncells){
    int3 cell;
    cell.x = icell%3 - 1;
    cell.y = (icell/3)%3 - 1;
    cell.z = (icell/9) - 1;
    return cell;
  }
  
  inline real3 apply_mic(real3 pos, real3 boxSize){
    return pos - floorf((pos/boxSize + real(0.5)))*boxSize;
  }

  inline int3 pbc_cell(int3 cell, int3 ncells){
    if(cell.x < 0) cell.x += ncells.x;
    if(cell.x >= ncells.x) cell.x -= ncells.x;
    if(cell.y < 0) cell.y += ncells.y;
    if(cell.y >= ncells.y) cell.y -= ncells.y;
    if(cell.z < 0) cell.z += ncells.z;
    if(cell.z >= ncells.z) cell.z -= ncells.z;
    return cell;
  }

  void initialize(int numberParticles, real3 i_boxSize, real cutOff){
    this->boxSize = i_boxSize;
    this->cells = make_int3(boxSize/cutOff);
    cells.x = std::max(cells.x, 3);
    cells.y = std::max(cells.y, 3);
    cells.z = std::max(cells.z, 3);
    head.resize(cells.x*cells.y*cells.z);
    list.resize(numberParticles+1);
    std::fill(head.begin(), head.end(), 0);
 } 

  void updateList(real3* pos, real3 i_boxSize, real cutOff, int N){
    initialize(N, i_boxSize, cutOff);
    for(int i = 0; i<N; i++){
      int icell = getCellIndex(getCell(pos[i]));
      list[i+1] = head[icell];
      head[icell] = i+1;
    }
  }
  
  inline int3 getCell(real3 pos){
    auto posInBox = apply_mic(pos, this->boxSize);
    return make_int3((posInBox/this->boxSize + 0.5)*make_real3(this->cells));
  }

  inline int getCellIndex(int3 ic){
    return ic.x + (ic.y + ic.z*cells.y)*cells.x;
  }

  inline int getNeighbourCellIndex(int ic, int3 cell_i){
    const int3 cell_j = pbc_cell(cell_i + indexToCell3D(ic, cells), cells);
    const int jcell = getCellIndex(cell_j);
    return jcell;
  }
  
};



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
  //CellListBase d_nl;
  NlistCPU h_nl;
  thrust::device_vector<real4> pos;
  thrust::device_vector<real3> tmp;
  thrust::device_vector<int> errorStatus;
  
  cudaStream_t st;
  real rcut;
  int NperFiber;

  HeadAndList linkList;
  bool GPUMode = false;
public:
  py::array_t<int> nneigh;
  int stride;
  py::array_t<int> list;

  
  NListUAMMD(){
    this->stride = 4;
  }
  
  ~NListUAMMD(){
    if(GPUMode){
      cudaDeviceSynchronize();
      cudaStreamDestroy(st);
    }
  }
  
    void updateList(py::array_t<real> h_pos, real Lx, real Ly, real Lz, int numberParticles,
     int NperFiber, real rcut, bool useGPU, int maxNeighbors, int nThr){
        if (numberParticles < 0 or rcut < 0){
           std::cerr<<"ERROR: Invalid parameters"<<std::endl;
           return;
        }
        linkList.updateList((real3*)h_pos.data(), {Lx, Ly, Lz}, rcut, numberParticles);
        bool failed;
        try { 
            failed = fillCPUListWithLinkList((real3*)h_pos.data(), {Lx, Ly, Lz}, rcut, numberParticles, NperFiber, maxNeighbors, nThr);
        } catch (const char* msg){
        } 
        if (failed){
            throw "Not enough max neighbors!";
        }
        h_nl.list.erase(std::remove(h_nl.list.begin(),h_nl.list.end(),-1),h_nl.list.end());
        this->nneigh = py::array_t<int>(h_nl.nneigh.size(), h_nl.nneigh.data());
        this->stride = h_nl.stride;
        this->list = py::array_t<int>(h_nl.list.size(), h_nl.list.data());

    }

    bool fillCPUListWithLinkList(real3* pos, real3 L, real rcut, int N, int NperFiber, int maxNeighbors, int nThr){
    constexpr int numberNeighbourCells = 27;
    Box box(L);
    const real rc2 = rcut*rcut;
    h_nl.stride = 2*maxNeighbors;
    h_nl.nneigh.resize(N);
    h_nl.list.resize(h_nl.stride*N);
    std::fill(h_nl.list.begin(), h_nl.list.end(), -1);
    bool NotEnoughMaxNeighbors = false;
    # pragma omp parallel for num_threads(nThr)
    for(int i = 0; i<N; i++){
        const real3 pi = pos[i];
        const int3 cell_i = linkList.getCell(pos[i]);
        const int fiber_i = i/NperFiber;
        int nneigh = 0;
        for(int ic = 0; ic<numberNeighbourCells; ic++){
            int jcell = linkList.getNeighbourCellIndex(ic, cell_i);
            int j = linkList.head[jcell];
            while(j!=0){
              int jj = j-1;
              const int fiber_j = jj/NperFiber;
              if(fiber_i != fiber_j && i < jj){
                const real3 pj = pos[jj];
                const auto rij = box.apply_pbc(pi-pj);
                const real r2 = dot(rij, rij);
                if(r2 <  rc2){
                  if(nneigh >= maxNeighbors){
                    NotEnoughMaxNeighbors = true;
                  }
                  h_nl.list[h_nl.stride*i + 2*nneigh] = i;
                  h_nl.list[h_nl.stride*i + 2*nneigh+1] = jj;
                  nneigh++;
                } 
              }
            j = linkList.list[j];
            }
      }
      h_nl.nneigh[i] = nneigh;
    }
    return NotEnoughMaxNeighbors;
  }

};

using namespace pybind11::literals;
PYBIND11_MODULE(NeighborSearch, m) {
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
	"rcut"_a,
	"useGPU"_a = true,
	"maxNeighbors"_a = 10,
	"nThr"_a=1).
    def_readonly("nneigh", &NListUAMMD::nneigh, "Number of neighbours per particle").
    def_readonly("stride", &NListUAMMD::stride, "Offset between particles in the list").
    def_readonly("list", &NListUAMMD::list, "List of neighbours per particle, neighbour j of particle i (of a total nneigh[i]) is at list[i*stride + j]");     
    }

