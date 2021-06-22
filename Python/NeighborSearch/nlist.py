#import NeighborSearch
from SpatialDatabase import ckDSpatial, CellLinkedList
from Domain import PeriodicShearedDomain
from scipy.spatial import cKDTree
import numpy as np
import time

numberParticles = 80000
lx = ly = lz = 2.4
rcut = 0.07500000000000001

#nlist = NeighborSearch.NList()

np.random.seed(1)
precision = np.float32
positions = np.array(np.random.rand(numberParticles, 3), precision)
positions[:, 0] *= 2*lx
positions[:, 1] *= 2*ly
positions[:, 2] *= 2*lz

nThr=16;
"""
thist = time.time()

nlist.updateList(pos=positions,
             Lx=lx, Ly=ly, Lz=lz,
             numberParticles=numberParticles,
             rcut=rcut,useGPU=False,maxNeighbors=25,nThr=nThr)
print('GPU Update list time %f' %(time.time()-thist))
thist = time.time()
neighbors=nlist.list;
AllNeighbors = np.reshape(neighbors,(len(neighbors)//2,2));
print('GPU sort time %f' %(time.time()-thist))
print(AllNeighbors.shape)
Neighbs1=np.savetxt('N1.txt',AllNeighbors)
"""

# Check with ckd tree
Dom = PeriodicShearedDomain(lx,ly,lz);
SpatialChk = ckDSpatial(positions,Dom);
thist = time.time()
SpatialChk.updateSpatialStructures(positions,Dom);
AllNeighbors2 = SpatialChk.selfNeighborList(rcut*1.1)
print('CPU Update list time %f' %(time.time()-thist))
print(AllNeighbors2.shape)
Neighbs2=np.savetxt('N2.txt',AllNeighbors2)

# Check with linked list class
Dom = PeriodicShearedDomain(lx,ly,lz);
SpatialChk = CellLinkedList(positions,Dom,nThr=nThr);
thist = time.time()
SpatialChk.updateSpatialStructures(positions,Dom);
AllNeighbors2 = SpatialChk.selfNeighborList(rcut)
print('Class LL Update list time %f' %(time.time()-thist))
SpatialChk.updateSpatialStructures(positions,Dom);
thist = time.time()
AllNeighbors2 = SpatialChk.selfNeighborList(rcut*1.1)
print('Class LL 2nd Update list time %f' %(time.time()-thist))
print(AllNeighbors2.shape)
Neighbs2=np.savetxt('N3.txt',AllNeighbors2)

