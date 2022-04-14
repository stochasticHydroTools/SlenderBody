import uammd
import numpy as np
import timeit

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
for i in range(0, 10):
    nlist.updateList(pos=positions,
                     Lx=lx, Ly=ly, Lz=lz,
                     numberParticles=numberParticles,
                     NperFiber=NperFiber,
                     rcut=rcut,
                     useGPU=useGPU)

Ntest = 100
start = timeit.default_timer()
for i in range(0, Ntest):
    nlist.updateList(pos=positions,
                     Lx=lx, Ly=ly, Lz=lz,
                     numberParticles=numberParticles,
                     NperFiber=NperFiber,
                     rcut=rcut,
                     useGPU=useGPU)

elapsed = timeit.default_timer() - start

print("Elapsed: "+str(elapsed*1000/Ntest)+" ms per test")

npairs = int(len(nlist.pairList)/2)
print("Found ", npairs, "pairs")
i = 1
# Reshape and sort the list to check for correctness more easily
nl = nlist.pairList.reshape((npairs, 2))
for i in range(0, npairs):
    if nl[i, 0] > nl[i, 1]:
        a = nl[i, 0]
        nl[i, 0] = nl[i, 1]
        nl[i, 1] = a
nl = np.sort(nl, axis=0)

# Print some particle pairs
for j in range(0, 10):
    ii = nl[j, 0]
    jj = nl[j, 1]
    print(ii, jj)
