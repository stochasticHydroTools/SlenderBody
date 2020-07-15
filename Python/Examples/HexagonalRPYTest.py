from RPYVelocityEvaluator import EwaldSplitter
from SpatialDatabase import SpatialDatabase,ckDSpatial, LinkedListSpatial
from Domain import Domain, PeriodicShearedDomain
import numpy as np
import time

"""
This file performs the hexagonal packing test in Appendix B of our paper. 
"""

# The hexagonal array calculation
# Initialization
g = 0.5; # strain for parallelepiped
Lx = 2.0;
Ly = 2.0;
Lz = 2.0;
a = 1e-2;
mu = 3.0;
xi = 5.0;
Npts=4;
pts = np.array([[0,0,0],[1,0,0],[0.5,1,0],[1.5,1,0]]);
forces = np.array([[1.0,1,1],[-1,-1,-1],[2,2,2],[-2.0,-2,-2]]);
ShearDom = PeriodicShearedDomain(Lx,Ly,Lz);
ShearDom.setg(g);
Ewald = EwaldSplitter(a,mu,xi,ShearDom,Npts);
SpatialkD = ckDSpatial(pts,ShearDom);

# Velocity of the blobs - parallelogram domain
uParallelogram = Ewald.calcBlobTotalVel(pts,forces,ShearDom,SpatialkD);
print('Velocities')
print(uParallelogram)

# Velocity - rectangular domain
ptsRect = np.concatenate((pts,[[0,Ly,0],[1,Ly,0],[0.5,1+Ly,0],[1.5,1+Ly,0]]));
forcesRect = np.concatenate((forces,[[-1,-1,-1],[1, 1, 1],[-2, -2, -2], [2, 2, 2]]));
RectDom = PeriodicShearedDomain(Lx,2*Ly,Lz);
Ewald = EwaldSplitter(a,mu,xi,RectDom,2*Npts);
SpatialkD = ckDSpatial(ptsRect,RectDom);
uRectangle = Ewald.calcBlobTotalVel(ptsRect,forcesRect,RectDom,SpatialkD);

print('Relative errors between rectangular domain and parallelepiped domain:')
for iPt in range(4):
    print(np.linalg.norm(uRectangle[iPt,:]-uParallelogram[iPt,:])/np.linalg.norm(uRectangle[iPt,:]));

