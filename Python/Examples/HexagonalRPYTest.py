from RPYVelocityEvaluator import EwaldSplitter
from SpatialDatabase import SpatialDatabase,ckDSpatial
from Domain import Domain, PeriodicShearedDomain
import numpy as np
import time
import uammd

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
par = uammd.PSEParameters(psi=Ewald._xi, viscosity=mu, hydrodynamicRadius=a, tolerance=1e-6, Lx=Lx,Ly=Ly,Lz=Lz)
pse = uammd.UAMMD(par,Npts);
#pse.Mdot assumes interleaved positions and forces, that is x1,y1,z1,x2,y2,z2,...
shearedpts = pts.copy();
shearedpts[:,0]-=g*shearedpts[:,1]; # THIS INCLUDES TRANSFER TO SHEARED SPACE!
positions = np.array(np.reshape(shearedpts,3*Npts), np.float64);
forcesR = np.array(np.reshape(forces,3*Npts), np.float64);
MF=np.zeros(3*Npts, np.float64);
pse.setShearStrain(g)
pse.computeHydrodynamicDisplacements(positions, forcesR, MF)
MF = np.reshape(MF,(Npts,3))
print('Velocities parallelogram')
print(uParallelogram)
print('Error from Raul')
print(MF-uParallelogram)

# Velocity - rectangular domain
ptsRect = np.concatenate((pts,[[0,Ly,0],[1,Ly,0],[0.5,1+Ly,0],[1.5,1+Ly,0]]));
forcesRect = np.concatenate((forces,[[-1,-1,-1],[1, 1, 1],[-2, -2, -2], [2, 2, 2]]));
RectDom = PeriodicShearedDomain(Lx,2*Ly,Lz);
Ewald = EwaldSplitter(a,mu,xi,RectDom,2*Npts);
SpatialkD = ckDSpatial(ptsRect,RectDom);
uRectangle = Ewald.calcBlobTotalVel(ptsRect,forcesRect,RectDom,SpatialkD);
# Raul version
par = uammd.PSEParameters(psi=Ewald._xi, viscosity=mu, hydrodynamicRadius=a, tolerance=1e-6, Lx=Lx,Ly=2*Ly,Lz=Lz);
pse = uammd.UAMMD(par,2*Npts);
#pse.Mdot assumes interleaved positions and forces, that is x1,y1,z1,x2,y2,z2,...
positions = np.array(np.reshape(ptsRect,6*Npts), np.float64);
forcesR = np.array(np.reshape(forcesRect,6*Npts), np.float64);
MFR=np.zeros(6*Npts, np.float64);
pse.setShearStrain(0.0)
pse.computeHydrodynamicDisplacements(positions, forcesR, MFR)
MFR = np.reshape(MFR,(2*Npts,3))
print('Velocities rectangle')
print(uRectangle)
print('Error from Raul')
print(MFR-uRectangle)

print('Relative errors between rectangular domain and parallelepiped domain:')
for iPt in range(4):
    print(np.linalg.norm(uRectangle[iPt,:]-uParallelogram[iPt,:])/np.linalg.norm(uRectangle[iPt,:]));

