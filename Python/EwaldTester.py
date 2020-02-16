from EwaldSplitter import EwaldSplitter
from Domain import Domain
import numpy as np
import time

# This was just temporary for debugging
def writeArray(inar,outFile,wa):
    of = open(outFile,wa);
    r,c = inar.shape;
    for i in range(r):
        for j in range(c):
            of.write('%14.15f       ' % inar[i,j]);
        of.write('\n');
    of.close();

g = 0.125;
Lx = 2.4;
Ly = 3.0;
Lz = 3.6;
a=0.012;
mu = 1.5;
xi = 6.7;
Dom = Domain(Lx,Ly,Lz);
Dom.setg(g);

Ewald = EwaldSplitter(xi,a,mu);


# Testing stuff
Npts=8000;
#np.random.seed(0);
pts = (np.random.rand(Npts,3))*[-Lx, -Ly, -Lz];
forces = np.random.rand(Npts,3);
#writeArray(pts,'points.txt','w');
#writeArray(forces,'forces.txt','w');
velFar = Ewald._EwaldFarVel(Npts,pts,forces,Dom);
#writeArray(velFar,'velFar.txt','w');
ptTree = Dom.makekDTree(pts);
s=time.time();
velNear = Ewald._EwaldNearVelkD(Npts,pts,Dom,ptTree,forces);
print('Time to do C++ %f' %(time.time()-s));
# Call once to compile
velNearPy = Ewald._EwaldNearVelkDPython(Npts,pts,Dom,ptTree,forces);
# Second time to time it
e1=time.time();
velNearPy = Ewald._EwaldNearVelkDPython(Npts,pts,Dom,ptTree,forces);
print('Time to do python %f' %(time.time()-e1));
e2=time.time();
#velNearQ = Ewald._EwaldNearVelQuad(Npts,pts,forces,Dom);
e3=time.time();
print('Time to do quadratic %f' %(e3-e2));

#print(np.amax(np.abs(velNear-velNearQ)))
#print(np.amax(np.abs(velNearPy-velNearQ)))
print(np.amax(np.abs(velNearPy-velNear)))
