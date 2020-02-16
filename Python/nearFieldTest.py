from math import pi
from fiberCollection import fiberCollection
from FiberDiscretization import FiberDiscretization
from DiscretizedFiber import DiscretizedFiber
from Domain import Domain
import chebfcns as cf
import numpy as np

# This was just temporary for debugging
def writeArray(inar,outFile,wa):
    of = open(outFile,wa);
    r,c = inar.shape;
    for i in range(r):
        for j in range(c):
            of.write('%14.15f       ' % inar[i,j]);
        of.write('\n');
    of.close();


# 2 Fiber Ewald Test
N=32;
Lf=2;
Lx = 2.5; # periodic domain size
Ly = 3.7;
Lz = 2.2;
xi = 4; # Ewald parameter
g = 0.1; # shift in periodic domain
mu=1;    
epsilon=1e-3;
nFib=2;
tint=2;
omega = 1;
gam0 = 0.1;
s=cf.chebPts(N,[0,Lf],1);
D=cf.diffMat(1,[0,Lf],N,N,1,1);
X1 = np.concatenate(([np.cos(s)],[np.sin(s)],[s]),axis=0).T;
X1/=np.sqrt(2);
X1old = X1.copy();
Xs1 = np.dot(D,X1);
X1 = np.reshape(X1,3*N);
Xs1 = np.reshape(Xs1,3*N);
f1 = np.concatenate(([np.cos(2*pi*s/Lf)],[np.sin(2*pi*s/Lf)], [s-Lf/2.0]),axis=0).T;
sh=0.39+Ly;
X2 = np.concatenate(([np.zeros(N)+0.25],[np.zeros(N)+0.25],[s+0.1]),axis=0).T;
X2+= sh*np.array([g,1,0]);
Xs2 = np.dot(D,X2);
X2 = np.reshape(X2,3*N);
Xs2 = np.reshape(Xs2,3*N);
f2 = np.concatenate(([s-Lf/2.0],[np.sin(2*pi*s/Lf)],[np.cos(2*pi*s/Lf)]),axis=0).T;
fext = np.concatenate((f1,f2));
fibDisc = FiberDiscretization(Lf,epsilon,0,1,mu,N);
fibList = [None]*nFib;
fibList[0] = DiscretizedFiber(fibDisc,X1,Xs1);
fibList[1] = DiscretizedFiber(fibDisc,X2,Xs2);
Dom = Domain(Lx,Ly,Lz);
allFibers = fiberCollection(nFib,fibDisc,1,Dom,xi,mu,omega,gam0,Lf,epsilon);
allFibers.initFibList(fibList);
nLvelocity = allFibers.nonLocalBkgrndVelocity(tint,1,fext,pi/2);
print 'g value: %f' %allFibers.getg()
print 'g value: %f' %Dom.getg()
writeArray(np.reshape(nLvelocity,(2*N,3)),'nLvel.txt','w');

#st,_=cf.chebpts(1000,[0,Lf],1);
#targs = np.concatenate(([np.cos(st)],[np.sin(st)],[st]),axis=0).T;
#targs/=np.sqrt(2);
#targs+=0.01*np.random.rand(1000,3);
#cvels=0*targs;
#for iTarg in xrange(1000):
#    print 'Doint targ %d' %iTarg
#    cvels[iTarg,:] = flu.correctVel(targs[iTarg,:],\
#            fibList[0],X1old,f1*np.reshape(w,(16,1)),f1,0*X1old,2);
#writeArray(targs,'tagrs.txt','w');
#writeArray(cvels,'cvels.txt','w');
"""
nFib=50;
N=20;
epsilon=1e-3;
Lf=2;
Lx = 2.4; # periodic domain size
Ly = 1.5;
Lz = 3.7;
xi=5;
g = 0.5; # shift in periodic domain
omega=0;
gam0=0;
mu=1.5;

import time
np.random.seed(0);
Dom = Domain(Lx,Ly,Lz);
flu = fiberCollection(nFib,N,1,Dom,xi,mu,omega,gam0,Lf,epsilon)
Dom.setg(g);
Npts=nFib*N;
flu._ptsxyz = (np.random.rand(Npts,3))*[-Lx, -Ly, -Lz];
flu._ptsuni = (np.random.rand(16*nFib,3))*2*[Lx, Ly, Lz];
Dom.updatekDTreeCheb(flu._ptsxyz);
Dom.updatekDTreeUnif(flu._ptsuni);
s=time.time();
targs3, fibers3, methods3, shifts3 = flu.determineQuadkDLists(Dom);
print 'Time to do lists %f' %(time.time()-s);
targs2, fibers2, methods2, shifts2 = flu.determineQuad_quad(Dom);
print np.amax(np.abs(np.sort(targs2)-np.sort(targs3)))
print np.amax(np.abs(np.sort(fibers2)-np.sort(fibers3)));
print np.amax(np.abs(np.sort(methods2)-np.sort(methods3)));
print np.amax(np.abs(np.sort(shifts2)-np.sort(shifts3)));
"""


