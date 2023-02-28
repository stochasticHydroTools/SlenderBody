import numpy as np
from CrossLinkForceEvaluator import CrossLinkForceEvaluator as CForceEvalCpp
from warnings import warn 
from SpatialDatabase import CellLinkedList
from StericForceEvaluatorC import StericForceEvaluatorC
import time
verbose = 1;

# Documentation last updated: 02/25/2023

class StericForceEvaluator(object):

    """
    The purpose of this class is to evaluate steric forces between fibers. 
    The base class upsamples to uniform points and finds neighbors, then applies a 
    spherical repulsive potential
    
    TODO: Try other function for repulsion (tanh), double integral w exact integration
    """
    
    def __init__(self,nFib,Nx,Nunisites,fibDisc,X,Dom,FibSize,nThreads=1):
        """
        
        """
        self._Npf = Nx; 
        self._NUni = Nunisites;
        self._nFib = nFib;
        # Resampling matrix for uniform pts
        self._Rupsamp = fibDisc.UniformUpsamplingMatrix(Nunisites);
        self._FibSize = FibSize;
        self._DLens = Dom.getPeriodicLens();
        for iD in range(len(self._DLens)):
            if (self._DLens[iD] is None):
                self._DLens[iD] = 1e99;
        self._CppEvaluator = StericForceEvaluatorC(nFib, Nx, Nunisites,self._Rupsamp, self._FibSize,self._DLens,nThreads)
        Xuniform = self._CppEvaluator.getUniformPoints(X);
        self._NeighborSearcher = CellLinkedList(Xuniform,Dom,nThreads);
        self._TwoFibNeighborSearcher = CellLinkedList(Xuniform[:2*self._NUni,:],Dom,nThreads);
        
    def StericForces(self,X,Dom):
        """
        The force here is 
        F(r) = F_0 when r < d and F_0*exp((d-r)/delta) when r > d. 
        d is always the size of the fiber
        There are then 2 free parameters: F_0 and delta. 
        We choose F_0*delta = 4*kBT, so this leaves the choice of delta, the 
        radius at which potential decays. We take delta = diameter, so that the 
        potential decays to exp(-4) \approx 0.02 over 4 diameters. So we truncate 
        at d-r = 4*delta. 
        """
        delta = self._FibSize;
        dcrit = self._FibSize;
        F0 = 4*4.1e-3/delta;
        nStds = 4;
        CutoffDistance = nStds*delta+self._FibSize;
        ClosePts, UniformPts = self.CheckContacts(X,Dom,Cutoff=CutoffDistance);
        StericForce = self._CppEvaluator.getStericForces(ClosePts,UniformPts,Dom.getg(), dcrit, delta,F0);
        return StericForce;
        
        
    def CheckContacts(self,X,Dom,Cutoff=None):
        if (verbose > 0):
            thist = time.time();
        if (Cutoff is None):
            Cutoff = self._FibSize;
        # Eval uniform points
        Xuniform = self._CppEvaluator.getUniformPoints(X);
        self._NeighborSearcher.updateSpatialStructures(Xuniform,Dom);
        try:
            uniNeighbs = self._NeighborSearcher.selfNeighborList(Cutoff,self._NUni);
        except: # If N sites per f is not defined, it will just do 1 site per fiber and then the code below will process
            uniNeighbs = self._NeighborSearcher.selfNeighborList(Cutoff,1);
        uniNeighbs = uniNeighbs.astype(np.int64);
        # Filter the list of neighbors to exclude those on the same fiber
        Fibs = self.mapUniPtToFiber(uniNeighbs);
        delInds = np.arange(len(Fibs[:,0]));
        ClosePts = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
        return ClosePts, Xuniform;
        if (verbose > 0):
            print('Neighbor search and organize time %f ' %(time.time()-thist))  
            thist = time.time();
    
    def CheckIntersectWithPrev(self,fibList,iFib,Dom,nDiameters):
        # Intersection of two fibers
        Xi, _, _= fibList[iFib].getPositions()
        Xi_uni = np.dot(self._Rupsamp,Xi);
        for jFib in range(iFib):
            Xj, _, _= fibList[jFib].getPositions()
            Xj_uni = np.dot(self._Rupsamp,Xj);
            Xconcat = np.concatenate((Xi_uni,Xj_uni));
            self._TwoFibNeighborSearcher.updateSpatialStructures(Xconcat,Dom);
            Cutoff = nDiameters*self._FibSize;
            uniNeighbs = self._TwoFibNeighborSearcher.selfNeighborList(Cutoff,self._NUni);
            uniNeighbs = uniNeighbs.astype(np.int64);
            # Filter the list of neighbors to exclude those on the same fiber
            Fibs = self.mapUniPtToFiber(uniNeighbs);
            delInds = np.arange(len(Fibs[:,0]));
            ClosePts = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
            for iPair in ClosePts:
                iPt = iPair[0];
                jPt = iPair[1];
                rvec = Dom.calcShifted(Xconcat[iPt,:]-Xconcat[jPt,:]);
                if (np.linalg.norm(rvec) < Cutoff):
                    return True;
        print('Fiber %d accepted' %iFib)
        return False;    
    
    def mapUniPtToFiber(self,sites):
        """
        The fiber(s) associated with a set of uniform site(s)
        """
        return sites//self._NUni;
        
    
    
