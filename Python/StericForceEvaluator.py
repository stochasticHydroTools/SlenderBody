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
    
    There is a child class below (SegmentBasedStericForceEvaluator) which uses 
    segments to obtain the force. 
    
    """
    
    def __init__(self,nFib,Nx,Nunisites,fibDisc,X,Dom,radius,kbT,nThreads=1):
        """
        Constructor. Inputs: number of fibers (nFib), number of Chebyshev collocation
        points per fiber (Nx), number of uniform sites for resampling, a copy of the 
        fiber discretization object (to form the uniform resampling matrix), and positions
        X and Domain (to initialize). radius = the radius of the fibers (for the steric 
        repulsion)
        """
        self._Npf = Nx; 
        self._NUni = Nunisites;
        self._nFib = nFib;
        # Resampling matrix for uniform pts
        self._su, self._Rupsamp = fibDisc.UniformUpsamplingMatrix(Nunisites);
        self._radius = radius;
        self._DLens = Dom.getPeriodicLens();
        for iD in range(len(self._DLens)):
            if (self._DLens[iD] is None):
                self._DLens[iD] = 1e99;
        self._CppEvaluator = StericForceEvaluatorC(nFib, Nx, Nunisites,self._su,self._Rupsamp, self._DLens,nThreads)
        Xuniform = self._CppEvaluator.getUniformPoints(X,'u');
        self._NeighborSearcher = CellLinkedList(Xuniform,Dom,nThreads);
        self._TwoFibNeighborSearcher = CellLinkedList(Xuniform[:2*self._NUni,:],Dom,nThreads);
        self.SetForceParameters(kbT);
    
    def SetForceParameters(self,kbT):
        """
        The force (per unit area) here is 
        F(r) = F_0 when r < d and F_0*exp((d-r)/delta) when r > d. 
        d is always the size of the fiber
        There are then 2 free parameters: F_0 and delta. 
        We choose F_0*delta*a^2 = 4*kBT, so this leaves the choice of delta, the 
        radius at which potential decays. 
        """
        self._delta = self._radius/2;
        self._dcrit = 2*self._radius;
        nStds = 4;
        self._CutoffDistance = nStds*self._delta+2*self._radius; 
        F0 = 4*kbT/(self._delta*self._radius**2);
        self._CppEvaluator.SetForceFcnParameters(self._dcrit,self._delta,F0,self._CutoffDistance)
        
        
    def StericForces(self,X,Dom):
        """
        Compute steric forces. In this base class, this is two simple steps:
        first, upsample the fibers to uniform points (typical spacing 1/eps)
        and get a list of neighbors. Then, pass the list of neighbors to 
        C++ to compute forces. 
        """
        tiii = time.time();
        ClosePts, UniformPts = self.CheckContacts(X,Dom,Cutoff=self._CutoffDistance);
        print('Time for neighbor search %f ' %(time.time()-tiii))
        tiii = time.time();
        StericForce = self._CppEvaluator.ForcesFromGlobalUniformBlobs(ClosePts,UniformPts,Dom.getg());
        print('Time for force calc %f ' %(time.time()-tiii))
        return StericForce;
        
        
    def CheckContacts(self,X,Dom,Cutoff=None,excludeSelf=False):
        """
        Use upsampling to uniform points to check contacts between fibers. 
        The arguments are the points X, the domain Dom, and the cutoff distance
        (optional, if not passed code will use the size of the fibers), and 
        whether to exclude the self term from the interactions
        """
        if (verbose > 0):
            thist = time.time();
        if (Cutoff is None):
            Cutoff = self._FibSize;
        # Eval uniform points
        Xuniform = self._CppEvaluator.getUniformPoints(X,'u');
        self._NeighborSearcher.updateSpatialStructures(Xuniform,Dom);
        uniNeighbs = self._NeighborSearcher.selfNeighborList(Cutoff,1);
        uniNeighbs = uniNeighbs.astype(np.int64);
        if (excludeSelf):
            Fibs = self.mapUniPtToFiber(uniNeighbs);
            delInds = np.arange(len(Fibs[:,0]));
            ClosePts = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
            return ClosePts, Xuniform;
        return uniNeighbs, Xuniform
        if (verbose > 0):
            print('Neighbor search and organize time %f ' %(time.time()-thist))  
            thist = time.time();
    
    def CheckIntersectWithPrev(self,fibList,iFib,Dom,nDiameters):
        """
        This is for RSA of fibers. The idea is that you pass an object fibList, where the last
        fiber (fiber number iFib) is a candidate fiber. This code then loops through the previous
        fibers and checks if those two fibers are interacting. This is obviously slow (python based)
        but it is only called once at the beginning of the simulation.
        """
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
        #print('Fiber %d accepted' %iFib)
        return False;    
    
    def mapUniPtToFiber(self,sites):
        """
        The fiber(s) associated with a set of uniform site(s)
        """
        return sites//self._NUni;

class SegmentBasedStericForceEvaluator(StericForceEvaluator):

    """
    This is a more efficient implementation where we sample the fiber at a series of 
    SEGMENTS, then try to find the closest points on those segments. 
    
    We still leave the functionality to sample at 1/eps uniform points (the variable self._NUni)
    because it could be useful for checking. Here we add addditional functionality to pass
    in a number of SEGMENTS (where we need to sample at the endpoints and midpoints of those
    segments). 
    
    """
    
    def __init__(self,nFib,Nx,Nunisites,fibDisc,X,Dom,radius,kbT,Nsegments,nThreads=1):
        super().__init__(nFib,Nx,Nunisites,fibDisc,X,Dom,radius,kbT,nThreads);
        # Initialize parts of the code pertaining to SEGMENTS
        self._Nseg = Nsegments;
        self._sMP, self._RSegMP = fibDisc.UniformUpsamplingMatrix(Nsegments,'s');
        self._sEP, self._RSegEP = fibDisc.UniformUpsamplingMatrix(Nsegments+1,'u');
        self._Lseg=self._sEP[1]-self._sEP[0];
        self._CutoffSegmentSearch = self._CutoffDistance + self._Lseg;
        self._CppEvaluator.initSegmentVariables(fibDisc._L,Nsegments,self._CutoffSegmentSearch,self._RSegMP,self._RSegEP)
        SegMidpoints = self._CppEvaluator.getUniformPoints(X,'m');
        self._MidpointNeighborSearcher = CellLinkedList(SegMidpoints,Dom,nThreads);
        # Initialize code pertaining to quadrature on the segments (number of points and ds)
        # Donev: Don't hard-wire 8 here: 
        NumberQuadPoints = 8; # must be even to avoid double counting endpoints
        dsInnerQuad = self._radius;
        self._CppEvaluator.initInterpolationVariables(fibDisc._sX, fibDisc.StackedValsToCoeffsMatrix(),NumberQuadPoints,dsInnerQuad);
        
        
    def StericForces(self,X,Dom):
        # Step 1: sample at midpoints and perform neighbor search
        tiii = time.time();
        SegMidpoints = self._CppEvaluator.getUniformPoints(X,'m');
        self._MidpointNeighborSearcher.updateSpatialStructures(SegMidpoints,Dom);
        print(self._CutoffSegmentSearch)
        SegmentNeighbs = self._MidpointNeighborSearcher.selfNeighborList(self._CutoffSegmentSearch,1);
        SegmentNeighbs = SegmentNeighbs.astype(np.int64);
        print('Time for neighbor search %f ' %(time.time()-tiii))
        tiii = time.time();
        # Remove adjacent segments
        Fibs = self.mapSegToFiber(SegmentNeighbs);
        delInds = np.arange(len(Fibs[:,0]));
        DeleteThis = np.logical_and(Fibs[:,0]==Fibs[:,1], abs(SegmentNeighbs[:,1]-SegmentNeighbs[:,0]) <=1)
        SegmentNeighbs = np.delete(SegmentNeighbs,delInds[DeleteThis],axis=0);
        print('Time for post-process %f ' %(time.time()-tiii))
        print(SegmentNeighbs.shape)
        tiii = time.time();
        # Pass the list of segments to C++ for processiing (see the method there)
        SegEndpoints = self._CppEvaluator.getUniformPoints(X,'e')
        StericForce = self._CppEvaluator.ForceFromCloseSegments(SegmentNeighbs,X,SegMidpoints,SegEndpoints,Dom.getg())
        print('Time for force calc %f ' %(time.time()-tiii))
        return StericForce
        
    def mapSegToFiber(self,segs):
        """
        The fiber(s) associated with a set of uniform site(s)
        """
        return segs//self._Nseg;
          
    
    
