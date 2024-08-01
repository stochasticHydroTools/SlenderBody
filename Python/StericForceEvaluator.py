import numpy as np
from CrossLinkForceEvaluator import CrossLinkForceEvaluator as CForceEvalCpp
from warnings import warn 
from SpatialDatabase import CellLinkedList, ckDSpatial
from StericForceEvaluatorC import StericForceEvaluatorC
import time
verbose = -1;

class StericForceEvaluator(object):

    """
    The purpose of this class is to evaluate steric forces between fibers. 
    
    In this base class, we upsample the fiber to uniform points, then apply a 
    spherical potential between the uniform points. We then downsample that to the 
    Chebyshev points.
    
    More information about this class can be found in Section 9.2 of `Maxian's
    PhD thesis <https://www.proquest.com/docview/2832979813>`_. 
    
    """
    
    def __init__(self,nFib,Nx,Nunisites,fibDisc,X,Dom,radius,kbT,nThreads=1):
        """
        Constructor. 
        
        Parameters
        ----------
        nFib: int 
            Number of fibers 
        Nx: int 
            Number of Chebyshev collocation points per fiber 
        Nunisites: int 
            Number of uniform sites for resampling
        fibDisc: FibCollocationDiscretization object
            A copy of the fiber discretization object (to form the uniform resampling matrix)
        X: array
            Collocation point positions
        Dom: Domain object
            The domain we are using (to establish the periodicity)
        radius: double
            The radius of the fibers (for the steric repulsion)
        kbT: double
            Thermal energy
        nThreads: int, optional
            The number of OpenMP threads
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
        self._NeighborSearcher = ckDSpatial(Xuniform,Dom,nThreads);
        self._TwoFibNeighborSearcher = ckDSpatial(Xuniform[:2*self._NUni,:],Dom,nThreads);
        self.SetForceParameters(kbT);
        self._DontEvalForce = False;
    
    def SetForceParameters(self,kbT):
        """
        Throughout this class, we use an energy density (between two 
        spheres) given by
        $$\\hat{\\mathcal{E}}(r) = \\frac{\\mathcal{E}_0}{a^2}\\textrm{erf}\\left(r/(\\delta \\sqrt{2})\\right),$$
        where $a$ is the fiber radius (set in the constructor). The force between blobs 
        is the derivative of energy 
        $$\\frac{d \\mathcal{E}}{d r} = \\frac{\\mathcal{E}_0}{a^2 \\delta} \\sqrt{\\frac{2}{\\pi}} \\exp{(-r^2/(2\\delta^2))}$$
        times the displacement vector between points. 
        
        The purpose of this function is to set the parameters of the force and 
        energy functions. We set $\\delta = a$ (although that can be changed here), 
        $\\mathcal{E}_0=4 k_B T$, and set the the cutoff for the Gaussian at four 
        standard deviations.
        
        Parameters
        ----------
        kbT: double
            Thermal energy
        """
        nRadiusPerStd = 1;
        self._delta = nRadiusPerStd*self._radius;
        nStdsToCutoff = 4;
        self._CutoffDistance = nStdsToCutoff*self._delta;
        F0 = 1000*kbT/(self._radius*self._delta)*np.sqrt(2/np.pi);
        self._CppEvaluator.SetForceFcnParameters(self._delta,F0,self._CutoffDistance)
        
        
    def StericForces(self,X,Dom):
        """
        Compute steric forces. In this base class, this are two simple steps:
        first, upsample the fibers to uniform points (typical spacing 1/eps)
        and get a list of neighbors. Then, pass the list of neighbors to 
        C++ to compute forces. 
        
        This function will IGNORE the self steric force. 
        
        Parameters
        ----------
        X: array
            The Chebyshev collocation points
        Dom: Domain object
            The Domain object (to find neighbors)
        
        Returns
        -------
        array
            The steric forces along all fibers. 
        """
        if (verbose > 0):
            tiii = time.time();
        if (self._DontEvalForce):
            return 0
        ClosePts, UniformPts, _ = self.CheckContacts(X,Dom,Cutoff=self._CutoffDistance);
        if (verbose > 0):
            print('Time for neighbor search %f ' %(time.time()-tiii))
            tiii = time.time();
        StericForce = self._CppEvaluator.ForcesFromGlobalUniformBlobs(ClosePts,UniformPts,Dom.getg());
        if (verbose > 0):
            print('Time for force calc %f ' %(time.time()-tiii))
        return StericForce;
        
        
    def CheckContacts(self,X,Dom,Cutoff=None,excludeSelf=False):
        """
        Use upsampling to uniform points to check contacts between fibers.
        
        Parameters
        ----------
        X: array
            The Chebyshev collocation points
        Dom: Domain object
            The Domain object (to find neighbors)
        Cutoff: double, optional
            The cutoff distance at which we say fibers are in contact. If not 
            passed, the code will use the size of the fibers (sometimes we might want
            a buffer zone and pass a larger cutoff). 
        excludeSelf: bool, defaults to False
            Whether to exclude interactions between a fiber and itself.
        
        Returns
        -------
        (array, array)
            Two arrays are returned. The first is a 2 column array that gives
            the indices of neighboring uniform points (points within the cutoff distance).
            The second is the uniform points. 
        """
        if (verbose > 0):
            thist = time.time();
        if (Cutoff is None):
            Cutoff = 2*self._radius;
        # Eval uniform points
        Xuniform = self._CppEvaluator.getUniformPoints(X,'u');
        self._NeighborSearcher.updateSpatialStructures(Xuniform,Dom);
        uniNeighbs = self._NeighborSearcher.selfNeighborList(Cutoff,1);
        uniNeighbs = uniNeighbs.astype(np.int64);
        if (excludeSelf):
            Fibs = self.mapUniPtToFiber(uniNeighbs);
            delInds = np.arange(len(Fibs[:,0]));
            ClosePts = np.delete(uniNeighbs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
            Fibs = np.delete(Fibs,delInds[Fibs[:,0]==Fibs[:,1]],axis=0);
            CloseFibs = np.unique(Fibs,axis=0);
            return ClosePts, Xuniform, CloseFibs;
        if (verbose > 0):
            print('Neighbor search and organize time %f ' %(time.time()-thist))  
            thist = time.time();
        return uniNeighbs, Xuniform, None;
    
    def CheckIntersectWithPrev(self,fibList,iFib,Dom,nDiameters):
        """
        This function will check if a proposed fiber intersects with other fibers that 
        are already in the domain. The purpose is to use this for random sequential addition (RSA). 
        The idea is that you pass an object fibList, where the last
        fiber (fiber number iFib) is a candidate fiber. This code then loops through the previous
        fibers and checks if those two fibers are interacting. This is obviously slow (python based)
        but it is only called once at the beginning of the simulation.
        
        Parameters
        ----------
        fibList: list of DiscretizedFiber objects
            List that contains the positions (Chebyshev collocation points) of all 
            fibers already in the system
        iFib: int
            Index of fiber we want to add. Assumed to be at the END of fibList
        Dom: Domain object
            The domain that we do the computation on.
        nDiameters: double
            The minimum number of diameters we want the fibers to be apart
        
        Returns
        --------
        bool
            True if the proposed fiber (iFib) is within nDiameters diameters 
            of the ones already in the system, false if not. 
        """
        # Intersection of two fibers
        Xi, _, _= fibList[iFib].getPositions()
        Xi_uni = np.dot(self._Rupsamp,Xi);
        for jFib in range(iFib):
            Xj, _, _= fibList[jFib].getPositions()
            Xj_uni = np.dot(self._Rupsamp,Xj);
            Xconcat = np.concatenate((Xi_uni,Xj_uni));
            self._TwoFibNeighborSearcher.updateSpatialStructures(Xconcat,Dom);
            Cutoff = nDiameters*2*self._radius;
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
        
        Parameters
        ----------
        sites: array
            Array of site indicies
        
        Returns
        -------
        array
            Fiber numbers associated with those sites.
        """
        return sites//self._NUni;

class SegmentBasedStericForceEvaluator(StericForceEvaluator):

    """
    This is a more efficient implementation where we sample the fiber at a series of 
    SEGMENTS, then try to find the closest points on those segments. The method "StericForces"
    gives a more precise listing of what we are doing, but the basic idea is to divide 
    a fiber into segments, perform neighbor search using the segments to get an initial guess
    for Newton's method, then use Newton's method to identify the closest points between 
    pairs of fibers. We then put a quadrature grid around those closest points to compute 
    the energy and force.
    
    We still leave the functionality to sample at 1/eps uniform points (the variable self._NUni)
    because it could be useful for checking. Here we add addditional functionality to pass
    in a number of SEGMENTS (where we need to sample at the endpoints and midpoints of those
    segments). 
    
    """
    
    def __init__(self,nFib,Nx,Nunisites,fibDisc,X,Dom,radius,kbT,Nsegments,NumQuadPointsPerStd=1,nThreads=1):
        """
        Constructor. 
        
        Parameters
        ----------
        nFib: int 
            Number of fibers 
        Nx: int 
            Number of Chebyshev collocation points per fiber 
        Nunisites: int 
            Number of uniform sites for resampling
        fibDisc: FibCollocationDiscretization object
            A copy of the fiber discretization object (to form the uniform resampling matrix)
        X: array
            Collocation point positions
        Dom: Domain object
            The domain we are using (to establish the periodicity)
        radius: double
            The radius of the fibers (for the steric repulsion)
        kbT: double
            Thermal energy
        Nsegments: int
            Number of segments we use on each fiber for initial neighbor search
        NumQuadPointsPerStd: double
            Number of quadrature points we use per standard deviation of the Gaussian that
            describes the steric force.
        nThreads: int, optional
            The number of OpenMP threads
        """
        super().__init__(nFib,Nx,Nunisites,fibDisc,X,Dom,radius,kbT,nThreads);
        # Initialize parts of the code pertaining to SEGMENTS
        self._Nseg = Nsegments;
        self._sMP, self._RSegMP = fibDisc.UniformUpsamplingMatrix(Nsegments,'s');
        self._sEP, self._RSegEP = fibDisc.UniformUpsamplingMatrix(Nsegments+1,'u');
        self._Lseg=self._sEP[1]-self._sEP[0];
        if (self._Lseg <= self._CutoffDistance):
            raise ValueError('Segments have to be longer than cutoff')
        self._CutoffSegmentSearch = self._CutoffDistance + self._Lseg;
        self._CppEvaluator.initSegmentVariables(fibDisc._L,Nsegments,self._CutoffSegmentSearch,self._RSegMP,self._RSegEP)
        SegMidpoints = self._CppEvaluator.getUniformPoints(X,'m');
        self._MidpointNeighborSearcher = CellLinkedList(SegMidpoints,Dom,nThreads);
        # Initialize code pertaining to quadrature on the segments (number of points and ds)
        self._CppEvaluator.initInterpolationVariables(fibDisc._sX, fibDisc.StackedValsToCoeffsMatrix(),fibDisc.DiffXMatrix());
        self.PretabulateGL(NumQuadPointsPerStd);
        
    def PretabulateGL(self,NumQuadPointsPerStd):
        """
        Pretabulate the Gauss-Legendre points and weights for integration. This 
        computes a maximum number of points if the whole segment were to be included
        in the quadrature, then computes all grid sizes up to that size. The result goes
        into a big array of points and weights which gets passed to C++. 
        
        Parameters
        ----------
        NumQuadPointsPerStd: double
            Number of quadrature points we use per standard deviation of the Gaussian that
            describes the steric force.
        """
        MaxPts = int((self._Lseg*self._Nseg)/self._delta*NumQuadPointsPerStd)+1;
        TotNum = int((1+MaxPts)*MaxPts/2);
        StartIndex=0;
        AllGLPts = np.zeros(TotNum);
        AllGLWts = np.zeros(TotNum);
        for iNum in range(MaxPts):
            x, w = np.polynomial.legendre.leggauss(iNum+1);
            AllGLPts[StartIndex:StartIndex+iNum+1]=x;
            AllGLWts[StartIndex:StartIndex+iNum+1]=w;
            StartIndex+=iNum+1;
        self._CppEvaluator.initLegQuadVariables(NumQuadPointsPerStd,AllGLPts,AllGLWts)
        
    def StericForces(self,X,Dom):
        """
        Compute steric forces in the segment-based algorithm. The steps are as follows (see 
        the link at the top of this class for more information):
        
        1) Sample the fiber to form straight segments of length $L_{seg}$.
           Perform a neighbor search over the segment midpoints using the
           cutoff distance $r_c+L_{seg}$ to obtain list of potentially 
           interacting segments
        2) For each pair of fiber pieces triggered by the neighbor search,
           approximate curved fiber pieces as straight segments and solve
           a quadratic equation to determine closest point of interaction
           between straight segments. 
        3) Use this distance to determine if the fiber pieces are close
           enough to interact. 
        4) Use the segment minimum as an initial guess for Newton's method
           to the true minimum distance between the fiber curves.
        5) Use a quadratic approximation around the point with minimum distance
           to find the region where the fibers are interacting.
        6) Merge intervals that overlap from the same pair of fibers.
        7) Put down a Chebyshev grid on this region and integrate to get forces.
        
        This function will NOT compute steric forces if the "closest points" 
        found in step 4 are actually the same point on the same fiber. By doing this,
        we ignore the self steric force.
        
        Parameters
        ----------
        X: array
            The Chebyshev collocation points
        Dom: Domain object
            The Domain object (to find neighbors)
        
        Returns
        -------
        array
            The steric forces along all fibers. 
        """
        if (self._DontEvalForce):
            return 0
        # Step 1: sample at midpoints and perform neighbor search
        if (verbose > 0):
            tiii = time.time();
        SegMidpoints = self._CppEvaluator.getUniformPoints(X,'m');
        self._MidpointNeighborSearcher.updateSpatialStructures(SegMidpoints,Dom);
        SegmentNeighbs = self._MidpointNeighborSearcher.selfNeighborList(self._CutoffSegmentSearch,1);
        SegmentNeighbs = SegmentNeighbs.astype(np.int64);
        if (verbose > 0):
            print('Time for midpoint neighbor search %f ' %(time.time()-tiii))
            tiii = time.time();
        # Remove adjacent segments
        Fibs = self.mapSegToFiber(SegmentNeighbs);
        delInds = np.arange(len(Fibs[:,0]));
        DeleteThis = np.logical_and(Fibs[:,0]==Fibs[:,1], abs(SegmentNeighbs[:,1]-SegmentNeighbs[:,0]) <=1)
        SegmentNeighbs = np.delete(SegmentNeighbs,delInds[DeleteThis],axis=0);
        if (verbose > 0):
            print('Time for post-process to remove adjacent segments %f ' %(time.time()-tiii))
            tiii = time.time();
        # Step 2: Use Newton's method to identify regions on the fibers that are interacting
        # This will return some repeats
        SegEndpoints = self._CppEvaluator.getUniformPoints(X,'e')
        InteractionInts = self._CppEvaluator.IntervalsOfInteraction(SegmentNeighbs,X,SegMidpoints,SegEndpoints,Dom.getg())
        if (verbose > 0):
            print('Time for to find intervals of interaction %f ' %(time.time()-tiii))
            tiii = time.time();
        # Step 3: Use Python machinery to remove repeats, then C++ to merge intervals for quadrature
        InteractionInts = InteractionInts[InteractionInts[:,0]!=-1,:] # C++ returns rows of -1 when the pair is not interacting
        # The unique row that gets chosen is different between Matlab and C++. For this reason the
        # difference between the two codes will be larger than machine precision
        TolForUniqueRow = self._CppEvaluator.getNewtonTol()*10;
        IntegerInteractionInts = InteractionInts/TolForUniqueRow;
        IntegerInteractionInts = IntegerInteractionInts.astype(int);
        _,inds=np.unique(IntegerInteractionInts,return_index=True,axis=0);
        nPairs = len(inds);
        InteractionInts=InteractionInts[inds,:];
        iFibjFib = np.concatenate(([InteractionInts[:,0]],[InteractionInts[:,3]]),axis=0).T
        iFibjFib, UniqueInds, numberOfPairs = np.unique(iFibjFib, return_index=True, return_counts=True,axis=0);
        ExactlyOnePairs = InteractionInts[UniqueInds[numberOfPairs==1],:];
        RepeatedPairs = InteractionInts[np.concatenate((UniqueInds[numberOfPairs>1],np.setdiff1d(np.arange(nPairs),UniqueInds))),:];
        # Sort the repeated pairs
        # Sort by (iFib, jFib, s1start) in that order
        RepeatedPairs = RepeatedPairs[RepeatedPairs[:,1].argsort()] # First sort doesn't need to be stable.
        RepeatedPairs = RepeatedPairs[RepeatedPairs[:,3].argsort(kind='mergesort')]
        RepeatedPairs = RepeatedPairs[RepeatedPairs[:,0].argsort(kind='mergesort')]
        MergedRepeats = self._CppEvaluator.MergeRepeatedIntervals(RepeatedPairs)
        AllIntervals = np.concatenate((ExactlyOnePairs,MergedRepeats),axis=0)
        if (verbose > 0):
            print('Time for python post-process %f ' %(time.time()-tiii))
            tiii = time.time();
        # Step 4: Calculate forces from intervals
        iFibjFib = AllIntervals[:,[0, 3]];
        ArclengthIntervals =  AllIntervals[:,[1,2,4,5]];
        Forces = self._CppEvaluator.ForceFromIntervals(iFibjFib, X, AllIntervals[:,6:9],ArclengthIntervals);
        if (verbose > 0):
            print('Time for to compute forces %f ' %(time.time()-tiii))
        return Forces
        
    def mapSegToFiber(self,segs):
        """
        The fiber(s) associated with a set of segment(s)
        
        Parameters
        ----------
        segs: array
            Array of segment indicies
        
        Returns
        -------
        array
            Fiber numbers associated with those segments.
        """
        return segs//self._Nseg;
          
    
    
