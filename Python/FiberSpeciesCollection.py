import numpy as np
from SpatialDatabase import SpatialDatabase, ckDSpatial
import copy

verbose = -1;

# Documentation last updated: 03/12/2021

class FiberSpeciesCollection(object):

    """
    This is a class that operates on collections of fiber collections, where 
    each fiber collection (species) has a unique discretization.
    This class is basically a bunch of for loops that do things for each species, 
    the only exception being nonlocal hydrodynamic calculations. 
    """

    ## ====================================================
    ##              METHODS FOR INITIALIZATION
    ## ====================================================
    def __init__(self,nSpecies,fiberCollectionList,Dom,nonLocal):
        """
        Constructor 
        Inputs: nSpecies = integer number of species, fiberCollectionList = list of 
        fiberCollection objects that contains all the information about each species. 
        Dom = Domain object, nonLocal = what kind of nonlocal hydrodynamics to do, 
        (0 = local drag, 1 = nonlocal with special quadrature (not implemented yet for
        species collections), 2 = not implemented, 3 = upsampled nonlocal,
        4 = local + finite part only)
        """
        self._nSpecies = nSpecies;
        self._fiberCollectionsBySpecies = fiberCollectionList;
        self._RowStartBySpecies = np.zeros(nSpecies+1,dtype=np.int64);
        self._AlphaStartBySpecies = np.zeros(nSpecies+1,dtype=np.int64);
        self._DirectQuadStartBySpecies = np.zeros(nSpecies+1,dtype=np.int64);
        self._UniformPointStartBySpecies = np.zeros(nSpecies+1,dtype=np.int64);
        self._nUniformBySpecies = np.zeros(nSpecies,dtype=np.int64);
        self._nChebBySpecies = np.zeros(nSpecies,dtype=np.int64);
        self._LengthBySpecies = np.zeros(nSpecies);
        self._NFibersBySpecies = np.zeros(nSpecies,dtype=np.int64);
        self._nonLocal = nonLocal;
        if (nonLocal == 1):
            raise NotImplementedError('Special quadrature not supported for different fiber lengths, \
                set nonLocal = 3 to do upsampling instead')
        if (nonLocal == 2):
            raise NotImplementedError('nonLocal = 2 not implemented for FiberSpeciesCollection')
        for iSpecies in range(nSpecies):
            fibCol = fiberCollectionList[iSpecies];
            if (self._nonLocal > 0 and self._nonLocal < 4):
                fibCol._nonLocal = 4; # it will do the finite part only
                print('Multiple species - nonlocal calculations will be done in the FiberSpeciesCollection class')
                print('Will overwrite the nonlocal variable in each individual fiberCollection to do local drag + FP only')
            self._RowStartBySpecies[iSpecies+1]=self._RowStartBySpecies[iSpecies]+fibCol._totnum;
            self._AlphaStartBySpecies[iSpecies+1]=self._AlphaStartBySpecies[iSpecies]+len(fibCol._alphas);
            self._DirectQuadStartBySpecies[iSpecies+1]=self._DirectQuadStartBySpecies[iSpecies]+fibCol._Ndirect*fibCol._Nfib;
            self._UniformPointStartBySpecies[iSpecies+1]=self._UniformPointStartBySpecies[iSpecies]+fibCol._Nunifpf*fibCol._Nfib;
            self._nUniformBySpecies[iSpecies] = fibCol._Nunifpf;
            self._nChebBySpecies[iSpecies] = fibCol._Npf;
            self._NFibersBySpecies[iSpecies] = fibCol._Nfib;
            self._LengthBySpecies[iSpecies] = fibCol._Lf;
        if (np.amin(self._NFibersBySpecies) < 1):
            raise ValueError('You have a species with 0 or negative fibers in it!')
        self._UniformPointIndexToFiberIndex = np.zeros(self._UniformPointStartBySpecies[nSpecies],dtype=np.int64);
        fibstartIndex =0;
        for iSpecies in range(nSpecies):
            fibCol = fiberCollectionList[iSpecies];
            self._UniformPointIndexToFiberIndex[self._UniformPointStartBySpecies[iSpecies]:self._UniformPointStartBySpecies[iSpecies+1]] = \
               fibstartIndex+fibCol.FiberIndexUniformPoints();
            fibstartIndex+=fibCol._Nfib;
        self._Nfib = fibstartIndex;
        self.initPointForceVelocityArrays(Dom);
        self.updateLargeListsFromEachSpecies();
    
    def initPointForceVelocityArrays(self, Dom):
        """
        Method to initialize the memory for lists of points, tangent 
        vectors and lambdas for the fiber collection
        Input: Dom = Domain object 
        """
        # Establish large lists that will be used for the non-local computations
        self._totnum = self._RowStartBySpecies[self._nSpecies];                         
        self._ptsCheb=np.zeros((self._totnum,3)); 
        self._sCheb = np.zeros(self._totnum);
        self._wtsCheb=np.zeros(self._totnum);
        self._tanvecs=np.zeros((self._totnum,3));                     
        self._lambdas=np.zeros((self._totnum*3));   
        self._totalphas = self._AlphaStartBySpecies[self._nSpecies];                 
        # Initialize the spatial database objects
        self._SpatialCheb = ckDSpatial(self._ptsCheb,Dom);
        self._totnumDirect = self._DirectQuadStartBySpecies[self._nSpecies];
        self._allUpsampledWts = np.zeros((self._totnumDirect,1));
        for iSpecies in range(self._nSpecies):    
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            firstDir = self._DirectQuadStartBySpecies[iSpecies];
            lastDir = self._DirectQuadStartBySpecies[iSpecies+1];
            self._allUpsampledWts[firstDir:lastDir] = np.reshape(np.tile(fibCol._fiberDisc.getwDirect(),fibCol._Nfib),(fibCol._Ndirect*fibCol._Nfib,1));
            self._sCheb[first:last] = np.tile(fibCol._fiberDisc.gets(),fibCol._Nfib)
            self._wtsCheb[first:last] = np.tile(fibCol._fiberDisc.getw(),fibCol._Nfib)
        self._SpatialDirectQuad = ckDSpatial(np.zeros((self._totnumDirect,3)),Dom);
        self._totnumUniform = self._UniformPointStartBySpecies[self._nSpecies];
        self._SpatialUni = ckDSpatial(np.zeros((self._totnumUniform,3)),Dom);
    
    def updateLargeListsFromEachSpecies(self):
        """
        Copy the X and Xs arguments from self._fiberCollectionsBySpecies 
        (list of fiber objects) into large (tot#pts x 3) arrays 
        that are stored in memory
        """
        #print('Calling fill point arrays')
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            self._ptsCheb[first:last,:] = fibCol._ptsCheb;
            self._tanvecs[first:last,:] = fibCol._tanvecs;
            self._lambdas[3*first:3*last] = fibCol._lambdas;  
                
    ## ====================================================
    ##      PUBLIC METHODS NEEDED EXTERNALLY
    ## ====================================================   
    def formBlockDiagRHS(self, X_nonLoc,Xs_nonLoc,t,exForceDen,lamstar,Dom,RPYEval):
        """
        RHS for the block diagonal GMRES system. 
        Inputs: X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, t = system time, exForceDen = external force density (treated explicitly), 
        lamstar = lambdas for the nonlocal calculation, Dom = Domain object, RPYEval = RPY velocity
        evaluator for the nonlocal terms, FPimplicit = whether to do the finite part integral implicitly. 
        The block diagonal RHS is 
        [M^Local*(F*X^n+f^ext) + M^NonLocal * (F*X^(n+1/2,*)+lambda*+f^ext)+u_0; 0], and is constructed 
        in the individual fiberCollection classes.
        """
        AllSpeciesRHS = np.zeros(3*self._totnum+self._totalphas)
        AllForceDs = np.zeros(3*self._totnum);
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];  
            This_RHS, AllForceDs[3*first:3*last] = fibCol.formBlockDiagRHS(X_nonLoc[first:last,:],Xs_nonLoc[first:last],\
                t,exForceDen[3*first:3*last],lamstar[3*first:3*last], Dom, RPYEval,returnForce=True)
            AllSpeciesRHS[3*first:3*last] = This_RHS[:3*(last-first)];
        # Still need to account for the inter-fiber hydro here
        if (self._nonLocal == 3):
            AllSpeciesRHS[:3*self._totnum]+=self.interFiberVelocity(X_nonLoc,Xs_nonLoc,AllForceDs, Dom, RPYEval)
        return AllSpeciesRHS;
    
    def calcNewRHS(self,BlockDiagAnswer,X_nonLoc,Xs_nonLoc,dtimpco,lamstar, Dom, RPYEval,FPimplicit=0):
        """
        New RHS (after block diag solve). This is the residual form of the system that gets 
        passed to GMRES. 
        Inputs: BlockDiagAnswer = the answer for lambda and alpha from the block diagonal solver, 
        X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, dtimpco = delta t * implicit coefficient  (usually dt/2),
        lamstar = lambdas for the nonlocal calculation, Dom = Domain object, RPYEval = RPY velocity
        evaluator for the nonlocal terms, FPimplicit = whether to do the finite part integral implicitly.
        The RHS for the residual system is  
        [M^NonLocal*(F*(X^n+dt*impco*K*alpha-X^(n+1/2,*)+lambda - lambda^*)); 0], where by lambda and alpha
        we mean the results from the block diagonal solver 
        """
        AllSpeciesNewRHS = np.zeros(3*self._totnum+self._totalphas)
        AllForceDs = np.zeros(3*self._totnum);
        numLams = 3*self._totnum;
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            alphafirst = self._AlphaStartBySpecies[iSpecies];
            alphalast = self._AlphaStartBySpecies[iSpecies+1];
            this_lamalph = np.zeros(3*(last-first)+alphalast-alphafirst);
            this_lamalph[:3*(last-first)] = BlockDiagAnswer[3*first:3*last];
            this_lamalph[3*(last-first):] = BlockDiagAnswer[numLams+alphafirst:numLams+alphalast];
            This_RHS, AllForceDs[3*first:3*last] = fibCol.calcNewRHS(this_lamalph,X_nonLoc[first:last],Xs_nonLoc[first:last],\
                dtimpco,lamstar[3*first:3*last], Dom, RPYEval,returnForce=True);
            AllSpeciesNewRHS[3*first:3*last] = This_RHS[:3*(last-first)];
        # Still need to account for the inter-fiber hydro here
        if (self._nonLocal == 3):
            AllSpeciesNewRHS[:numLams]+=self.interFiberVelocity(X_nonLoc,Xs_nonLoc,AllForceDs, Dom, RPYEval)
        return AllSpeciesNewRHS;
    
    def Mobility(self,lamalph,impcodt,X_nonLoc,Xs_nonLoc,Dom,RPYEval):
        """
        Mobility calculation for GMRES
        Inputs: lamalph = the input lambdas and alphas, impcodt = delta t * implicit coefficient,
        X_nonLoc = Npts * 3 array of Chebyshev point locations, Xs_nonLoc = Npts * 3 array of 
        tangent vectors, Dom = Domain object, RPYEval = RPY velocity evaluator for the nonlocal terms, 
        The calculation is [-(M^Local+M^NonLocal)*(impco*dt*F*K*alpha +lambda); K*lambda ]
        """
        lamalph = np.reshape(lamalph,len(lamalph))
        numLams = 3*self._totnum;
        AllForceDs = np.zeros(3*self._totnum);
        Mf = np.zeros(numLams+self._totalphas);
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            alphafirst = self._AlphaStartBySpecies[iSpecies];
            alphalast = self._AlphaStartBySpecies[iSpecies+1];
            this_lamalph = np.zeros(3*(last-first)+alphalast-alphafirst);
            this_lamalph[:3*(last-first)] = lamalph[3*first:3*last];
            this_lamalph[3*(last-first):] = lamalph[numLams+alphafirst:numLams+alphalast];
            this_Mf, AllForceDs[3*first:3*last] = fibCol.Mobility(this_lamalph,impcodt,X_nonLoc[first:last,:],\
                Xs_nonLoc[first:last,:],Dom,RPYEval,returnForce=True);
            Mf[3*first:3*last]=this_Mf[:3*(last-first)];
            Mf[numLams+alphafirst:numLams+alphalast] = this_Mf[3*(last-first):];
        # Still need to account for the inter-fiber hydro here
        if (self._nonLocal == 3):
            Mf[:numLams]-=self.interFiberVelocity(X_nonLoc,Xs_nonLoc,AllForceDs, Dom, RPYEval) # mobility has - sign!
        return Mf;
       
    def BlockDiagPrecond(self,b,Xs_nonLoc,dt,implic_coeff,X_nonLoc,doFP=0):
        """
        Block diagonal preconditioner for GMRES. 
        b = RHS vector. Xs_nonLoc = tangent vectors as an Npts x 3 array. 
        dt = timestep, implic_coeff = implicit coefficient, X_nonLoc = fiber positions as 
        an Npts x 3 array, doFP = whether the finite part is treated implicitly and included
        in the matrix. The preconditioner matrix is 
        P = [-M^Local K-impco*dt*M^Local*F*K; K^* 0]
        """
        b1D = np.reshape(b,len(b));
        numLams = 3*self._totnum;
        lamalph = np.zeros(numLams+self._totalphas);
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            alphafirst = self._AlphaStartBySpecies[iSpecies];
            alphalast = self._AlphaStartBySpecies[iSpecies+1];
            this_b = np.zeros(3*(last-first)+alphalast-alphafirst);
            this_b[:3*(last-first)] = b1D[3*first:3*last];
            this_b[3*(last-first):] = b1D[numLams+alphafirst:numLams+alphalast];
            this_lamalph = fibCol.BlockDiagPrecond(this_b,Xs_nonLoc[first:last,:],dt,implic_coeff,X_nonLoc[first:last,:],doFP);
            lamalph[3*first:3*last]=this_lamalph[:3*(last-first)];
            lamalph[numLams+alphafirst:numLams+alphalast] = this_lamalph[3*(last-first):];
        return lamalph;
        
    def interFiberVelocity(self,X_nonLoc,Xs_nonLoc,forceDs, Dom, RPYEval):
        """
        Compute the nonlocal velocity due to the fibers.
        Inputs: Arguments X_nonLoc, Xs_nonLoc = fiber positions and tangent vectors as Npts x 3 arrays, 
        forceDs = force densities as an 3*npts 1D numpy array
        Dom = Domain object the computation happens on,  
        RPYEval = EwaldSplitter (RPYVelocityEvaluator) to compute velocities
        Outputs: nonlocal velocity as a tot#ofpts*3 one-dimensional array.
        """
        # If not doing nonlocal hydro, return nothing
        if (self._nonLocal==0):
            return np.zeros(self._totnum*3);
           
        forceDs = np.reshape(forceDs,(self._totnum,3));       
        
        # Finite part has already been calculated in individual fiberCollection objects
        # We are only going to support direct quad with fixed upsampling ratio 
        Xupsampled = self.getPointsForUpsampledQuad(X_nonLoc);
        self._SpatialDirectQuad.updateSpatialStructures(Xupsampled,Dom);
        fupsampled = self.getPointsForUpsampledQuad(forceDs);
        forcesUp = fupsampled*self._allUpsampledWts;

        if (verbose>=0):
            print('Upsampling time %f' %(time.time()-thist));
            thist=time.time();
        RPYVelocityUp = RPYEval.calcBlobTotalVel(Xupsampled,forcesUp,Dom,self._SpatialDirectQuad,\
            self._fiberCollectionsBySpecies[0]._nThreads);
        if (verbose>=0):
            print('Upsampled Ewald time %f' %(time.time()-thist));
            thist=time.time();
        # Subtract self terms species by species
        SelfTerms = np.zeros((self._totnumDirect,3))
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            firstDir = self._DirectQuadStartBySpecies[iSpecies];
            lastDir = self._DirectQuadStartBySpecies[iSpecies+1];
            SelfTerms[firstDir:lastDir,:]=fibCol.GPUSubtractSelfTerms(fibCol._Ndirect,Xupsampled[firstDir:lastDir,:],forcesUp[firstDir:lastDir,:]);
        if (verbose>=0):
            print('Self time %f' %(time.time()-thist));
            thist=time.time();
        RPYVelocityUp -= SelfTerms;
        RPYVelocity = self.getValuesfromDirectQuad(RPYVelocityUp);
        if (verbose>=0):
            print('Downsampling time %f' %(time.time()-thist));
        if (np.any(np.isnan(RPYVelocity))):
            raise ValueError('Velocity is nan - stability issue!') 
        return np.reshape(RPYVelocity,self._totnum*3);
         
    def updateLambdaAlpha(self,lamalph,Xsarg):
        """
        Update the lambda and alphas after the solve is complete. 
        Inputs: lamalph = answer for lambda and alpha. Xsarg = tangent vectors for nonlocal
        """
        # Update master lambda
        numLams = 3*self._totnum;
        self._lambdas = lamalph[:numLams];
        # Set lambda and alpha in each fiber collection
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            alphafirst = self._AlphaStartBySpecies[iSpecies];
            alphalast = self._AlphaStartBySpecies[iSpecies+1];
            this_lamalph = np.zeros(3*(last-first)+alphalast-alphafirst);
            this_lamalph[:3*(last-first)] = lamalph[3*first:3*last];
            this_lamalph[3*(last-first):] = lamalph[numLams+alphafirst:numLams+alphalast];
            fibCol.updateLambdaAlpha(this_lamalph,Xsarg[first:last,:]);
           
    def updateAllFibers(self,dt,XsforNL,exactinex=1):
        """
        Update the fiber configurations, assuming self._alphas has been computed above. 
        Inputs: dt = timestep, XsforNL = the tangent vectors we use to compute the 
        Rodriguez rotation, exactinex = whether to preserve exact inextensibility
        """
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            fibCol.updateAllFibers(dt,XsforNL[first:last,:],exactinex);
        # Copy back to me
        self.updateLargeListsFromEachSpecies();
        return np.amax(self._ptsCheb);
    
    def initializeLambdaForNewFibers(self,newfibs,exForceDen,t,dt,implic_coeff,other=None):
        """
        Solve local problem to initialize values of lambda on the fibers.
        Inputs: newfibs = nSpecies list, where each element of the list has the new fibers 
        for that species, exForceDen = external (gravity/CL) force density on those fibers, 
        t = system time, dt = time step, implic_coeff = implicit coefficient (usually 1/2), 
        other = the other (previous time step) fiber collection
        """
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            otherFibCol = None;
            if (other is not None):
                otherFibCol = other._fiberCollectionsBySpecies[iSpecies];
            fibCol.initializeLambdaForNewFibers(newfibs[iSpecies],exForceDen[3*first:3*last],t,dt,implic_coeff,otherFibCol);
            self._lambdas[3*first:3*last] = fibCol._lambdas;
            if other is not None:
                other._lambdas[3*first:3*last] = otherFibCol._lambdas;   
                
    def FiberBirthAndDeath(self,tstep,other=None):
        """
        Turnover filaments. tstep = time step,
        other = the previous time step fiber collection (for Crank-Nicolson)
        """
        if (tstep==0):
            return [];
        AllBornFibs = [];
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            otherFibCol = None;
            if (other is not None):
                otherFibCol = other._fiberCollectionsBySpecies[iSpecies];
            AllBornFibs.append(fibCol.FiberBirthAndDeath(tstep,otherFibCol));
        # Replace my X and Xs
        self.updateLargeListsFromEachSpecies();
        if (other is not None):
            other.updateLargeListsFromEachSpecies(); 
        return AllBornFibs;
    
    def writeFiberLocations(self,of):
        """
        Write the locations of all fibers to a file
        object named of.
        """
        for iSpecies in range(self._nSpecies):
            self._fiberCollectionsBySpecies[iSpecies].writeFiberLocations(of);
    
    def writeFiberTangentVectors(self,of):
        """
        Write the locations of all fibers to a file
        object named of.
        """
        for iSpecies in range(self._nSpecies):
            self._fiberCollectionsBySpecies[iSpecies].writeFiberTangentVectors(of);
                
    def getg(self,t):
        return self._fiberCollectionsBySpecies[0].getg(t);
    
    def getX(self):
        return self._ptsCheb;

    def getXs(self):
        return self._tanvecs;
    
    def getUniformSpatialData(self):
        return self._SpatialUni;
    
    def nChebByFib(self):
        return np.repeat(self._nChebBySpecies,self._NFibersBySpecies)
    
    def nUniByFib(self):
        return np.repeat(self._nUniformBySpecies,self._NFibersBySpecies)
    
    def calcCurvatures(self,X):
        """
        Evaluate fiber curvatures on fibers with locations X. 
        Returns an Nfib array of the mean L^2 curvature by fiber 
        """
        Curvatures=np.zeros(self._Nfib);
        fibstartIndex =0;
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            Curvatures[fibstartIndex:fibstartIndex+fibCol._Nfib] = fibCol.calcCurvatures(X[first:last,:])
            fibstartIndex+=fibCol._Nfib;
        return Curvatures;
    
    def getUniformPoints(self,chebpts):
        UniformPts = np.zeros((self._UniformPointStartBySpecies[self._nSpecies],3));
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1]; 
            firstUni = self._UniformPointStartBySpecies[iSpecies];
            lastUni = self._UniformPointStartBySpecies[iSpecies+1];
            UniformPts[firstUni:lastUni,:] = fibCol.getUniformPoints(chebpts[first:last,:]);
        return UniformPts
    
    def FiberStress(self,XforNL,Lambdas,Dom):
        TotLam = 0.0;
        TotEl = 0.0;
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];     
            lamstr,elstr=fibCol.FiberStress(XforNL[first:last,:],Lambdas[3*first:3*last],Dom)
            TotLam+=lamstr;
            TotEl+=elstr;
        return TotLam,TotEl
    
    def getLambdas(self):
        return self._lambdas; 
    
    def uniformForce(self,strengths):
        """
        A uniform force density on all fibers with strength strength 
        """
        return np.tile(strengths,self._totnum);
    
    def getSysDimension(self):
        """
        Dimension of (lambda,alpha) sysyem for GMRES
        """
        return 3*self._totnum+self._totalphas;
      
    def getPointsForUpsampledQuad(self,chebpts):
        AllUpsampledPts = np.zeros((self._totnumDirect,3));
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            firstDir = self._DirectQuadStartBySpecies[iSpecies];
            lastDir = self._DirectQuadStartBySpecies[iSpecies+1];
            AllUpsampledPts[firstDir:lastDir,:]=fibCol.getPointsForUpsampledQuad(chebpts[first:last,:]);
        return AllUpsampledPts;
    
    def getValuesfromDirectQuad(self,upsampledVals):
        """
        Downsample the upsampled (velocities) to get the values on the original cheb pts
        """
        AllDownsampledPts = np.zeros((self._totnum,3));
        for iSpecies in range(self._nSpecies):
            fibCol = self._fiberCollectionsBySpecies[iSpecies];
            first = self._RowStartBySpecies[iSpecies];
            last = self._RowStartBySpecies[iSpecies+1];
            firstDir = self._DirectQuadStartBySpecies[iSpecies];
            lastDir = self._DirectQuadStartBySpecies[iSpecies+1];
            AllDownsampledPts[first:last,:]=fibCol.getValuesfromDirectQuad(upsampledVals[firstDir:lastDir,:]);
        return AllDownsampledPts;   
