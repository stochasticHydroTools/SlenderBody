import numba as nb
import numpy as np

"""
Functions for many fibers that use numba to speed up the calculations
"""
c=False;
   
@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.int64),cache=True)
def deAliasIntegralNumba(f, UpsampledchebPolys,DpInv,N):
    """
    Method to dealias the product of f (an N array) with each of the first
    N-1 Chebyshev polynomials
    Upsample to a 2N grid, perform multiplication and integration, then 
    downsample to an N point grid.
    Inputs: f = values of function f, Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid,
    UpsampMat, DpInv = psuedo-inverse of Cheb differentiation matrix on 2N point grid, N = number of coefficients 
    """
    # Upsample the multiplication of f with Chebyshev polys for anti-aliasing
    UpSampMulti = (f*UpsampledchebPolys).T; 
    # Integrals on the upsampled grid
    Integrals2N = np.dot(DpInv,UpSampMulti);
    return Integrals2N;

@nb.njit((nb.float64[:], nb.float64[:], nb.float64[:]),cache=True)
def cart2sph(x,y,z):
    """
    Method to convert point(s) (x,y,z) to 
    spherical coordinates (azimuth, elevation, r)
    """
    azimuth = np.arctan2(y,x);
    elevation = np.arctan2(z,np.sqrt(x**2 + y**2));
    r = np.sqrt(x**2 + y**2 + z**2);
    azimuth[(np.abs(np.abs(elevation)-np.pi/2) < 1e-12)] = 0;
    return azimuth, elevation, r;

@nb.njit((nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64),cache=True)
def calcKNumba(Xs,UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N):
    """
    Computes the matrix K(X). Inputs: X_s, N x 3 array of tangent vectors
    Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid,
    UpsampMat, DownSampMat = matrices for upsampling and downsampling from N to 2N grid.  
    DpInv = psuedo-inverse of Cheb differentiation matrix on 2N point grid, N = number of coefficients 
    Output: the kinematic matrix K for a single fiber 
    """
    XsUpsampled = np.dot(UpsampMat,Xs);
    theta, phi, r = cart2sph(XsUpsampled[:,0],XsUpsampled[:,1],XsUpsampled[:,2]);
    n1x = -np.sin(theta);
    n1y = np.cos(theta);
    #n1z = np.zeros(2*N);
    n2x = -np.cos(theta)*np.sin(phi);
    n2y = -np.sin(theta)*np.sin(phi);
    n2z = np.cos(phi);
    J = np.zeros((6*N,2*N-2));
    J[0::3,0:N-1]= deAliasIntegralNumba(n1x,UpsampledchebPolys,DpInv,N);
    J[1::3,0:N-1]= deAliasIntegralNumba(n1y,UpsampledchebPolys,DpInv,N);
    #J[2::3,0:N-1]= deAliasIntegralNumba(n1z,Lmat, UpsampMat,DpInv,N); #n1z is zero! Don't need this
    J[0::3,N-1:2*N-2]= deAliasIntegralNumba(n2x,UpsampledchebPolys,DpInv,N);
    J[1::3,N-1:2*N-2]= deAliasIntegralNumba(n2y,UpsampledchebPolys,DpInv,N);
    J[2::3,N-1:2*N-2]= deAliasIntegralNumba(n2z,UpsampledchebPolys,DpInv,N);
    K = np.dot(LeastSquaresDownsampler,J)
    return K, J;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:]),cache=True)
def calcKtNumba(J,WeightedUpsamplingMat):
    """
    Calculates the matrix K^* (L^2 adjoint of K) given K, weights
    on the 2N Chebyshev grid, and stacked upsampling matrix
    """
    return np.dot(J.T,WeightedUpsamplingMat);

@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:],nb.float64,nb.int64),cache=True)
def calcMNumba(Xs,c, mu, N):
    """
    Calculates the local drag matrix M. 
    Inputs: Xs = fiber tangent vectors as a 3N array, c = N vector of local drag coefficients, 
    mu = viscosity, N = number of points 
    """
    M = np.zeros((3*N,3*N));
    viscInv = 1/(8*np.pi*mu);
    for j in range(N):
        v = Xs[j*3:j*3+3];
        XsXs = np.outer(v,v);
        M[j*3:j*3+3,j*3:j*3+3]=viscInv*(c[j]*(np.identity(3)+XsXs)+  np.identity(3)-3*XsXs);
    return M;

@nb.njit(nb.float64[:](nb.int64,nb.int64,nb.float64[:],nb.float64[:,:],nb.boolean[:]),parallel=True,cache=c)
def EvalAllBendForces(N,Nfib,Xstacked,FEMatrix,isActive):
    forceDs=np.zeros(N*Nfib*3);
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            forceDs[iFib*3*N:(iFib+1)*3*N] = np.dot(FEMatrix,Xstacked[iFib*3*N:(iFib+1)*3*N]);
    return forceDs;    
        
@nb.njit(nb.float64[:](nb.float64[:],nb.float64[:],nb.float64[:],nb.float64,nb.int64,nb.int64,nb.boolean[:]),parallel=True,cache=c)
def calcLocalVelocities(Xs_nonLoc,forceDsAll,localcs,mu,N,Nfib,isActive):
    """
    Compute the local velocities, i.e. M^loc*forceDen for all fibers. 
    Inputs: Xs_nonLoc = list of tangent vectors as a 3*N*nFib 1d array, forceDsAll = 
    list of all force densities as a 3*N*nFib 1d array, localcs = local drag coefficient, 
    mu = fluid viscosity, N = number of Cheb points per fiber, N = number of fibs 
    """
    LocalOnly = np.zeros(len(Xs_nonLoc));
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            Xs = Xs_nonLoc[iFib*3*N:(iFib+1)*3*N];
            forceD = forceDsAll[iFib*3*N:(iFib+1)*3*N];
            # Compute self velocity from RPY (which will be subtracted), finite part integral
            # velocity, and purely local velocity (this is needed in the non-local corrections
            # for very close fibers).
            M = calcMNumba(Xs,localcs,mu,N);
            LocalOnly[iFib*3*N:(iFib+1)*3*N] = np.dot(M,forceD);
    return LocalOnly;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.float64[:,:],nb.boolean[:]),parallel=True,cache=c)
def getUniformPointsNumba(chebpts,Nfib,N,Nuni,UniformFromChebMatrix,isActive):
    """
    Obtain uniform points from a set of Chebyshev points. 
    Inputs: chebpts as a (tot#ofpts x 3) array
    Outputs: uniform points as a (nPtsUniform*Nfib x 3) array.
    """
    uniPoints = np.zeros((Nfib*Nuni,3));
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            uniPoints[iFib*Nuni:(iFib+1)*Nuni,:]=np.dot(UniformFromChebMatrix,chebpts[iFib*N:(iFib+1)*N,:]);
    return uniPoints;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.float64[:,:],nb.boolean[:]),parallel=True,cache=c) 
def useNumbatoUpsample(chebVals,Nfib,N,Nup,upsampMat,isActive):
    upsampledValues = np.zeros((Nfib*Nup,3));
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            upsampledValues[iFib*Nup:(iFib+1)*Nup]=np.dot(upsampMat,chebVals[iFib*N:(iFib+1)*N]);
    return upsampledValues; 

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.float64[:,:],nb.boolean[:]),parallel=True,cache=c) 
def useNumbatoDownsample(upsampledVals,Nfib,N,Nup,downsampMat,isActive):
    chebValues = np.zeros((Nfib*N,3));
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            chebValues[iFib*N:(iFib+1)*N]=np.dot(downsampMat,upsampledVals[iFib*Nup:(iFib+1)*Nup]);
    return chebValues; 


@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
         nb.float64[:],nb.int64),cache=True)
def FinitePartMatrix(ChebPoints,Xs,FPMatrix,DiffMat,s,N):
    """
    3N x 3N Matrix for the finite part integral M_FP*f, formed explicitly
    """
    ActualFPMat = np.zeros((3*N,3*N));
    BigDiff = np.zeros((3*N,3*N)); # 3N x 3N differentiation matrix
    for iPt in range(N):
        for jPt in range(N):
            for iD in range(3):
                BigDiff[3*iPt+iD,3*jPt+iD] = DiffMat[iPt,jPt];
    DfPart = np.zeros((3*N,3*N));
    Xss = np.dot(DiffMat,Xs);
    for iPt in range(N):
        for jPt in range(N):
            if (iPt==jPt):
                # Diagonal block
                ActualFPMat[3*iPt:3*iPt+3,3*iPt:3*iPt+3] += 0.5*(np.outer(Xs[iPt,:],Xss[iPt,:])+\
                    np.outer(Xss[iPt,:],Xs[iPt,:]))*FPMatrix[iPt,iPt];
                # Derivative part
                DfPart[3*iPt:3*iPt+3,3*iPt:3*iPt+3] = (np.identity(3)+np.outer(Xs[iPt,:],Xs[iPt,:]))*FPMatrix[iPt,iPt];
            else:
                rvec = ChebPoints[iPt,:]-ChebPoints[jPt,:];
                r = np.linalg.norm(rvec);
                oneoverr = 1.0/r;
                ds = s[jPt]-s[iPt];
                oneoverds = 1.0/ds;
                ActualFPMat[3*iPt:3*iPt+3,3*jPt:3*jPt+3] = (np.identity(3) + np.outer(rvec,rvec)*oneoverr*oneoverr)\
                                                        *oneoverr*np.abs(ds)*oneoverds*FPMatrix[iPt,jPt];
                ActualFPMat[3*iPt:3*iPt+3,3*iPt:3*iPt+3]-= (np.identity(3)+np.outer(Xs[iPt,:],Xs[iPt,:]))*oneoverds*FPMatrix[iPt,jPt]; 
                                    
    return ActualFPMat+np.dot(DfPart,BigDiff);

@nb.njit((nb.int64,nb.int64,nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64,nb.float64[:],\
          nb.float64, nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
          nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.int64,nb.boolean[:]),parallel=True,cache=c)
def linSolveAllFibersForGM(Nfib,N,b,XsVec,XsAll,impcodt, cs, mu, UpsampMat,UpsampledchebPolys, WeightedUpsamplingMat, LeastSquaresDownsampler,DpInv,D4BC,I,wIt,\
    XVecs,FPMatrix,DiffMat,snodes,doFP,isActive):
    """
    Linear solve on all fibers together to obtain alphas and lambdas. 
    Inputs: Nfib = number of fibers, N = number of points per fibers, ptsCheb = Chebyshev points (as an N*Nfib x 3 array), 
    b = RHS vector, XsVec = tangent vectors (as an N*Nfib x 3 array), XsAll = tangent vectors, row stacked 
    (because numba doesn't support reshape), dt = timestep, 
    impco = implicit coefficient for linear solves, cs = local drag coefficients for each s, mu = fluid viscosity, 
    Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid, 
    UpsampMat, DownSampMat = matrices for upsampling and downsampling from N to 2N grid. DpInv = psuedo-inverse of Cheb
    differentiation matrix, w = Chebyshev weights, D4BC = bending force calculation operator, I = N x 3 identity matrix, 
    wIt = 3 x N matrix  that integrates functions on the grid, XVecs = positions, FPMatrix = matrix for finite part integral, 
    DiffMat = Cheb differentiatio matrix, snodes = Chebyshev nodes on fiber, doFP = include the finite part integral implicitly
    (1 for yes, 0 for no)
    Outputs: the alphas and lambdas on all the fibers 
    """
    Allalphas = np.zeros(Nfib*(2*N+1));
    Alllambdas = np.zeros(Nfib*3*N);
    #print(nb.config.NUMBA_NUM_THREADS)
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            M = calcMNumba(XsAll[iFib*3*N:(iFib+1)*3*N],cs, mu, N);
            if (doFP):
                MFP = FinitePartMatrix(XVecs[iFib*N:(iFib+1)*N,:],XsVec[iFib*N:(iFib+1)*N,:],FPMatrix,DiffMat,snodes,N);
                M+=MFP;
            K, J = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N);
            b1 = b[iFib*3*N:(iFib+1)*3*N];
            b2 = b[3*N*Nfib+(iFib)*(2*N+1):3*N*Nfib+(iFib+1)*(2*N+1)];
            Kt = calcKtNumba(J,WeightedUpsamplingMat);
            B = np.concatenate((K-impcodt*np.dot(M,np.dot(D4BC,K)),\
                I-impcodt*np.dot(M,np.dot(D4BC,I))),axis=1);
            C = np.concatenate((Kt,wIt));
            #Minv = np.linalg.inv(M);
            RHS = b2+np.dot(C,np.linalg.solve(M,b1));
            S = np.dot(C,np.linalg.solve(M,B));
            #alphaU,res,rank,k = np.linalg.lstsq(S,RHS,-1);
            alphaU = np.linalg.solve(S,RHS);
            Balph = np.dot(B,alphaU);
            lambdas = np.linalg.solve(M,Balph-b1);
            Allalphas[iFib*(2*N+1):(iFib+1)*(2*N+1)]=alphaU;
            Alllambdas[iFib*3*N:(iFib+1)*3*N] = lambdas;
    return np.concatenate((Alllambdas,Allalphas));
  
@nb.njit((nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
        nb.float64[:],nb.boolean[:]),parallel=True,cache=c)
def calcKAlphas(Nfib,N,XsVec,UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler, DpInv,I,allalphas,isActive):
    """
    Compute K*alpha for given alpha on all fibers (input allalphas) 
    See linSolveAllFibersForGM docstring for parameters 
    """
    Kalph = np.zeros(Nfib*3*N);
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            K, _ = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N);
            thisAlpha = allalphas[iFib*(2*N+1):(iFib+1)*(2*N+1)];
            Kalph[iFib*3*N:(iFib+1)*3*N] = np.dot(K,thisAlpha[:2*N-2])+np.dot(I,thisAlpha[2*N-2:]);
    return Kalph;

@nb.njit((nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
        nb.float64[:,:],nb.float64[:],nb.float64[:],nb.boolean[:]),parallel=True,cache=c)
def calcKAlphasAndKstarLambda(Nfib,N,XsVec,UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,\
    WeightedUpsamplingMat, DpInv,I,wIt,allalphas,alllambdas,isActive):
    """
    Compute K*alpha and K^T *lambda for inputs allalphas and alllambdas 
    See linSolveAllFibersForGM docstring for parameters 
    """
    Kalph = np.zeros(Nfib*3*N);
    Kstlam = np.zeros(Nfib*(2*N+1))
    for iFib in nb.prange(Nfib):
        if (isActive[iFib]):
            K, J = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N);
            thisAlpha = allalphas[iFib*(2*N+1):(iFib+1)*(2*N+1)];
            Kalph[iFib*3*N:(iFib+1)*3*N] = np.dot(K,thisAlpha[:2*N-2])+np.dot(I,thisAlpha[2*N-2:]);
            Kt = calcKtNumba(J,WeightedUpsamplingMat);
            Kstlam[iFib*(2*N+1):(iFib+1)*(2*N+1)] =  np.dot(np.concatenate((Kt,wIt)),alllambdas[iFib*3*N:(iFib+1)*3*N]);
    return Kalph, Kstlam;
