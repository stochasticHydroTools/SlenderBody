import numba as nb
import numpy as np

"""
Functions for many fibers that use numba to speed up the calculations
Documentation last updated: 03/12/2021

Some of the functions here have a lot of parameters that are the same at
every time step. I have partially remedied this by moving these functions to 
C++, where I have a class where I can store these variables as members. For numba,
the issue is that parallelization is not yet implemented for classes, so I have to
write regular functions for the parallelization.
"""
c=True; # DO NOT CACHE WHEN CHANGING THE NUMBER OF NUMBA THREADS!

@nb.njit((nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],\
    nb.int64,nb.int64,nb.float64[:]),parallel=True,cache=c)
def FiberStressNumba(X, fbend,lams,Lens,Nfib,N,w):
    """
    Compute the [1,0] stress in the suspension due to the fiber via Batchelor's formula
    Inputs: lams = constraint forces on the fibers as a totnum*3 1D numpy array, 
    Lens = 3 array of periodic lengths (to compute volume for the stress)
    Output: the 3 x 3 stress tensor 
    """
    stressLams = 0;
    stressFB = 0; # from bending 
    for iPt in nb.prange(N*Nfib):
        wt = w[iPt%N];
        stressLams-= X[iPt,1]*lams[3*iPt]*wt;
        stressFB-= X[iPt,1]*fbend[3*iPt]*wt
    stressLams/=np.prod(Lens);
    stressFB/=np.prod(Lens);
    return stressLams, stressFB;
   
@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.int64),cache=True)
def deAliasIntegralNumba(f, UpsampledchebPolys,DpInv,N):
    """
    Method to dealias the product of f (an N array) with each of the first
    nP Chebyshev polynomials
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

@nb.njit((nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64,nb.int64),cache=True)
def calcKNumba(Xs,UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N,nPolys):
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
    J = np.zeros((6*N,2*nPolys));
    J[0::3,0:nPolys]= deAliasIntegralNumba(n1x,UpsampledchebPolys,DpInv,N);
    J[1::3,0:nPolys]= deAliasIntegralNumba(n1y,UpsampledchebPolys,DpInv,N);
    #J[2::3,0:nPolys]= deAliasIntegralNumba(n1z,Lmat, UpsampMat,DpInv,N); #n1z is zero! Don't need this
    J[0::3,nPolys:2*nPolys]= deAliasIntegralNumba(n2x,UpsampledchebPolys,DpInv,N);
    J[1::3,nPolys:2*nPolys]= deAliasIntegralNumba(n2y,UpsampledchebPolys,DpInv,N);
    J[2::3,nPolys:2*nPolys]= deAliasIntegralNumba(n2z,UpsampledchebPolys,DpInv,N);
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

@nb.njit(nb.float64[:](nb.int64,nb.int64,nb.float64[:],nb.float64[:,:]),parallel=True,cache=c)
def EvalAllBendForces(N,Nfib,Xstacked,FEMatrix):
    """
    Evaluate the bending forces on all fibers
    N = number of Cheb nodes per fiber, Nfib = number of fibers, Xstacked = locations of 
    points as a 1D stacked vector, FEMatrix = the matrix D_BC^4 that evaluates the bending forces
    """
    forceDs=np.zeros(N*Nfib*3);
    for iFib in nb.prange(Nfib):
        forceDs[iFib*3*N:(iFib+1)*3*N] = np.dot(FEMatrix,Xstacked[iFib*3*N:(iFib+1)*3*N]);
    return forceDs;    
        
@nb.njit(nb.float64[:](nb.float64[:],nb.float64[:],nb.float64[:],nb.float64,nb.int64,nb.int64),parallel=True,cache=c)
def calcLocalVelocities(Xs_nonLoc,forceDsAll,localcs,mu,N,Nfib):
    """
    Compute the local velocities, i.e. M^loc*forceDen for all fibers. 
    Inputs: Xs_nonLoc = list of tangent vectors as a 3*N*nFib 1d array, forceDsAll = 
    list of all force densities as a 3*N*nFib 1d array, localcs = local drag coefficient, 
    mu = fluid viscosity, N = number of Cheb points per fiber, Nfib = number of fibs 
    """
    LocalOnly = np.zeros(len(Xs_nonLoc));
    for iFib in nb.prange(Nfib):
        Xs = Xs_nonLoc[iFib*3*N:(iFib+1)*3*N];
        forceD = forceDsAll[iFib*3*N:(iFib+1)*3*N];
        M = calcMNumba(Xs,localcs,mu,N);
        LocalOnly[iFib*3*N:(iFib+1)*3*N] = np.dot(M,forceD);
    return LocalOnly;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.float64[:,:]),parallel=True,cache=c)
def getUniformPointsNumba(chebpts,Nfib,N,Nuni,UniformFromChebMatrix):
    """
    Obtain uniform points from a set of Chebyshev points. 
    Inputs: chebpts as a (tot#ofpts x 3) array, Nfib = number of fibers, N = number of 
    Cheb nodes, Nuni = number of uniform nodes, UniformFromChebMatrix = matrix that 
    transforms the Chebyshev locations to the uniform ones
    Outputs: uniform points as a (nPtsUniform*Nfib x 3) array.
    """
    uniPoints = np.zeros((Nfib*Nuni,3));
    for iFib in nb.prange(Nfib):
        uniPoints[iFib*Nuni:(iFib+1)*Nuni,:]=np.dot(UniformFromChebMatrix,chebpts[iFib*N:(iFib+1)*N,:]);
    return uniPoints;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:],nb.float64,nb.int64))
def rotate(vectors,axisHat,angle,Nvectors):
    return vectors*np.cos(angle)+np.cross(axisHat,vectors)*np.sin(angle)\
        +axisHat*np.reshape(np.dot(vectors,axisHat),(Nvectors,1))*(1.0-np.cos(angle));

@nb.njit((nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.int64,nb.int64,\
    nb.float64,nb.float64,nb.float64,nb.float64,nb.float64[:]),parallel=True,cache=c)
def BrownianVelocities(X,Xs,wRs,alphaBetaGamma,Nfib,Npf,mu,Lf,kbT,dt,Randoms):
    BVel = np.zeros(Npf*Nfib*3);
    alpha = alphaBetaGamma[0];
    beta = alphaBetaGamma[1];
    gamma = alphaBetaGamma[2];
    newX = X;
    newXs = Xs;
    for iFib in nb.prange(Nfib):
        Xfib = X[iFib*Npf:(iFib+1)*Npf,:];
        Xsfib = Xs[iFib*Npf:(iFib+1)*Npf,:];
        Xbar = np.reshape(1/Lf*np.dot(wRs.T,Xfib),3);
        Xsbar = np.reshape(1/Lf*np.dot(wRs.T,Xsfib),3);
        Nhalf = np.zeros((6,6))
        Nhalf[0:3,0:3]=1/np.sqrt(mu*Lf)*(np.sqrt(alpha)*np.identity(3)\
            +(-np.sqrt(alpha)+np.sqrt(alpha+beta))*np.outer(Xsbar,Xsbar));
        Nhalf[3:,3:]=np.sqrt(gamma/(mu*Lf**3))*(np.identity(3)-np.outer(Xsbar,Xsbar)); 
        RandomVec = Randoms[6*iFib:6*(iFib+1)];
        UOmega = np.sqrt(2*kbT/dt)*np.dot(Nhalf,RandomVec);
        URigid = np.zeros(3*Npf);
        dX = np.zeros((Npf,3));
        normOmega = np.linalg.norm(UOmega[3:])
        OmegaHat = UOmega[3:]/normOmega;
        newX[iFib*Npf:(iFib+1)*Npf,:] = Xbar+rotate(Xfib-Xbar,OmegaHat,normOmega*dt,Npf);
        newXs[iFib*Npf:(iFib+1)*Npf,:] = rotate(Xsfib,OmegaHat,normOmega*dt,Npf);
        for iPt in range(Npf):
            newX[iFib*Npf+iPt,:]+=UOmega[:3]*dt;
    return newX, newXs
  

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.float64[:,:]),parallel=True,cache=c) 
def useNumbatoUpsample(chebVals,Nfib,N,Nup,upsampMat):
    upsampledValues = np.zeros((Nfib*Nup,3));
    for iFib in nb.prange(Nfib):
        upsampledValues[iFib*Nup:(iFib+1)*Nup]=np.dot(upsampMat,chebVals[iFib*N:(iFib+1)*N]);
    return upsampledValues; 

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.int64,nb.int64,nb.float64[:,:]),parallel=True,cache=c) 
def useNumbatoDownsample(upsampledVals,Nfib,N,Nup,downsampMat):
    chebValues = np.zeros((Nfib*N,3));
    for iFib in nb.prange(Nfib):
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

@nb.njit((nb.int64,nb.int64,nb.int64,nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64,nb.float64[:],\
          nb.float64, nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
          nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.int64),parallel=True,cache=c)
def linSolveAllFibersForGM(Nfib,nPolys,N,b,XsVec,XsAll,impcodt, cs, mu, UpsampMat,UpsampledchebPolys,
    WeightedUpsamplingMat, LeastSquaresDownsampler,DpInv,D4BC,I,wIt,\
    XVecs,FPMatrix,DiffMat,snodes,doFP):
    """
    Linear solve on all fibers together to obtain alphas and lambdas. 
    Inputs: Nfib = number of fibers, N = number of points per fibers,
    b = RHS vector, XsVec = tangent vectors (as an N*Nfib x 3 array), XsVec = Npts x 3 vector of positions,
    XsAll = tangent vectors, row stacked (because numba doesn't support reshape), impcodt = implicit coefficient*timestep, 
    cs = local drag coefficients for each s, mu = fluid viscosity, UpsampMat = matrix for upsampling from N to 2N grid. 
    UpsampledChebPolys = values of Chebyshev polynomials on the 2N grid. WeightedUpsamplingMat = the upsampling matrix 
    with integration weights (used in computing K^*), LeastSquaresDownsampler = matrix we use for downsampling, 
    DpInv = psuedo-inverse of Cheb differentiation matrix, D4BC = bending force calculation operator, I = N x 3 identity matrix, 
    wIt = 3 x N matrix  that integrates functions on the grid, XVecs = positions, FPMatrix = matrix for finite part integral, 
    DiffMat = Cheb differentiation matrix, snodes = Chebyshev nodes on fiber, doFP = include the finite part integral implicitly
    (1 for yes, 0 for no)
    Outputs: the alphas and lambdas on all the fibers 
    """
    Allalphas = np.zeros(Nfib*(2*nPolys+3));
    Alllambdas = np.zeros(Nfib*3*N);
    for iFib in nb.prange(Nfib):
        M = calcMNumba(XsAll[iFib*3*N:(iFib+1)*3*N],cs, mu, N);
        if (doFP):
            MFP = FinitePartMatrix(XVecs[iFib*N:(iFib+1)*N,:],XsVec[iFib*N:(iFib+1)*N,:],FPMatrix,DiffMat,snodes,N);
            M+=MFP;
        K, J = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N,nPolys);
        b1 = b[iFib*3*N:(iFib+1)*3*N];
        b2 = b[3*N*Nfib+(iFib)*(2*nPolys+3):3*N*Nfib+(iFib+1)*(2*nPolys+3)];
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
        Allalphas[iFib*(2*nPolys+3):(iFib+1)*(2*nPolys+3)]=alphaU;
        Alllambdas[iFib*3*N:(iFib+1)*3*N] = lambdas;
    return np.concatenate((Alllambdas,Allalphas));
  
@nb.njit((nb.int64,nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
        nb.float64[:]),parallel=True,cache=c)
def calcKAlphas(Nfib,N,nPolys,XsVec,UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler, DpInv,I,allalphas):
    """
    Compute K*alpha for given alpha on all fibers (input allalphas) 
    See linSolveAllFibersForGM docstring for parameters 
    """
    Kalph = np.zeros(Nfib*3*N);
    for iFib in nb.prange(Nfib):
        K, _ = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N,nPolys);
        thisAlpha = allalphas[iFib*(2*nPolys+3):(iFib+1)*(2*nPolys+3)];
        Kalph[iFib*3*N:(iFib+1)*3*N] = np.dot(K,thisAlpha[:2*nPolys])+np.dot(I,thisAlpha[2*nPolys:]);
    return Kalph;

@nb.njit((nb.int64,nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
        nb.float64[:,:],nb.float64[:],nb.float64[:]),parallel=True,cache=c)
def calcKAlphasAndKstarLambda(Nfib,N,nPolys,XsVec,UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,\
    WeightedUpsamplingMat, DpInv,I,wIt,allalphas,alllambdas):
    """
    Compute K*alpha and K^T *lambda for inputs allalphas and alllambdas 
    See linSolveAllFibersForGM docstring for parameters 
    """
    Kalph = np.zeros(Nfib*3*N);
    Kstlam = np.zeros(Nfib*(2*nPolys+3))
    for iFib in nb.prange(Nfib):
        K, J = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],UpsampMat,UpsampledchebPolys, LeastSquaresDownsampler,DpInv,N,nPolys);
        thisAlpha = allalphas[iFib*(2*nPolys+3):(iFib+1)*(2*nPolys+3)];
        Kalph[iFib*3*N:(iFib+1)*3*N] = np.dot(K,thisAlpha[:2*nPolys])+np.dot(I,thisAlpha[2*nPolys:]);
        Kt = calcKtNumba(J,WeightedUpsamplingMat);
        Kstlam[iFib*(2*nPolys+3):(iFib+1)*(2*nPolys+3)] =  np.dot(np.concatenate((Kt,wIt)),alllambdas[iFib*3*N:(iFib+1)*3*N]);
    return Kalph, Kstlam;
