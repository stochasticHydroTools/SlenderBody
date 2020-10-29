import numba as nb
import numpy as np

"""
Functions for many fibers that use numba to speed up the calculations
"""
   
@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64),cache=True)
def deAliasIntegralNumba(f, Lmat, UpsampMat,DpInv,N):
    """
    Method to dealias the product of f (an N array) with each of the first
    N-1 Chebyshev polynomials
    Upsample to a 2N grid, perform multiplication and integration, then 
    downsample to an N point grid.
    Inputs: f = values of function f, Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid,
    UpsampMat, DpInv = psuedo-inverse of Cheb differentiation matrix on 2N point grid, N = number of coefficients 
    """
    # Upsample the multiplication of f with Chebyshev polys for anti-aliasing
    UpSampMulti = (f*(np.dot(UpsampMat,Lmat[:,:N-1])).T).T;
    # Integrals on the original grid (integrate on upsampled grid and downsample)
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

@nb.njit((nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.int64),cache=True)
def calcKNumba(Xs,Lmat, UpsampMat, stackUpSampMat,DpInv,w2N,N):
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
    n1z = np.zeros(2*N);
    n2x = -np.cos(theta)*np.sin(phi);
    n2y = -np.sin(theta)*np.sin(phi);
    n2z = np.cos(phi);
    J = np.zeros((6*N,2*N-2));
    J[0::3,0:N-1]= deAliasIntegralNumba(n1x,Lmat, UpsampMat,DpInv,N);
    J[1::3,0:N-1]= deAliasIntegralNumba(n1y,Lmat, UpsampMat,DpInv,N);
    J[2::3,0:N-1]= deAliasIntegralNumba(n1z,Lmat, UpsampMat,DpInv,N);
    J[0::3,N-1:2*N-2]= deAliasIntegralNumba(n2x,Lmat, UpsampMat,DpInv,N);
    J[1::3,N-1:2*N-2]= deAliasIntegralNumba(n2y,Lmat, UpsampMat,DpInv,N);
    J[2::3,N-1:2*N-2]= deAliasIntegralNumba(n2z,Lmat, UpsampMat,DpInv,N);
    UTWU = np.dot(stackUpSampMat.T,np.dot(np.diag(np.repeat(w2N,3)),stackUpSampMat));
    K = np.linalg.solve(UTWU,np.dot(stackUpSampMat.T,np.dot(np.diag(np.repeat(w2N,3)),J)));
    return K, J;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:],nb.float64[:,:]),cache=True)
def calcKtNumba(J,w2N,stackUpsampMat):
    """
    Calculates the matrix K^* (L^2 adjoint of K) given K, weights
    on the 2N Chebyshev grid, and stacked upsampling matrix
    """
    return np.dot(J.T,np.dot(np.diag(np.repeat(w2N,3)),stackUpsampMat));

@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:],nb.float64,nb.int64),cache=True)
def calcMNumba(Xs,c, mu, N):
    """
    Calculates the local drag matrix M. 
    Inputs: Xs = fiber tangent vectors as a 3N array, c = N vector of local drag coefficients, 
    mu = viscosity, N = number of points 
    """
    M = np.zeros((3*N,3*N));
    for j in range(N):
        v = Xs[j*3:j*3+3];
        XsXs = np.outer(v,v);
        M[j*3:j*3+3,j*3:j*3+3]=1/(8*np.pi*mu)*\
         (c[j]*(np.identity(3)+XsXs)+  np.identity(3)-3*XsXs);
    return M;

@nb.njit(nb.float64[:](nb.float64[:],nb.float64[:],nb.float64[:],nb.float64,nb.int64,nb.int64),cache=True)
def calcLocalVelocities(Xs_nonLoc,forceDsAll,localcs,mu,N,Nfib):
    """
    Compute the local velocities, i.e. M^loc*forceDen for all fibers. 
    Inputs: Xs_nonLoc = list of tangent vectors as a 3*N*nFib 1d array, forceDsAll = 
    list of all force densities as a 3*N*nFib 1d array, localcs = local drag coefficient, 
    mu = fluid viscosity, N = number of Cheb points per fiber, N = number of fibs 
    """
    LocalOnly = np.zeros(len(Xs_nonLoc));
    for iFib in range(Nfib):
        Xs = Xs_nonLoc[iFib*3*N:(iFib+1)*3*N];
        forceD = forceDsAll[iFib*3*N:(iFib+1)*3*N];
        # Compute self velocity from RPY (which will be subtracted), finite part integral
        # velocity, and purely local velocity (this is needed in the non-local corrections
        # for very close fibers).
        M = calcMNumba(Xs,localcs,mu,N);
        LocalOnly[iFib*3*N:(iFib+1)*3*N] = np.dot(M,forceD);
    return LocalOnly;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
         nb.float64[:],nb.int64),cache=True)
def FinitePartMatrixColumns(ChebPoints,Xs,FPMatrix,DiffMat,s,N):
    """
    3N x 3N Matrix for the finite part integral M_FP*f, formed column by column
    """
    ActualFPMat = np.zeros((3*N,3*N));
    Xss = np.dot(DiffMat,Xs);
    for kPt in range(N):
        for iD in range(3):
            FDens = np.zeros((N,3));
            uFP = np.zeros((N,3));
            FDens[kPt,iD]=1;
            fprime = np.dot(DiffMat,FDens);
            for iPt in range(N):
                Xsdotf = np.dot(Xs[iPt,:],FDens[iPt,:]);
                Xssdotf = np.dot(Xss[iPt,:],FDens[iPt,:]);
                Xsdotfprime = np.dot(Xs[iPt,:],fprime[iPt,:]);
                for jPt in range(N):
                    if (iPt==jPt):
                        uFP[iPt,:]+= (0.5*(Xs[iPt,:]*Xssdotf+Xss[iPt,:]*Xsdotf)+ fprime[iPt,:]+Xs[iPt,:]*Xsdotfprime) *FPMatrix[iPt,jPt];
                    else:
                        rvec = ChebPoints[iPt,:]-ChebPoints[jPt,:];
                        r = np.linalg.norm(rvec);
                        oneoverr = 1.0/r;
                        ds = s[jPt]-s[iPt];
                        oneoverds = 1.0/ds;
                        rdotf = np.dot(rvec,FDens[jPt,:]);
                        uFP[iPt,:]+=((FDens[jPt,:] + rvec*rdotf*oneoverr*oneoverr)*oneoverr*np.abs(ds)-\
                                    (FDens[iPt,:]+Xs[iPt,:]*Xsdotf))*oneoverds*FPMatrix[iPt,jPt]; 
            ActualFPMat[:,3*kPt+iD] = np.reshape(uFP.copy(),3*N);
    return ActualFPMat;

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

@nb.njit((nb.int64,nb.int64,nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64,nb.float64,nb.float64[:],\
          nb.float64, nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:,:],\
          nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.int64),parallel=True,cache=True)
def linSolveAllFibersForGM(Nfib,N,b,XsVec,XsAll,dt,impco, cs, mu, Lmat, Upsamp, stackUpsamp, stackDownSamp,DpInv, w,w2N,D4BC,I,wIt,\
    XVecs,FPMatrix,DiffMat,snodes,doFP):
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
    for iFib in nb.prange(Nfib):
        M = calcMNumba(XsAll[iFib*3*N:(iFib+1)*3*N],cs, mu, N);
        if (doFP):
            MFP = FinitePartMatrix(XVecs[iFib*N:(iFib+1)*N,:],XsVec[iFib*N:(iFib+1)*N,:],FPMatrix,DiffMat,snodes,N);
            M+=MFP;
        K, J = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],Lmat, Upsamp, stackUpsamp,DpInv,w2N,N);
        b1 = b[iFib*3*N:(iFib+1)*3*N];
        b2 = b[3*N*Nfib+(iFib)*(2*N+1):3*N*Nfib+(iFib+1)*(2*N+1)];
        Kt = calcKtNumba(J,w2N,stackUpsamp);
        #Kt = np.dot(K.T,np.diag(np.repeat(w,3)));
        B = np.concatenate((K-impco*dt*np.dot(M,np.dot(D4BC,K)),\
            I-impco*dt*np.dot(M,np.dot(D4BC,I))),axis=1);
        C = np.concatenate((Kt,wIt));
        #Minv = np.linalg.inv(M);
        RHS = b2+np.dot(C,np.linalg.solve(M,b1));
        S = np.dot(C,np.linalg.solve(M,B));
        alphaU,res,rank,k = np.linalg.lstsq(S,RHS,-1);
        Balph = np.dot(B,alphaU);
        lambdas = np.linalg.solve(M,Balph-b1);
        Allalphas[iFib*(2*N+1):(iFib+1)*(2*N+1)]=alphaU;
        Alllambdas[iFib*3*N:(iFib+1)*3*N] = lambdas;
    return np.concatenate((Alllambdas,Allalphas));
  
@nb.njit((nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.float64[:]),\
        parallel=True,cache=True)
def calcKAlphas(Nfib,N,XsVec,Lmat, Upsamp, stackUpsamp, DpInv,I,allalphas,w2N):
    """
    Compute K*alpha for given alpha on all fibers (input allalphas) 
    See linSolveAllFibersForGM docstring for parameters 
    """
    Kalph = np.zeros(Nfib*3*N);
    for iFib in nb.prange(Nfib):
        K, _ = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],Lmat, Upsamp, stackUpsamp,DpInv,w2N,N);
        thisAlpha = allalphas[iFib*(2*N+1):(iFib+1)*(2*N+1)];
        Kalph[iFib*3*N:(iFib+1)*3*N] = np.dot(K,thisAlpha[:2*N-2])+np.dot(I,thisAlpha[2*N-2:]);
    return Kalph;

@nb.njit((nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
        nb.float64[:,:], nb.float64[:],nb.float64[:],nb.float64[:],nb.float64[:]),parallel=True,cache=True)
def calcKAlphasAndKstarLambda(Nfib,N,XsVec,Lmat, Upsamp, stackUpsamp,stackDownSamp, DpInv,I,wIt,w,w2N,allalphas,alllambdas):
    """
    Compute K*alpha and K^T *lambda for inputs allalphas and alllambdas 
    See linSolveAllFibersForGM docstring for parameters 
    """
    Kalph = np.zeros(Nfib*3*N);
    Kstlam = np.zeros(Nfib*(2*N+1))
    for iFib in nb.prange(Nfib):
        K, J = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],Lmat, Upsamp, stackUpsamp,DpInv,w2N,N);
        thisAlpha = allalphas[iFib*(2*N+1):(iFib+1)*(2*N+1)];
        Kalph[iFib*3*N:(iFib+1)*3*N] = np.dot(K,thisAlpha[:2*N-2])+np.dot(I,thisAlpha[2*N-2:]);
        Kt = calcKtNumba(J,w2N,stackUpsamp);
        #Kt = np.dot(K.T,np.diag(np.repeat(w,3)));
        Kstlam[iFib*(2*N+1):(iFib+1)*(2*N+1)] =  np.dot(np.concatenate((Kt,wIt)),alllambdas[iFib*3*N:(iFib+1)*3*N]);
    return Kalph, Kstlam;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.float64[:]),cache=True)
def intCoefficients(incoefs,N,dom):
    """
    Function that applies the Chebyshev integration matrix to the series of
    N coefficients incoefs on domain dom.
    There is an unknown constant since this is indefinite integration
    that we set to zero here.
    Inputs: incoefs = numpy array (N x ? dims) of Chebyshev coefficients, 
    N = number of coefficients in each series, dom = domain ([0,L])
    """
    intcoefs = np.zeros(incoefs.shape);
    intcoefs[1,:]=incoefs[0,:]-0.5*incoefs[2,:];
    for j in range(2,N-1):
        intcoefs[j,:]=1.0/(2*j)*(incoefs[j-1,:]-incoefs[j+1,:]);
    intcoefs[N-1,:]=1.0/(2*(N-1))*incoefs[N-2,:];
    intcoefs*=0.5*(dom[1]-dom[0]);
    return intcoefs;

@nb.njit((nb.int64,nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],\
    nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64,nb.float64),cache=True,parallel=True)
def updateXsNumba(Nfib,N,XAllNow,XsAllNow,XsforNL,UpsampMat,DownSampMat,Lmat,Dmat,AllVels,dt,L):
    """
    Method to update the tangent vectors and positions using Rodriguez rotation and Chebyshev integration
    for many fibers in parallel. 
    Inputs (too many): Nfib = number of fibers, N = number of points per fiber. XAllNow = current Chebyshev 
    points (time n) on the fiber, XsAllNow = current tangent vectors (time n), XsforNL = tangent vectors for 
    non-local (Omega) calculations, UpsampMat, DownSampMat = matrices for upsampling and downsampling from N 
    to 2N grid. Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid, 
    Dmat = differentiation matrix, AllVels = fiber velocities using dX/dt = K*alpha dt = timestep, L = fiber length
    Outputs: all of the new tangent vectors and positions
    """
    AllXs = np.zeros((Nfib*N,3));
    AllNewX = np.zeros((Nfib*N,3));
    for iFib in nb.prange(Nfib):
        XsforOmega = XsforNL[iFib*N:(iFib+1)*N,:];
        # Compute DKalpha on the N point grid
        DKalpha = np.dot(Dmat,AllVels[iFib*N:(iFib+1)*N,:]);
        # Upsample DKalpha and cross with upsampled Xs
        DKalphaUp = np.dot(UpsampMat,DKalpha);
        XsUpsampled = np.dot(UpsampMat,XsforOmega);
        # Compute Omega on the upsampled grid
        Omega = np.cross(XsUpsampled,DKalphaUp);
        # Downsample Omega
        Omega = np.dot(DownSampMat,Omega);
        nOm = np.sqrt(Omega[:,0]*Omega[:,0]+Omega[:,1]*Omega[:,1]+Omega[:,2]*Omega[:,2]);
        k = ((Omega.T) / nOm).T;
        k[nOm < 1e-6,:]=0;
        Xsrs = XsAllNow[iFib*N:(iFib+1)*N,:];
        # Rodriguez rotation
        kdotXsAll = np.dot(k,Xsrs.T); # this is wasteful but it's the only thing that got numba parallel to work
        kdotXs = np.zeros(N);
        for iPt in range(N):
            kdotXs[iPt] = kdotXsAll[iPt,iPt]; # extract diagonal elements of all possible dot products
        Xsp1 = Xsrs.T*np.cos(nOm*dt)+np.cross(k,Xsrs).T*np.sin(nOm*dt)+ \
            k.T*kdotXs*(1-np.cos(nOm*dt));
        Xshat = np.linalg.solve(Lmat,Xsp1.T); # coefficients
        Xhat = intCoefficients(Xshat,N,np.array([0,L])); # one integration
        X = np.dot(Lmat,Xhat); # back to real space
        # Add the constant velocity so that point 0 is the same
        X += -X[0,:]+XAllNow[iFib*N,:]+dt*AllVels[iFib*N,:]
        AllXs[iFib*N:(iFib+1)*N,:]=Xsp1.T;
        AllNewX[iFib*N:(iFib+1)*N,:]=X;
    return AllXs, AllNewX;
