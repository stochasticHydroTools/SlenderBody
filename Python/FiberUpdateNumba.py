import numba as nb
import numpy as np
   
@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64))
def deAliasIntegralNumba(f, Lmat, UpsampMat, DownSampMat,DpInv,N):
    """
    Method to dealias the product of f (an N array) with each of the first
    N-1 Chebyshev polynomials
    Upsample to a 2N grid, perform multiplication and integration, then 
    downsample to an N point grid.
    Inputs: f = values of function f, Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid,
    UpsampMat, DownSampMat = matrices for upsampling and downsampling from N to 2N grid.  
    DpInv = psuedo-inverse of Cheb differentiation matrix on 2N point grid, N = number of coefficients 
    """
    # Upsample the multiplication of f with Chebyshev polys for anti-aliasing
    UpSampMulti = (f*(np.dot(UpsampMat,Lmat[:,:N-1])).T).T;
    # Integrals on the original grid (integrate on upsampled grid and downsample)
    OGIntegrals = np.dot(DownSampMat,np.dot(DpInv,UpSampMulti));
    return OGIntegrals;

@nb.njit((nb.float64[:], nb.float64[:], nb.float64[:]))
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

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64))
def calcKNumba(Xs,Lmat, UpsampMat, DownSampMat,DpInv,N):
    """
    Computes the matrix K(X). The only input is X_s, which
    has to be input as a 3N one-dimensional array.
    From X_s this method computes the normals and then 
    the matrix K.   
    """
    XsUpsampled = np.dot(UpsampMat,Xs);
    theta, phi, r = cart2sph(XsUpsampled[:,0],XsUpsampled[:,1],XsUpsampled[:,2]);
    n1x = -np.sin(theta);
    n1y = np.cos(theta);
    n1z = np.zeros(2*N);
    n2x = -np.cos(theta)*np.sin(phi);
    n2y = -np.sin(theta)*np.sin(phi);
    n2z = np.cos(phi);
    K = np.zeros((3*N,2*N-2));
    K[0::3,0:N-1]= deAliasIntegralNumba(n1x,Lmat, UpsampMat, DownSampMat,DpInv,N);
    K[1::3,0:N-1]= deAliasIntegralNumba(n1y,Lmat, UpsampMat, DownSampMat,DpInv,N);
    K[2::3,0:N-1]= deAliasIntegralNumba(n1z,Lmat, UpsampMat, DownSampMat,DpInv,N);
    K[0::3,N-1:2*N-2]= deAliasIntegralNumba(n2x,Lmat, UpsampMat, DownSampMat,DpInv,N);
    K[1::3,N-1:2*N-2]= deAliasIntegralNumba(n2y,Lmat, UpsampMat, DownSampMat,DpInv,N);
    K[2::3,N-1:2*N-2]= deAliasIntegralNumba(n2z,Lmat, UpsampMat, DownSampMat,DpInv,N);
    return K;

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.float64[:],nb.int64))
def calcKtNumba(K,w,N):
    """
    Calculates the matrix K^* (L^2 adjoint of K) given K, weights
    on the Chebyshev grid, and number of nodes N
    """
    Kt = (K.copy()).T;
    Kt[:,0::3]*=w;
    Kt[:,1::3]*=w;
    Kt[:,2::3]*=w;
    return Kt;

@nb.njit(nb.float64[:,:](nb.float64[:],nb.float64[:],nb.float64,nb.int64))
def calcMNumba(Xs,c, mu, N):
    """
    Calculates the local drag matrix M. The only input is X_s which
    has to be input as a 3N one-dimensional array.
    From X_s this method computes the matrix M (changes for 
    ellipsoidal vs. cylindrical fibers).
    """
    M = np.zeros((3*N,3*N));
    for j in range(N):
        v = Xs[j*3:j*3+3];
        XsXs = np.outer(v,v);
        M[j*3:j*3+3,j*3:j*3+3]=1/(8*np.pi*mu)*\
         (c[j]*(np.identity(3)+XsXs)+  np.identity(3)-3*XsXs);
    return M;

@nb.njit((nb.float64[:,:],nb.float64[:,:],nb.float64,nb.float64,nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.int64))
def concatMats(K,Kt,impco,dt,M,D4BC,I,wIt,N):
    """
    Form the matrices for Schur complement solve
    """
    B = np.concatenate((K-impco*dt*np.dot(M,np.dot(D4BC,K)),\
     I-impco*dt*np.dot(M,np.dot(D4BC,I))),axis=1);
    C = np.concatenate((Kt,wIt));
    D1 = np.zeros((2*N-2,2*N+1));
    D2 = impco*dt*np.dot(wIt, np.dot(D4BC,K));
    D3 = impco*dt*np.dot(wIt, np.dot(D4BC,I));
    D = np.concatenate((D1,np.concatenate((D2,D3),axis=1)));
    return B, C, D

@nb.njit((nb.int64,nb.int64,nb.float64[:],nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64[:],nb.float64,nb.float64,nb.float64[:],\
          nb.float64, nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:,:],nb.float64[:],nb.float64[:,:],\
          nb.float64[:,:],nb.float64[:,:]),parallel=True)
def linSolveAllFibers(Nfib,N,ptsCheb,XsVec,XsAll,nLvel,forceExt,dt,impco, cs, mu, Lmat, Upsamp, DownSamp, DpInv, w,D4BC,I,wIt):
    """
    Linear solve on all fibers together to obtain alphas and lambdas. 
    Inputs: Nfib = number of fibers, N = number of points per fibers, ptsCheb = Chebyshev points (as an N*Nfib x 3 array), 
    XsVec = tangent vectors (as an N*Nfib x 3 array), XsAll = tangent vectors, row stacked (because numba doesn't support 
    reshape), nLvel = non-local and background velocities on the fibers, forceExt = external forcing, dt = timestep, 
    impco = implicit coefficient for linear solves, cs = local drag coefficients for each s, mu = fluid viscosity, 
    Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid, 
    UpsampMat, DownSampMat = matrices for upsampling and downsampling from N to 2N grid. DpInv = psuedo-inverse of Cheb
    differentiation matrix, w = Chebyshev weights, D4BC = bending force calculation operator, I = N x 3 identity matrix, 
    wIt = 3 x N matrix  that integrates functions on the grid.
    Outputs: the alphas on all fibers, velocities K*alpha, and lambdas
    """
    Allalphas = np.zeros(Nfib*(2*N-2));
    Allvelocities = np.zeros(Nfib*3*N);
    Alllambdas = np.zeros(Nfib*3*N);
    for iFib in nb.prange(Nfib):
        M = calcMNumba(XsAll[iFib*3*N:(iFib+1)*3*N],cs, mu, N);
        K = calcKNumba(XsVec[iFib*N:(iFib+1)*N,:],Lmat, Upsamp, DownSamp,DpInv,N);
        Kt = calcKtNumba(K,w,N);
        B, C, D = concatMats(K,Kt,impco,dt,M,D4BC,I,wIt,N)
        fE = np.dot(D4BC,ptsCheb[iFib*3*N:(iFib+1)*3*N]);
        exF = forceExt[iFib*3*N:(iFib+1)*3*N];
        myNL = nLvel[iFib*3*N:(iFib+1)*3*N];
        Minv = np.linalg.inv(M);
        RHS = np.dot(C,fE+exF)+np.concatenate((np.zeros(2*N-2),\
            -np.dot(wIt,fE)))+np.dot(C,np.dot(Minv,myNL));
        S = np.dot(C,np.dot(Minv,B))+D;
        alphaU,res,rank,s = np.linalg.lstsq(S,RHS,-1);
        vel = np.dot(K,alphaU[0:2*N-2])+np.dot(I,alphaU[2*N-2:2*N+1]);
        lambdas = np.dot(Minv,vel-myNL)-fE-exF- impco*dt*np.dot(D4BC,vel);
        Allalphas[iFib*(2*N-2):(iFib+1)*(2*N-2)]=alphaU[:2*N-2];
        Allvelocities[iFib*3*N:(iFib+1)*3*N]=vel;
        Alllambdas[iFib*3*N:(iFib+1)*3*N] = lambdas;
    return Allalphas, Allvelocities, Alllambdas

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64,nb.float64[:]))
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
    nb.float64[:,:],nb.float64[:],nb.float64[:],nb.float64,nb.float64),parallel=True)
def updateXsNumba(Nfib,N,XAllNow,XsAllNow,XsforNL,UpsampMat,DownSampMat,Lmat,AllAlphas,AllVels,dt,L):
    """
    Method to update the tangent vectors and positions using Rodriguez rotation and Chebyshev integration
    for many fibers in parallel. 
    Inputs (too many): Nfib = number of fibers, N = number of points per fiber. XAllNow = current Chebyshev 
    points (time n) on the fiber, XsAllNow = current tangent vectors (time n), XsforNL = tangent vectors for 
    non-local (Omega) calculations, UpsampMat, DownSampMat = matrices for upsampling and downsampling from N 
    to 2N grid. Lmat = matrix of coefficients to values (Chebyshev polynomial values) on the N grid, 
    AllAlphas = alphas for the fibers, AllVels = fiber velocities using dX/dt = K*alpha (so that we match the 
    velocity at the first point), dt = timestep, L = fiber length
    Outputs: all of the new tangent vectors and positions
    """
    AllXs = np.zeros((Nfib*N,3));
    AllNewX = np.zeros((Nfib*N,3));
    for iFib in nb.prange(Nfib):
        XsforOmega = XsforNL[iFib*N:(iFib+1)*N,:];
        # Compute Omega on the upsampled grid
        XsUpsampled = np.dot(UpsampMat,XsforOmega);
        theta, phi, r = cart2sph(XsUpsampled[:,0],XsUpsampled[:,1],XsUpsampled[:,2]);
        n1 = np.zeros((2*N,3));
        n2 = np.zeros((2*N,3));
        n1[:,0] = -np.sin(theta);
        n1[:,1] = np.cos(theta);
        n2[:,0] = -np.cos(theta)*np.sin(phi);
        n2[:,1] = -np.sin(theta)*np.sin(phi);
        n2[:,2] = np.cos(phi);
        ChebPolys = Lmat[:,:N-1];
        alpha = AllAlphas[iFib*(2*N-2):(iFib+1)*(2*N-2)];
        g1 = np.dot(UpsampMat,np.dot(ChebPolys,alpha[0:N-1]));
        g2 = np.dot(UpsampMat,np.dot(ChebPolys,alpha[N-1:2*N-2]));
        Omega = (g1*(n2.T)-g2*(n1.T)).T;
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
        X+= -X[0,:]+XAllNow[iFib*N,:]+dt*AllVels[3*iFib*N:3*iFib*N+3]
        AllXs[iFib*N:(iFib+1)*N,:]=Xsp1.T;
        AllNewX[iFib*N:(iFib+1)*N,:]=X;
    return AllXs, AllNewX;