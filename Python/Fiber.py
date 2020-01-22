import numpy as np
import scipy.linalg as sp
import chebfcns as cf
import EwaldUtils as ewc
import FiberUtils as fc

class Fiber(object):
    """ 
    The object code for a fiber
    """
    def __init__(self, L, epsilon,el=0,Eb=1,mu=1,N=16,X=None,Xs=None,Lx=None,Ly=None,Lz=None):
        """
        Constructor. Object variables are:
        L = fiber length, epsilon = fiber aspect ratio, el = shape of fibers (1 for 
        ellipsoidal, 0 for cylindrical), Eb = bending stiffness, mu = fluid viscosity. 
        N = number of points on discretized fiber, X = input locations, Xs = input tangent
        vectors (both at t=0).
        """
        # Default is to initialize X
        self._L = L;
        self._epsilon = epsilon;
        self._el = el;
        self._Eb = Eb;
        self._mu = mu;
        self._N = N;
        self._X = X;
        self._Xs = Xs;
        self._lambdas = np.zeros((3*N));
        self._lambdaprev = np.zeros((3*N));
        # Chebyshev grid and stuff
        self._s,self._w = cf.chebpts(self._N,[0,self._L],1);
        # If X is none, initialize to a straight line inside the box
        if X is None:
            self.initFib(Lx,Ly,Lz);
        self._Xprev = self._X.copy();
        self._Xsprev = self._Xs.copy();
        self._Lmat = cf.Lmat(self._N,cf.th1(self._N));
        self._Dmat = cf.diffMat(1,[0,self._L],N,N,1,1);
        self.initIs();
        self.initD4BC();
        self.initFPMatrix();

    def initFib(self,Lx,Ly,Lz):
        """
        Initialize straight fiber locations and tangent 
        vectors on [0,Lx] x [0,Ly] x [0,Lz]
        """
        # Choose a random start point
        start = np.random.rand(3)*[Lx, Ly, Lz];
        u=1-2*np.random.rand();
        v=np.sqrt(1-u*u);
        w=2*np.pi*np.random.rand();
        tangent = np.array([v*np.cos(w), v*np.sin(w), u]);
        self._Xs = np.tile(tangent,self._N);
        self._X = np.reshape(start+np.reshape(self._s,(self._N,1))*tangent,3*self._N);

    def initIs(self):
        """
        Initialize the identity matrix and matrix that takes 
        definite integrals
        """
        N = self._N;
        self._I = np.zeros((3*N,3));
        self._wIt = np.zeros((3,3*N));
        for j in range(N):
            self._I[3*j:3*j+3,:]=np.identity(3);
            self._wIt[:,3*j:3*j+3]=np.identity(3)*self._w[j];
        
    def initD4BC(self):
        """
        Compute the operator D_4^BC that is necessary when computing 
        the bending forces
        """
        N = self._N;
        R = cf.RSMat(N,N+4);
        B2 = cf.diffMat(2,[0,self._L],2,N+4,2,2);
        B3 = cf.diffMat(3,[0,self._L],2,N+4,2,2);
        T = np.concatenate((R,B2,B3));
        RHS = np.concatenate((np.identity(N),np.zeros((4,N))));
        Tmat = np.linalg.solve(T,RHS);
        D4Up = cf.diffMat(4,[0,self._L],N,N+4,1,2);
        OneD4BC = -self._Eb*np.dot(D4Up,Tmat);
        self._D4BC = np.zeros((3*N,3*N));
        self._D4BC[0::3,0::3]=OneD4BC;
        self._D4BC[1::3,1::3]=OneD4BC;
        self._D4BC[2::3,2::3]=OneD4BC;
    
    def calcfE(self,X):
        """
        Calculates the bending force fE
        Inputs: position X. 
        Outpts: bending forces fE
        """
        return np.dot(self._D4BC,X);
    
    def initFPMatrix(self):
        """
        Initialize the matrix for the finite part integral. 
        Uses the adjoint method of Anna Karin Tornberg. 
        """
        if (self._N > 40):
            print 'Warning - the finite part integral is not \
                well conditioned with > 40 pts, loss of accuracy expected';
        k = np.arange(self._N);
        sscale1 = -1+2*self._s/self._L;
        sscale = np.reshape(sscale1,(self._N,1)); # Rescale everything to [-1,1]
        q=(1.0+(-1.0)**(k+1.0)-2*sscale**(k+1.0))/(k+1.0);
        V = np.vander(sscale1,increasing=True);
        self._FPMatrix = np.linalg.solve(V.T,q.T);        
        
    def calcKs(self,Xs):
        """
        Computes the matrix K(X). The only input is X_s, which
        has to be input as a 3N one-dimensional array.
        From X_s this method computes the normals and then 
        the matrix K.   
        """
        N = self._N;
        Rup = cf.RSMat(2*N,N,1,1); # resample to 2N type 1 grid
        Rdwn = cf.RSMat(N,2*N,1,1);
        Dup = cf.diffMat(1,[0,self._L],2*N,2*N,1,1);
        Dinvup = np.linalg.pinv(Dup);
        theta, phi, r = Fiber.cart2sph(Xs[0::3],Xs[1::3],Xs[2::3]);
        n1x = -np.sin(theta);
        n1y = np.cos(theta);
        n1z = np.zeros(N);
        n2x = -np.cos(theta)*np.sin(phi);
        n2y = -np.sin(theta)*np.sin(phi);
        n2z = np.cos(phi);
        K = np.zeros((3*N,2*N-2));
        K[0::3,0:N-1]= Fiber.deAliasInt(Rup,Rdwn,Dinvup,\
            n1x,self._Lmat[:,:N-1],N);
        K[1::3,0:N-1]= Fiber.deAliasInt(Rup,Rdwn,Dinvup,\
            n1y,self._Lmat[:,:N-1],N);
        K[2::3,0:N-1]= Fiber.deAliasInt(Rup,Rdwn,Dinvup,\
            n1z,self._Lmat[:,:N-1],N);
        K[0::3,N-1:2*N-2]= Fiber.deAliasInt(Rup,Rdwn,Dinvup,\
            n2x,self._Lmat[:,:N-1],N);
        K[1::3,N-1:2*N-2]= Fiber.deAliasInt(Rup,Rdwn,Dinvup,\
            n2y,self._Lmat[:,:N-1],N);
        K[2::3,N-1:2*N-2]= Fiber.deAliasInt(Rup,Rdwn,Dinvup,\
            n2z,self._Lmat[:,:N-1],N);
        Kt = K.copy();
        Kt[0::3,:]*=np.reshape(self._w,(N,1));
        Kt[1::3,:]*=np.reshape(self._w,(N,1));
        Kt[2::3,:]*=np.reshape(self._w,(N,1));
        Kt = Kt.T;
        return K, Kt;

    def calcM(self,Xs):
        """
        Calculates the local drag matrix M. The only input is X_s which
        has to be input as a 3N one-dimensional array.
        From X_s this method computes the matrix M (changes for 
        ellipsoidal vs. cylindrical fibers).
        """
        t = self._s*2/self._L-1;
        aeps = 2*self._epsilon;
        ccyl = np.log((2*(1-t**2)+2*np.sqrt((1-t**2)**2+\
            4*aeps**2))/aeps**2);
        cs = ccyl*(1-self._el)-np.log(self._epsilon**2)*self._el;
        matlist = [None]*self._N;
        for j in xrange(self._N):
            v = Xs[j*3:j*3+3];
            XsXs = np.outer(v,v);
            matlist[j]=1/(8*np.pi*self._mu)*\
             (cs[j]*(np.identity(3)+XsXs)+ \
              np.identity(3)-3*XsXs);
        Mloc =sp.block_diag(*matlist);
        return Mloc;
        
    
    def updateX(self,dt,tint,iters,nLvel,exF):
        """
        This method solves the linear system for 
        lambda and alpha for a given RHS. 
        Inputs: dt = timestep, tint = type of temporal integrator (2 for
        Crank-Nicolson, 1 for backward Euler, -1 for forward Euler), 
        iters= iteration count (really necessary to copy the previous lambda at the 
        right point), nLvel = non-local velocity as a 3N vector 
        (comes from the Fluid class), exF = external forces as a 3N vector
        (comes from the ExForces class)
        """
        addxs = float(np.abs(tint)-1)/2;
        Xsarg = (1+addxs)*self._Xs - addxs*self._Xsprev;
        M = self.calcM(Xsarg);
        K, Kt = self.calcKs(Xsarg);
        impco = -(1.0/3*tint**2-0.5*tint-5.0/6);
        # Schur complement solve
        B = np.concatenate((K-impco*dt*np.dot(M,np.dot(self._D4BC,K)),\
         self._I-impco*dt*np.dot(M,np.dot(self._D4BC,self._I))),axis=1);
        C = np.concatenate((Kt,self._wIt));
        D1 = np.zeros((2*self._N-2,2*self._N+1));
        D2 = impco*dt*np.dot(self._wIt, np.dot(self._D4BC,K));
        D3 = impco*dt*np.dot(self._wIt, np.dot(self._D4BC,self._I));
        D = np.concatenate((D1,np.concatenate((D2,D3),axis=1)));
        fE = self.calcfE(self._X);
        RHS = np.dot(C,fE+exF)+np.concatenate((np.zeros(2*self._N-2),\
            -np.dot(self._wIt,fE)))+np.dot(C,np.linalg.solve(M,nLvel));
        S = np.dot(C,np.linalg.solve(M,B))+D;
        self._alphaU,res,rank,s = np.linalg.lstsq(S,RHS,-1);
        self._vel = np.dot(K,self._alphaU[0:2*self._N-2])+\
            np.dot(self._I,self._alphaU[2*self._N-2:2*self._N+1]);
        if (iters==0):
            self._lambdaprev = self._lambdas.copy();
        self._lambdasm1 = self._lambdas.copy();
        self._lambdas = np.linalg.solve(M,self._vel-nLvel)-fE-exF- \
                impco*dt*np.dot(self._D4BC,self._vel);
        
    def notConverged(self,tol):
        """
        Check if fixed point iteration is converged with tolerance tol. 
        Return 1 if NOT converged and 0 if converged. 
        """
        diff = self._lambdas-self._lambdasm1;
        # The 1.0 is here so that if lambda = 0 the iteration will converge. 
        reler = np.linalg.norm(diff)/max(1.0,np.linalg.norm(self._lambdas));
        return (reler > tol);
    
    def updateXs(self,dt,tint,exactinex=1):
        """ 
        Update the fiber position and tangent vectors
        Inputs: dt = timestep and tint = temporal integrator.
        If exactinex = 1, the inextensibility will be done
        EXACTLY according to the Rodriguez rotation formula. 
        Otherwise, it will be inexact. 
        """
        if (not exactinex): # inexact update, just add the velocity*dt to X
            self._Xprev = self._X.copy();
            self._X+= dt*self._vel;
            self._Xsprev = self._Xs.copy();
            self._Xs = np.reshape(np.dot(self._Dmat,np.reshape(self._X,(self._N,3))),3*self._N);
            return;
        # For exact inextensbility
        addxs = float(np.abs(tint)-1)/2;
        Xsarg = (1+addxs)*self._Xs - addxs*self._Xsprev;
        theta, phi, r = Fiber.cart2sph(Xsarg[0::3],\
            Xsarg[1::3],Xsarg[2::3]);
        n1x = -np.sin(theta);
        n1y = np.cos(theta);
        n1z = np.zeros(self._N);
        n2x = -np.cos(theta)*np.sin(phi);
        n2y = -np.sin(theta)*np.sin(phi);
        n2z = np.cos(phi);
        Omegax = np.dot(self._Lmat[:,:self._N-1],\
            self._alphaU[0:self._N-1])*n2x - \
             np.dot(self._Lmat[:,:self._N-1],\
            self._alphaU[self._N-1:2*self._N-2])*n1x; #g1*n2-g2*n1
        Omegay = np.dot(self._Lmat[:,:self._N-1],\
            self._alphaU[0:self._N-1])*n2y - \
             np.dot(self._Lmat[:,:self._N-1],\
            self._alphaU[self._N-1:2*self._N-2])*n1y; #g1*n2-g2*n1
        Omegaz = np.dot(self._Lmat[:,:self._N-1],\
            self._alphaU[0:self._N-1])*n2z - \
             np.dot(self._Lmat[:,:self._N-1],\
            self._alphaU[self._N-1:2*self._N-2])*n1z; #g1*n2-g2*n1
        nOm = np.sqrt(Omegax*Omegax+Omegay*Omegay+Omegaz*Omegaz); 
        k =  np.concatenate(([Omegax],[Omegay],[Omegaz])).T;
        k= k / np.reshape(nOm,(self._N,1));
        k[nOm < 1e-6,:]=0;
        nOm = np.reshape(nOm,(self._N,1));
        Xsrs = np.reshape(self._Xs,(self._N,3));
        # Rodriguez rotation
        Xsp1 = Xsrs*np.cos(nOm*dt)+np.cross(k,Xsrs)*np.sin(nOm*dt)+ \
            k*np.reshape(np.sum(k*Xsrs,axis=1),(self._N,1))* \
            (1-np.cos(nOm*dt));    
        # Compute the new X from Xs
        Xshat = np.linalg.solve(self._Lmat,Xsp1);
        Xhat = cf.intMat(Xshat,self._N,[0,self._L]);
        Xnp1 = np.dot(self._Lmat,Xhat); # back to real space
        # Add the constant
        Xnp1+= -Xnp1[0,:]+self._X[0:3]+dt*self._vel[0:3]
        self._Xprev = self._X.copy();
        self._Xsprev = self._Xs.copy();
        self._X = np.reshape(Xnp1,3*self._N);                
        self._Xs = np.reshape(Xsp1,3*self._N);
    
    def calcFPVelocity(self,Xarg,Xsarg,forceDs):
        """ 
        Calculate the velocity due to the finite part integral. 
        Inputs: Xarg = N x 3 array of the positions X, 
        Xsarg = N x 3 array of the tangent vectors, 
        forceDs = N x 3 array of the force densities. In general it
        is better to have these as inputs rather than object variables as
        it can get messy with the LMMs and different non-local velocities.
        Outpts: the velocity due to the finite part integral as a 3N one-dimensional vector
        """
        FPvel = np.zeros((3,self._N));
        Xss = np.dot(self._Dmat,Xsarg);
        fprime = np.dot(self._Dmat,forceDs);
        for iPt in xrange(self._N):
            # Call the C++ function to compute the correct density,
            # 1/(s[j]-s[i])*((I+Rhat*Rhat)*f[j]*abs(s[j]-s[i])/R - (I+Xs*Xs)*f[i])
            gloc = fc.FPDensity(self._N,Xarg[:,0],Xarg[:,1],Xarg[:,2],Xsarg[iPt,:],Xss[iPt,:],fprime[iPt,:],\
                    forceDs[:,0],forceDs[:,1],forceDs[:,2],self._s,iPt);
            gloc = np.reshape(gloc,(self._N,3)).T;
            FPvel[:,iPt]=0.5*self._L*np.dot(gloc,self._FPMatrix[:,iPt]);
        return 1.0/(8.0*np.pi*self._mu)*np.reshape(FPvel.T,3*self._N);
    
    def calcLocalVelocity(self,Xsarg,forceDs):
        """
        Purely local velocity = M*f. 
        Inputs: tangent vectors Xs as a 3N one-dimensional vector, and 
        forceDs as a 3N one-dimensional vector
        Outputs: 3N one-d vector of local velocities.
        """
        M = self.calcM(Xsarg);
        return np.dot(M,forceDs);
    
    def subtractRPY(self,Xarg,forces):
        """
        Method to subtract the SELF RPY kernel. 
        Inputs: Xarg (N x 3 vector of the fiber points - 
        this is a linear combination of X and Xprev depending 
        on the temporal integrator), and the forces, also as an N x 3 vector.
        Outpts: the velocity due to N blobs (self RPY kernel) in FREE SPACE
        as an N x 3 vector
        """
        a = np.sqrt(3.0/2.0)*self._epsilon*self._L;
        # Python version:
        #selfVel = np.zeros((self._N,3));
        #RPY0 = self.RPYTot([0.0,0.0,0.0]);
        #for iPt in xrange(self._N):
        #    selfVel[iPt,:]+= np.dot(forces[iPt,:],RPY0.T);
        #    for jPt in xrange(iPt+1,self._N):
                # Subtract free space kernel (no periodicity)
        #        rvec=Xarg[iPt,:]-Xarg[jPt,:];
        #        selfVel[iPt,:]+=ewc.RPYTot(rvec,forces[jPt,:],self._mu,a, 0);
        #        selfVel[jPt,:]+=ewc.RPYTot(rvec,forces[iPt,:],self._mu,a, 0);
        # Call the C++ function to add up the RPY kernel
        RPYSBTvel = ewc.RPYSBTKernel(self._N,Xarg[:,0],Xarg[:,1],Xarg[:,2],self._N,Xarg[:,0],\
                                     Xarg[:,1],Xarg[:,2],forces[:,0],forces[:,1],forces[:,2],\
                                     self._mu,a,0);
        return np.reshape(RPYSBTvel,(self._N,3));

    def RPYSBTKernel(self,Xtarg,Xarg,forces,sbt=0):
        """
        Method to compute the RPY kernel at a target Xtarg due to fiber self
        Inputs: Xtarg = target point (3 vector), Xarg = points on the fiber (as an
        N x 3 dimensional vector), forces = N x 3 vector of forces (WITH weights) on the fiber. 
        sbt = 0 for the RPY kernel (they are different when r < 2a) and sbt = 1 for the SBT kernel.
        Outpts: the velocity due to N blobs (self RPY kernel) at target point as a 3 vector.
        This is a free space kernel; all periodic shifts are done elsewhere.
        """
        RPYvel = np.zeros(3);
        r,_ = Xarg.shape;
        a = np.sqrt(3.0/2.0)*self._epsilon*self._L;
        RPYSBTvel = ewc.RPYSBTKernel(1,np.array([Xtarg[0]]),np.array([Xtarg[1]]),np.array([Xtarg[2]]),\
                                  r,Xarg[:,0],Xarg[:,1],Xarg[:,2],forces[:,0],forces[:,1],forces[:,2],\
                                  self._mu,a,sbt);
        return np.array(RPYSBTvel);


    def SBTKernelSplit(self,Xtarg,Xarg,forceDs,w1,w3,w5):
        """
        Method to compute the SBT kernel at point Xtarg using 
        modified quadrature (special quadrature from Ludvig and Barnett).
        We have to split the kernel into 1/R, 1/R^3, and 1/R^5 because the weights
        are different for the separate kernels. 
        Inputs: Xtarg = target point (as a 3 vector), Xarg = fiber points (as an r x 3 vector)
        (r gets determined in the code), forceDs = force densities (as an r x 3 vector), 
        N vectors w1, w3, w5 of quadrature weights (those come from special quad).
        """
        SBTvel = np.zeros(3);
        r,_ = Xarg.shape;
        for jPt in xrange(r):
            # Subtract free space kernel (no periodicity)
            rvec=Xtarg-Xarg[jPt,:];
            r=np.sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
            rdotf = rvec[0]*forceDs[jPt,0]+rvec[1]*forceDs[jPt,1]+rvec[2]*forceDs[jPt,2];
            u1 = forceDs[jPt,:]/r;
            u3 = (rvec*rdotf+((self._epsilon*self._L)**2)*forceDs[jPt,:])/r**3;
            u5 = -((self._epsilon*self._L)**2)*3.0*rvec*rdotf/r**5;
            SBTvel+= u1*w1[jPt]+u3*w3[jPt]+u5*w5[jPt];
        # Rescale (weights are on [-1,1], and divide by viscosity
        return 1.0/(8.0*np.pi*self._mu)*SBTvel;

    """
    Python versions of C++ functions - all called in C++ now.
    def RPYTot(self,rvec,sbt=0):
        #
        #The total RPY kernel for a given input vector rvec.
        #Another input is whether the kernel is sbt (for sbt=1)
        #or RPY (for sbt=0). The kernels change close to the fiber.
        #Output is the matrix of the kernel.
        #
        a = np.sqrt(3.0/2.0)*self._epsilon*self._L;
        rvec = np.reshape(rvec,(1,3));
        r=np.sqrt(rvec[:,0]*rvec[:,0]+rvec[:,1]*rvec[:,1]+rvec[:,2]*rvec[:,2]);
        rhat=rvec/r;
        rhat[np.isnan(rhat)]=0;
        RR = np.dot(rhat.T,rhat);
        return 1.0/self._mu*(Fiber.FT(r,a,sbt)*(np.identity(3)-RR)+Fiber.GT(r,a,sbt)*RR);
    
    # Next 2 methods are utilities for RPY kernel
    @staticmethod
    def FT(r,a,sbt):
        if (r>2*a or sbt): # RPY far or slender body
            val = (2*a**2 + 3*r**2)/(24*np.pi*r**3);
        else:
            val = (32*a - 9*r)/(192*a**2*np.pi);
        return val;
    
    @staticmethod
    def GT(r,a,sbt):
        if (r>2*a or sbt): # RPY far or slender body
            val = (-2*a**2 + 3*r**2)/(12*np.pi*r**3);
        else:
            val = (16*a - 3*r)/(96*a**2*np.pi);
        return val;
    """
    
    def resample(self,Xarg,Nrs,typetarg=1):
        """
        Resample the fiber at Nrs type 1 Chebyshev nodes
        Inputs: Xarg = a self._N x 3 vector of the fiber points,
        Nrs = number of points where the resampling is desired. Type = type 
        of Chebyshev points (defaults to 1)
        Outputs: the resampled locations as an Nrs x 3 vector of points. 
        w = quadrature weights at those nodes
        """
        Xrs = np.dot(cf.RSMat(Nrs,self._N,typetarg,typsrc=1),Xarg);
        return Xrs;
    
    def newWeights(self,Nrs,typetarg=1):
        """
        New quadrature weights for a set of Chebyshev nodes
        Inputs: Nrs = number of new nodes, typetarg = type of new nodes
        Outputs: new weights wN
        """
        _, wN = cf.chebpts(Nrs,[0,self._L],typetarg);
        return wN;

    # Some getter methods
    def getUniLocs(self,Xarg,Nuni):
        """ 
        Get the locations of Nuni uniformly distributed nodes on the fiber. 
        Xarg = the fiber coordinates (self._N x 3 vector), Nuni = number of 
        uniform resampling points desired.
        """
        Rsuni = cf.RSMat(Nuni,self._N,typtarg='u',typsrc=1);
        Xuni = np.dot(Rsuni,Xarg);
        return Xuni;
    
    def getLocs(self):
        return self._X;
        
    def getw(self):
        return self._w;    

    def getNLargs(self,tint=2,addlam=1):
        """
        Get the arguments necessary for the non-local solves. 
        Inputs: tint = temporal integrator (2 for Crank-Nicolson, 
        1 for backward Euler, -1 for forward Euler), that tells
        the method whether to do LMM combinations for Xs and X)
        addlam = whether to do the LMM for lambda (1 to do the LMM, 
        0 otherwise)
        """
        addxs = float(np.abs(tint)-1)/2;
        Xsarg = np.reshape((1+addxs)*self._Xs - \
            addxs*self._Xsprev,(self._N,3));
        Xarg = np.reshape((1+addxs)*self._X - \
            addxs*self._Xprev,(self._N,3));
        fEarg = np.reshape((1+addxs)*self.calcfE(self._X) - \
            addxs*self.calcfE(self._Xprev),(self._N,3));
        lamarg = np.reshape((1+addlam)*self._lambdas- \
            addlam*self._lambdaprev,(self._N,3));
        return Xarg, Xsarg, fEarg, lamarg;
    
    
    def writeLocs(self,of):#outFile,wa='a'):
        """
        Write the locations to a file 
        with name outFile. wa = write or append
        """
        #of = open(outFile,wa);
        for i in range(self._N):
            for j in range(3):
                of.write('%14.15f     ' % self._X[3*i+j]);
            of.write('\n');
        #of.close();
        
    @staticmethod
    def cart2sph(x,y,z):
        """
        Method to convert point(s) (x,y,z) to 
        spherical coordinates (azimuth, elevation, r)
        """
        azimuth = np.arctan2(y,x);
        elevation = np.arctan2(z,np.sqrt(x**2 + y**2));
        r = np.sqrt(x**2 + y**2 + z**2);
        azimuth[(np.abs(np.abs(elevation)-np.pi/2) < 1e-12)] = 0;
        return azimuth, elevation, r    

    @staticmethod    
    def deAliasInt(Rup,Rdwn,Dinvup,f1,f2,N):
        """
        Method to dealias a product. 
        What this code does is: given input vectors of size N
        f1 and f2, it upsamples f1 and f2 to a 2N grid, does the 
        product on the 2N grid, then integrates (using the Dinvup)
        on the 2N grid. 
        It then downsamples the result to an N grid and returns.
        """
        temp = np.reshape(np.dot(Rup,f1),(2*N,1)) \
            *np.dot(Rup,f2);    # reshape just to get multiplication to work
        temp = np.dot(Rdwn,np.dot(Dinvup,temp));
        return temp;
    
