import numba as nb
import numpy as np
from math import pi, erfc, sqrt, exp

"""
These methods are sped up with numba. As a general comment, 
my experience is that C++ and PyBind11 performs better than 
numba if the computation involves more than 1 for loop. For 
this reason, the only thing I am using Numba for right now
is the SBT kernel computation for a  single fiber and target. 
This will be eliminated in future versions of the code. 
"""
 
@nb.njit(nb.float64(nb.float64,nb.float64,nb.float64))
def F(r,xi,a):
    if (r < 1e-10): # Taylor series
        val = 1.0/(4*sqrt(pi)*xi*a)*(1-exp(-4*a**2*xi**2)+\
            4*sqrt(pi)*a*xi*erfc(2*a*xi));
        return val;
    if (r>2*a):
        f0=0.0;
        f1=(18*r**2*xi**2+3)/(64*sqrt(pi)*a*r**2*xi**3);
        f2=(2*xi**2*(2*a-r)*(4*a**2+4*a*r+9*r**2)-2*a-3*r)/\
            (128*sqrt(pi)*a*r**3*xi**3);
        f3=(-2*xi**2*(2*a+r)*(4*a**2-4*a*r+9*r**2)+2*a-3*r)/\
            (128*sqrt(pi)*a*r**3*xi**3);
        f4=(3-36*r**4*xi**4)/(128*a*r**3*xi**4);
        f5=(4*xi**4*(r-2*a)**2*(4*a**2+4*a*r+9*r**2)-3)/(256*a*r**3*xi**4);
        f6=(4*xi**4*(r+2*a)**2*(4*a**2-4*a*r+9*r**2)-3)/(256*a*r**3*xi**4);
    else:
        f0=-(r-2*a)**2*(4*a**2+4*a*r+9*r**2)/(32*a*r**3);
        f1=(18*r**2*xi**2+3)/(64*sqrt(pi)*a*r**2*xi**3);
        f2=(2*xi**2*(2*a-r)*(4*a**2+4*a*r+9*r**2)-2*a-3*r)/\
            (128*sqrt(pi)*a*r**3*xi**3);
        f3=(-2*xi**2*(2*a+r)*(4*a**2-4*a*r+9*r**2)+2*a-3*r)/\
            (128*sqrt(pi)*a*r**3*xi**3);
        f4=(3-36*r**4*xi**4)/(128*a*r**3*xi**4);
        f5=(4*xi**4*(r-2*a)**2*(4*a**2+4*a*r+9*r**2)-3)/\
            (256*a*r**3*xi**4);
        f6=(4*xi**4*(r+2*a)**2*(4*a**2-4*a*r+9*r**2)-3)/\
            (256*a*r**3*xi**4);
    val = f0+f1*exp(-r**2*xi**2)+f2*exp(-(r-2*a)**2*xi**2)+\
        f3*exp(-(r+2*a)**2*xi**2)+f4*erfc(r*xi)+f5*erfc((r-2*a)*xi)+\
        f6*erfc((r+2*a)*xi);
    return val;

@nb.njit(nb.float64(nb.float64,nb.float64,nb.float64))
def G(r,xi,a):
    if (r < 1e-10):
        return 0;
    if (r>2*a):
        g0=0;
        g1=(6*r**2*xi**2-3)/(32*sqrt(pi)*a*r**2*xi**3);
        g2=(-2*xi**2*(r-2*a)**2*(2*a+3*r)+2*a+3*r)/\
            (64*sqrt(pi)*a*r**3*xi**3);
        g3=(2*xi**2*(r+2*a)**2*(2*a-3*r)-2*a+3*r)/\
            (64*sqrt(pi)*a*r**3*xi**3);
        g4=-3*(4*r**4*xi**4+1)/(64*a*r**3*xi**4);
        g5=(3-4*xi**4*(2*a-r)**3*(2*a+3*r))/(128*a*r**3*xi**4);
        g6=(3-4*xi**4*(2*a-3*r)*(2*a+r)**3)/(128*a*r**3*xi**4);
    else:
        g0=(2*a-r)**3*(2*a+3*r)/(16*a*r**3);
        g1=(6*r**2*xi**2-3)/(32*sqrt(pi)*a*r**2*xi**3);
        g2=(-2*xi**2*(r-2*a)**2*(2*a+3*r)+2*a+3*r)/\
            (64*sqrt(pi)*a*r**3*xi**3);
        g3=(2*xi**2*(r+2*a)**2*(2*a-3*r)-2*a+3*r)/\
            (64*sqrt(pi)*a*r**3*xi**3);
        g4=-3*(4*r**4*xi**4+1)/(64*a*r**3*xi**4);
        g5=(3-4*xi**4*(2*a-r)**3*(2*a+3*r))/(128*a*r**3*xi**4);
        g6=(3-4*xi**4*(2*a-3*r)*(2*a+r)**3)/(128*a*r**3*xi**4);
    val = g0+g1*exp(-r**2*xi**2)+g2*exp(-(r-2*a)**2*xi**2)+\
        g3*exp(-(r+2*a)**2*xi**2)+g4*erfc(r*xi)+g5*erfc((r-2*a)*xi)+\
        g6*erfc((r+2*a)*xi);
    return val;

@nb.njit(nb.float64[:](nb.float64[:],nb.float64,nb.float64[:]))
def calcShifted(rvec,g,Lens):
    # Mod r vec so it's on [-L/2,L/2]
    s1 = round(rvec[1]/Lens[1]);
    # Shift in y' direction and z direction
    rvec[0]-= g*Lens[1]*s1;
    rvec[1]-= Lens[1]*s1;
    rvec[2]-= Lens[2]*round(rvec[2]/Lens[2]);
    # Shift in x direction
    rvec[0]-= Lens[0]*round(rvec[0]/Lens[0]);
    return rvec;

@nb.njit(nb.float64[:,:](nb.int64,nb.int64[:,:],nb.float64[:,:],nb.float64[:,:],\
            nb.float64,nb.float64,nb.float64,nb.float64[:],nb.float64,nb.float64))
def RPYNearPairs(Npts,pairpts,ptsxyz,forces,mu,xi,a,Lens,g,rcut):
    """
    Evaluate the near field RPY kernel for pairs of points.
    Input: the vector rvec to evaluate the kernel at
    Output: value of kernel.
    """
    # Self terms
    F0 = F(0.0,xi,a);
    outFront = 1.0/(6.0*pi*mu*a);
    velNear = outFront*F0*forces;
    Npairs, _ = pairpts.shape;
    # Others
    for iPair in range(Npairs):
        iPt = pairpts[iPair,0];
        jPt = pairpts[iPair,1];
        rvec = ptsxyz[iPt,:]-ptsxyz[jPt,:];
        rvec = calcShifted(rvec,g,Lens);
        r = np.linalg.norm(rvec);
        if (r < rcut):
            co1 = F(r,xi,a)*outFront;
            co2 = G(r,xi,a)*outFront;
            rhat=rvec/r;
            velNear[iPt,:]+=co1*forces[jPt,:]+np.sum(rhat*forces[jPt,:])*(co2-co1)*rhat;
            velNear[jPt,:]+=co1*forces[iPt,:]+np.sum(rhat*forces[iPt,:])*(co2-co1)*rhat;
    return velNear

# numba version
@nb.njit(nb.float64[:](nb.float64[:],nb.int64,nb.float64[:,:],nb.float64[:,:],nb.float64,\
    nb.float64,nb.float64,nb.float64[:],nb.float64[:],nb.float64[:]))
def SBTKSpl(Xtarg,Npts,Xarg,forceDs,mu,epsilon,L,w1,w3,w5):
    SBTvel=np.zeros(3);
    for jPt in range(Npts):
        # Subtract free space kernel (no periodicity)
        rvec=Xtarg-Xarg[jPt,:];
        r=np.linalg.norm(rvec);
        rdotf = np.sum(rvec*forceDs[jPt,:]);
        u1 = forceDs[jPt,:]/r;
        u3 = (rvec*rdotf+((epsilon*L)**2)*forceDs[jPt,:])/r**3;
        u5 = -((epsilon*L)**2)*3.0*rvec*rdotf/r**5;
        SBTvel+= u1*w1[jPt]+u3*w3[jPt]+u5*w5[jPt];
    # Divide by viscosity
    return 1.0/(8.0*np.pi*mu)*SBTvel;


# Next 2 methods are utilities for RPY kernel
@nb.njit(nb.float64(nb.float64,nb.float64,nb.boolean))
def FT(r,a,sbt):
    if (r>2*a or sbt): # RPY far or slender body
        val = (2*a**2 + 3*r**2)/(24*np.pi*r**3);
    else:
        val = (32*a - 9*r)/(192*a**2*np.pi);
    return val;

@nb.njit(nb.float64(nb.float64,nb.float64,nb.boolean))
def GT(r,a,sbt):
    if (r>2*a or sbt): # RPY far or slender body
        val = (-2*a**2 + 3*r**2)/(12*np.pi*r**3);
    else:
        val = (16*a - 3*r)/(96*a**2*np.pi);
    return val;

@nb.njit(nb.float64[:,:](nb.int64,nb.float64[:,:],nb.int64,nb.float64[:,:],\
        nb.float64[:,:],nb.float64,nb.float64,nb.boolean))
def RPYSBTK(Ntarg,Xtarg,Nsrc,Xfib,forces,mu,a,sbt):
    """
    The total RPY kernel for a given input vector rvec.
    Another input is whether the kernel is sbt (for sbt=1)
    or RPY (for sbt=0). The kernels change close to the fiber.
    Output is the matrix of the kernel.
    """
    utot=np.zeros((Ntarg,3));
    oneOvermu = 1.0/mu;
    for iTarg in range(Ntarg):
        for iSrc in range(Nsrc):
            rvec = Xtarg[iTarg,:]-Xfib[iSrc,:];
            r = np.linalg.norm(rvec);
            rhat = rvec/r;
            rhat[np.isnan(rhat)]=0;
            rdotf = np.sum(rhat*forces[iSrc,:]);
            fval = FT(r,a,sbt);
            gval = GT(r,a,sbt);
            utot[iTarg,:]+= oneOvermu*(fval*forces[iSrc,:]+rdotf*(gval-fval)*rhat);
    return utot;

## C++ WITH PYTHON PERFORMS MUCH BETTER THAN NUMBA HERE
@nb.njit(nb.types.Tuple((nb.int64[:],nb.int64[:],nb.int64[:],nb.float64[:,:]))\
        (nb.int64,nb.int64,nb.int64,nb.int64[:],nb.int64[:,:],nb.float64[:,:],\
         nb.float64[:,:],nb.float64,nb.float64[:],nb.float64,nb.float64))
def testList(Nfib,NperFib,NunifperFib,numNeighbors,neighbors2DArray,\
             XnonLoc,Xunif,g,Lens,q1cut,q2cut):
    targets = np.empty(0,dtype=nb.int64);
    fibers = np.empty(0,dtype=nb.int64);
    methods = np.empty(0,dtype=nb.int64);
    shifts = np.empty(0);
    for iPt in range(Nfib*NperFib): # loop over points
        iFib = iPt//NperFib; # integer division to get the fiber point i is on
        pNeighbors = neighbors2DArray[iPt,:numNeighbors[iPt]]; # neighbors of point i
        fNeighbors = pNeighbors//NunifperFib; # fibers that are neighbors w point i
        oFibs = np.unique(fNeighbors[fNeighbors!=iFib]); # exclude itself
        for neFib in oFibs: # loop over neighboring fibers
            oPts = pNeighbors[fNeighbors==neFib]; # close points on those fibers
            qtype = 0;
            rmin = q1cut;
            for iuPt in range(len(oPts)):
                rvec = XnonLoc[iPt,:]-Xunif[oPts[iuPt],:];
                rvec = calcShifted(rvec,g,Lens);
                nr = np.linalg.norm(rvec);
                if (nr < q2cut):
                    qtype = 2;
                    shift = XnonLoc[iPt,:]-Xunif[oPts[iuPt],:]-rvec;
                    break;
                elif (nr < rmin):
                    rmin = nr;
                    qtype = 1;
                    shift = XnonLoc[iPt,:]-Xunif[oPts[iuPt],:]-rvec;
            if (qtype > 0): # add to the list of corrections
                targets = np.concatenate((targets,np.array([iPt])));
                fibers = np.concatenate((fibers,np.array([neFib])));
                methods = np.concatenate((methods,np.array([qtype])));
                shifts = np.concatenate((shifts,shift));
    return targets, fibers, methods, np.reshape(shifts,(len(targets),3));

