#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include "SpecQuadUtils.cpp"

//This is a pybind11 module of C++ functions that are being used
//for non-local velocity calculations in the SBT code.

//The RPY near field can be written as M_near = F(r,xi,a)*I+G(r,xi,a)*(I-RR)
//The next 2 functions are those F and G functions.
#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

double Fnear(double r, double xi, double a){
	double sqrtpi = sqrt(M_PI);
	if (r < 1e-10){ // Taylor series
        double val = 1.0/(4.0*sqrtpi*xi*a)*(1.0-exp(-4.0*a*a*xi*xi)+
            4*sqrtpi*a*xi*erfc(2*a*xi));
        return val;
	}
	double f0, f1, f2, f3, f4, f5, f6;
    if (r>2*a){
        f0=0.0;
        f1=(18.0*r*r*xi*xi+3.0)/(64.0*sqrtpi*a*r*r*xi*xi*xi);
        f2=(2.0*xi*xi*(2.0*a-r)*(4.0*a*a+4.0*a*r+9.0*r*r)-2*a-3*r)/ 
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f3=(-2.0*xi*xi*(2.0*a+r)*(4.0*a*a-4.0*a*r+9.0*r*r)+2.0*a-3.0*r)/
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f4=(3.0-36.0*r*r*r*r*xi*xi*xi*xi)/(128.0*a*r*r*r*xi*xi*xi*xi);
        f5=(4.0*xi*xi*xi*xi*(r-2.0*a)*(r-2.0*a)*(4.0*a*a+4.0*a*r+9.0*r*r)-3.0)/
			(256.0*a*r*r*r*xi*xi*xi*xi);
        f6=(4.0*xi*xi*xi*xi*(r+2.0*a)*(r+2.0*a)*(4.0*a*a-4.0*a*r+9.0*r*r)-3.0)/
			(256.0*a*r*r*r*xi*xi*xi*xi);
    } else{
        f0=-(r-2.0*a)*(r-2.0*a)*(4.0*a*a+4.0*a*r+9.0*r*r)/(32.0*a*r*r*r);
        f1=(18.0*r*r*xi*xi+3.0)/(64.0*sqrtpi*a*r*r*xi*xi*xi);
        f2=(2.0*xi*xi*(2.0*a-r)*(4.0*a*a+4.0*a*r+9.0*r*r)-2.0*a-3.0*r)/
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f3=(-2.0*xi*xi*(2.0*a+r)*(4.0*a*a-4.0*a*r+9.0*r*r)+2.0*a-3.0*r)/
            (128.0*sqrtpi*a*r*r*r*xi*xi*xi);
        f4=(3.0-36.0*r*r*r*r*xi*xi*xi*xi)/(128.0*a*r*r*r*xi*xi*xi*xi);
        f5=(4.0*xi*xi*xi*xi*(r-2.0*a)*(r-2.0*a)*(4.0*a*a+4.0*a*r+9.0*r*r)-3.0)/
            (256.0*a*r*r*r*xi*xi*xi*xi);
        f6=(4.0*xi*xi*xi*xi*(r+2.0*a)*(r+2.0*a)*(4.0*a*a-4.0*a*r+9.0*r*r)-3.0)/
            (256.0*a*r*r*r*xi*xi*xi*xi);
	}
    double val = f0+f1*exp(-r*r*xi*xi)+f2*exp(-(r-2.0*a)*(r-2.0*a)*xi*xi)+
        f3*exp(-(r+2.0*a)*(r+2.0*a)*xi*xi)+f4*erfc(r*xi)+f5*erfc((r-2*a)*xi)+
        f6*erfc((r+2*a)*xi);
    return val;
}

double Gnear(double r, double xi, double a){
	double sqrtpi = sqrt(M_PI);
	if (r < 1e-10){
        return 0;
	}
	double g0, g1, g2, g3, g4, g5, g6;
    if (r>2*a){
        g0=0;
        g1=(6.0*r*r*xi*xi-3.0)/(32.0*sqrtpi*a*r*r*xi*xi*xi);
        g2=(-2.0*xi*xi*(r-2.0*a)*(r-2.0*a)*(2.0*a+3.0*r)+2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g3=(2.0*xi*xi*(r+2.0*a)*(r+2.0*a)*(2.0*a-3.0*r)-2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g4=-3.0*(4.0*r*r*r*r*xi*xi*xi*xi+1.0)/(64.0*a*r*r*r*xi*xi*xi*xi);
        g5=(3.0-4.0*xi*xi*xi*xi*(2.0*a-r)*(2.0*a-r)*(2.0*a-r)*(2.0*a+3.0*r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
        g6=(3.0-4.0*xi*xi*xi*xi*(2.0*a-3.0*r)*(2.0*a+r)*(2.0*a+r)*(2.0*a+r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
    } else {
        g0=(2.0*a-r)*(2.0*a-r)*(2.0*a-r)*(2.0*a+3.0*r)/(16.0*a*r*r*r);
        g1=(6.0*r*r*xi*xi-3.0)/(32.0*sqrtpi*a*r*r*xi*xi*xi);
        g2=(-2.0*xi*xi*(r-2.0*a)*(r-2.0*a)*(2.0*a+3.0*r)+2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g3=(2.0*xi*xi*(r+2.0*a)*(r+2.0*a)*(2.0*a-3.0*r)-2.0*a+3.0*r)/
            (64.0*sqrtpi*a*r*r*r*xi*xi*xi);
        g4=-3.0*(4.0*r*r*r*r*xi*xi*xi*xi+1.0)/(64.0*a*r*r*r*xi*xi*xi*xi);
        g5=(3.0-4.0*xi*xi*xi*xi*(2.0*a-r)*(2.0*a-r)*(2.0*a-r)*(2.0*a+3.0*r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
        g6=(3.0-4.0*xi*xi*xi*xi*(2.0*a-3.0*r)*(2.0*a+r)*(2.0*a+r)*(2.0*a+r))/
			(128.0*a*r*r*r*xi*xi*xi*xi);
	}
    double val = g0+g1*exp(-r*r*xi*xi)+g2*exp(-(r-2.0*a)*(r-2.0*a)*xi*xi)+
        g3*exp(-(r+2.0*a)*(r+2.0*a)*xi*xi)+g4*erfc(r*xi)+g5*erfc((r-2.0*a)*xi)+
        g6*erfc((r+2.0*a)*xi);
    return val;
}


//The RPY and SBT kernels can both be written as M = Ft(r,xi,a)*I+Gt(r,xi,a)*(I-RR)
//The next 2 functions are those Ft and Gt functions.
//The kernels are different when r < 2a, and this code accounts for that by
//passing in an integer "SBT" which is 1 for the SBT kernel and 0 for the RPY kernel.
double Ftot(double r, double a, int SBT){
	if (r > 2*a || SBT){
        return (2.0*a*a+3.0*r*r)/(24*M_PI*r*r*r);
    }
    return (32.0*a-9.0*r)/(192.0*a*a*M_PI);
}

double Gtot(double r, double a, int SBT){
	if (r > 2*a || SBT){
        return (-2.0*a*a+3.0*r*r)/(12*M_PI*r*r*r);
    }
    return (16.0*a-3.0*r)/(96.0*a*a*M_PI);
}

// Some standard vector methods that are helpful in the
// RPY calculations
double dot(std::vector<double> a, std::vector <double> b){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

double normalize(std::vector<double> &rvec){
    double r = sqrt(dot(rvec,rvec));
    if (r > 1e-10){
        double rinv = 1.0/r;
        rvec[0]*=rinv;
        rvec[1]*=rinv;
        rvec[2]*=rinv;
    }
    return r;
}

// The near field kernel, multiplies M_near * f
// Inputs: 3 vector r (not normalized), 3 vector force,
// mu = viscosity, xi = Ewald parameter, and a = blob radius
// Returns the velocity M_near*f
std::vector<double> RPYNKer(std::vector<double> &rvec, std::vector<double> force,
                            double mu, double xi, double a){
    double r = normalize(rvec);
    double rdotf = dot(rvec,force);
    std::vector <double> unear(3);
    double F = Fnear(r, xi, a);
    double G = Gnear(r, xi, a);
    double co2 = rdotf*G-rdotf*F;
    double outFront = 1.0/(6.0*M_PI*mu*a);
    unear[0] = outFront*(F*force[0]+co2*rvec[0]);
    unear[1] = outFront*(F*force[1]+co2*rvec[1]);
    unear[2] = outFront*(F*force[2]+co2*rvec[2]);
    return unear;
}

// Compute shifted coordinates of a vector rvec. The goal is to shift a vector
// rvec periodically until it's on [-Lx/2,Lx/2] x [-Ly/2, Ly/2] x [-Lz/2, Lz/2].
// This must be done carefully, however, since the periodicity is not standard
// (slanted grid).
// Inputs: rvec = 3 vector to shift, g = slantedness of grid, Lx, Ly, Lz = periodic
// lengths in each direction.
// Output: the shifted vector rvec
std::vector <double> calcShifted(std::vector<double> rvec, double g, double Lx, double Ly, double Lz){
    double s1 = round(rvec[1]/Ly);
    // Shift in (y',z') directions
    rvec[0]-= g*Ly*s1;
    rvec[1]-= Ly*s1;
    rvec[2]-= Lz*round(rvec[2]/Lz);
    // Shift in x direction
    rvec[0]-= Lx*round(rvec[0]/Lx);
    return rvec;
}


// The near field kernel, multiplies M_near * f for paris of points
// Inputs: number of points N , number of pairs Npairs, iPts = one point in the pair
// jPts = the other point in the pair, xpts, ypts, zpts = the x, y, and z coordinates
// of the points in Cartesian coordinates. forcesX, forcesY, forcesZ = values of the forces
// mu = viscosity, a = hydro radius, Lx, Ly, Lz = periodic lenghts, g = strain in coordinate
// system, and rcut = truncation distance for the near field. 
std::vector<double> RPYNKerPairs(int N, int Npairs,std::vector<int> iPts, std::vector<int> jPts,
             std::vector<double> xpts, std::vector<double> ypts,
             std::vector<double> zpts, std::vector<double> forcesX,std::vector<double> forcesY,
             std::vector<double> forcesZ,double mu, double xi, double a,
             double Lx, double Ly, double Lz, double g, double rcut){
    std::vector <double> unear(N*3);
    // Self
    double outFront = 1.0/(6.0*M_PI*mu*a);
    double F0 = Fnear(0.0,xi,a);
    for (int iPt=0; iPt < N; iPt++){
        unear[3*iPt]=outFront*F0*forcesX[iPt];
        unear[3*iPt+1]=outFront*F0*forcesY[iPt];
        unear[3*iPt+2]=outFront*F0*forcesZ[iPt];
    }
    // Pairs
    std::vector<double> rvec(3);
    int iPt, jPt;
    double rdotfj, rdotfi, F, G, co2i, co2j, r;
    for (int iPair=0; iPair < Npairs; iPair++){
        iPt = iPts[iPair];
        jPt = jPts[iPair];
        rvec[0]=xpts[iPt]-xpts[jPt];
        rvec[1]=ypts[iPt]-ypts[jPt];
        rvec[2]=zpts[iPt]-zpts[jPt];
        rvec = calcShifted(rvec,g,Lx,Ly,Lz);
        r = normalize(rvec);
        if (r < rcut){ // only do the computation below truncation
            rdotfj = rvec[0]*forcesX[jPt]+rvec[1]*forcesY[jPt]+rvec[2]*forcesZ[jPt];
            rdotfi = rvec[0]*forcesX[iPt]+rvec[1]*forcesY[iPt]+rvec[2]*forcesZ[iPt];
            F = Fnear(r, xi, a);
            G = Gnear(r, xi, a);
            co2i = rdotfj*G-rdotfj*F;
            co2j = rdotfi*G-rdotfi*F;
            unear[3*iPt]+=outFront*(F*forcesX[jPt]+co2i*rvec[0]);
            unear[3*iPt+1]+=outFront*(F*forcesY[jPt]+co2i*rvec[1]);
            unear[3*iPt+2]+=outFront*(F*forcesZ[jPt]+co2i*rvec[2]);
            unear[3*jPt]+=outFront*(F*forcesX[iPt]+co2j*rvec[0]);
            unear[3*jPt+1]+=outFront*(F*forcesY[iPt]+co2j*rvec[1]);
            unear[3*jPt+2]+=outFront*(F*forcesZ[iPt]+co2j*rvec[2]);
        } // end doing the computation
    } // end looping over pairs
    return unear;
}

// Multiply M_RPY*f
// Inputs: 3 vector r (not normalized), 3 vector force,
// mu = viscosity, xi = Ewald parameter, a = blob radius,
// SBT = 1 for the SBT kernel, 0 for the RPY kernel
// Outputs: the velocity M_RPY*f
std::vector<double> RPYTot(std::vector<double> rvec, std::vector<double> force,
                            double mu, double a, int SBT){
    double r = normalize(rvec);
    double rdotf = dot(rvec,force);
    std::vector <double> utot(3);
    double F = Ftot(r, a, SBT);
    double G = Gtot(r, a, SBT);
    double co2 = rdotf*G-rdotf*F;
    double outFront = 1.0/mu;
    utot[0] = outFront*(F*force[0]+co2*rvec[0]);
    utot[1] = outFront*(F*force[1]+co2*rvec[1]);
    utot[2] = outFront*(F*force[2]+co2*rvec[2]);
    return utot;
}

// Compute the integral of the RPY/SBT kernel along a fiber with respect to some
// targets. Inputs: Ntarg = number of targets, xtarg, ytarg, ztarg = (x,y,z) coordinates
// of targets, respectively. Nsrc = number of sources (points along the fiber),
// xsrc, ysrc, zsrc = points along the fiber (x,y,z, respectively), fx, fy, fz = forces
// (force densities * weights) along the fiber, mu = fluid viscosity, a = radius of the blobs,
// SBT = 1 for SBT kernel, 0 for RPY kernel
// Return: velocity at the targets as an array (stacked by point)
std::vector<double> RPYSBTKernel(int Ntarg, const std::vector<double> &xtarg, const std::vector<double> &ytarg,
                            const std::vector<double> &ztarg, int Nsrc, const std::vector<double> &xsrc,
                            const std::vector<double> &ysrc, const std::vector<double> &zsrc,
                            const std::vector<double> &fx, const std::vector<double> &fy, 
                            const std::vector<double> &fz, double mu, double a, int SBT){
    std::vector<double> rvec(3);
    std::vector<double> forces(3);
    std::vector<double> utargs(Ntarg*3,0.0);
    std::vector<double> uadd(3);
    for (int iTarg=0; iTarg < Ntarg; iTarg++){
        for (int iSrc=0; iSrc < Nsrc; iSrc++){
            rvec[0]=xtarg[iTarg]-xsrc[iSrc];
            rvec[1]=ytarg[iTarg]-ysrc[iSrc];
            rvec[2]=ztarg[iTarg]-zsrc[iSrc];
            forces[0]=fx[iSrc];
            forces[1]=fy[iSrc];
            forces[2]=fz[iSrc];
            uadd = RPYTot(rvec,forces,mu,a,SBT);
            utargs[iTarg*3]+=uadd[0];
            utargs[iTarg*3+1]+=uadd[1];
            utargs[iTarg*3+2]+=uadd[2];
        }
    }
    return utargs;
}

std::vector<double> OneRPYSBTKernel(double xtarg, double ytarg,double ztarg,int Nsrc,
                            const std::vector<double> &xsrc,const std::vector<double> &ysrc,
                            const std::vector<double> &zsrc,const std::vector<double> &fx,
                            const std::vector<double> &fy,
                            const std::vector<double> &fz, double mu, double a, int SBT){
    std::vector<double> rvec(3);
    std::vector<double> forces(3);
    std::vector<double> utargs(3,0.0);
    std::vector<double> uadd(3);
    for (int iSrc=0; iSrc < Nsrc; iSrc++){
        rvec[0]=xtarg-xsrc[iSrc];
        rvec[1]=ytarg-ysrc[iSrc];
        rvec[2]=ztarg-zsrc[iSrc];
        forces[0]=fx[iSrc];
        forces[1]=fy[iSrc];
        forces[2]=fz[iSrc];
        uadd = RPYTot(rvec,forces,mu,a,SBT);
        for (int d=0; d<3; d++){
            utargs[d]+=uadd[d];
        }
    }
    return utargs;
}

template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n + 1;

	std::vector<T> vec(first, last);
	return vec;
}

std::vector<double> SBTKernelSplit(std::vector<double> targpt, int N, std::vector<double> xsrc,
                            std::vector<double> ysrc, std::vector<double> zsrc,
                            std::vector<double> fx, std::vector<double> fy, std::vector<double> fz,
                            double mu, double epsilon, double L, std::vector <double> w1,
                            std::vector <double> w3, std::vector <double> w5){
    std::vector<double> rvec(3);
    std::vector<double> forces(3);
    std::vector<double> utarg(3);
    double r, rdotf, u1, u3, u5;
    double outFront = 1.0/(8.0*M_PI*mu);
    for (int iPt=0; iPt < N; iPt++){
        rvec[0]=targpt[0]-xsrc[iPt];
        rvec[1]=targpt[1]-ysrc[iPt];
        rvec[2]=targpt[2]-zsrc[iPt];
        forces[0]=fx[iPt];
        forces[1]=fy[iPt];
        forces[2]=fz[iPt];
        rdotf = dot(rvec,forces);
        r = normalize(rvec);
        for (int d=0; d < 3; d++){
            u1 = forces[d]/r;
            u3 = (r*rvec[d]*rdotf+(epsilon*L)*(epsilon*L)*forces[d])/(r*r*r);
            u5 = -3.0*(epsilon*L)*(epsilon*L)*rvec[d]*r*rdotf/pow(r,5);
            utarg[d]+=outFront*(u1*w1[iPt]+u3*w3[iPt]+u5*w5[iPt]);
        }
    }
    return utarg;
}

// This method currently does all the special quad except for the targets which need 2 panels. 
// Those are assumed to be such a small number that they can be done in Python. 
std::tuple < std::vector <double>, std::vector<int>, std::vector <std::complex <double>>, std::vector <double> >
            RPYSBTAllFibers(int Nfib, int Ntarg, int Ncors, const std::vector <int> &numTargbyFib,
            const std::vector <int> &cumTargbyFib, const std::vector <int> sortedTargs,
            const std::vector<double> &xtarg,const std::vector<double> &ytarg, 
            const std::vector<double> &ztarg, int NperFib, const std::vector<double> &xAllFibs, 
            const std::vector<double> &yAllFibs, const std::vector<double> &zAllFibs, 
            const std::vector<double> &fxall, const std::vector<double> &fyall, 
            const std::vector<double> &fzall, const std::vector<double> &xUpFibs, 
            const std::vector<double> &yUpFibs, const std::vector<double> &zUpFibs, 
            const std::vector<double> &fxUp, const std::vector<double> &fyUp, 
            const std::vector<double> &fzUp,const std::vector<double> &forceDxUp, const std::vector<double> &forceDyUp, 
            const std::vector<double> &forceDzUp,int nUpsample, const std::vector<int> sortedMethods,
            int nCoeffs, const std::vector<double> &xHatAllFibs, const std::vector<double> &yHatAllFibs,
            const std::vector<double> &zHatAllFibs, const std::vector<double> &dxHatAllFibs, 
            const std::vector<double> &dyHatAllFibs,const std::vector<double> &dzHatAllFibs,
            const std::vector<double> &CLxHatAllFibs,const std::vector<double> &CLyHatAllFibs,
            std::vector<double> &CLzHatAllFibs,const std::vector<double> tnodes, double rho_crit, double mu, double a,
            double dstarCL, double dstarInterp, double dstar2panels, double Lf, double epsilon){
    std::vector <double> allus(3*Ntarg,0.0);
    std::vector <int> sqneeded(Ncors,0);
    std::vector <std::complex <double>> allRoots(Ncors); 
    std::vector <double> dstars(Ncors);
    //#pragma omp parallel for reduction(vec_double_plus : allus)
    for (int iFib=0; iFib < Nfib; iFib++){
        int nT = numTargbyFib[iFib];
        int targStart = cumTargbyFib[iFib];
        // Fill arrays for points and forces for fiber
        std::vector <double> fibptsX = slice(xAllFibs,iFib*NperFib,(iFib+1)*NperFib);
        std::vector <double> fibptsY = slice(yAllFibs,iFib*NperFib,(iFib+1)*NperFib);
        std::vector <double> fibptsZ = slice(zAllFibs,iFib*NperFib,(iFib+1)*NperFib);
        std::vector <double> fibforceX = slice(fxall,iFib*NperFib,(iFib+1)*NperFib);
        std::vector <double> fibforceY = slice(fyall,iFib*NperFib,(iFib+1)*NperFib);
        std::vector <double> fibforceZ = slice(fzall,iFib*NperFib,(iFib+1)*NperFib);
        std::vector <double> fibUpforceDX = slice(forceDxUp,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> fibUpforceDY = slice(forceDyUp,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> fibUpforceDZ = slice(forceDzUp,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> fibUpX = slice(xUpFibs,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> fibUpY = slice(yUpFibs,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> fibUpZ = slice(zUpFibs,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> forceUpX = slice(fxUp,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> forceUpY = slice(fyUp,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> forceUpZ = slice(fzUp,iFib*nUpsample,(iFib+1)*nUpsample);
        std::vector <double> fibhatX = slice(xHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibhatY = slice(yHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibhatZ = slice(zHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibhatdX = slice(dxHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibhatdY = slice(dyHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibhatdZ = slice(dzHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibCLvhatX = slice(CLxHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibCLvhatY = slice(CLyHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        std::vector <double> fibCLvhatZ = slice(CLzHatAllFibs,iFib*nCoeffs,(iFib+1)*nCoeffs);
        for (int iT=0; iT < nT; iT++){
            std::vector <double> uRPY(3,0.0);
            std::vector <double> uSBT(3,0.0);
            std::vector <double> CLpart(3,0.0);
            double CLwt = 0.0;
            double SBTwt = 1.0;
            // Subtract RPY kernel
            uRPY = OneRPYSBTKernel(xtarg[targStart+iT],ytarg[targStart+iT],
                ztarg[targStart+iT],NperFib,fibptsX,fibptsY,fibptsZ,fibforceX,
                fibforceY,fibforceZ,mu,a,0);
            if (sortedMethods[targStart+iT]==1){
                // Correct with upsampling (SBT kernel)
                uSBT = OneRPYSBTKernel(xtarg[targStart+iT],ytarg[targStart+iT],
                    ztarg[targStart+iT],nUpsample,fibUpX,fibUpY,fibUpZ,forceUpX,
                    forceUpY,forceUpZ,mu,a,1);
            } else if (sortedMethods[targStart+iT]==2){
                // Special quadrature
                // Calculate root
                std::complex<double> tinit = Onerootfinder_initial_guess(tnodes,fibUpX,fibUpY,fibUpZ,nUpsample,
                    xtarg[targStart+iT],ytarg[targStart+iT],ztarg[targStart+iT]);
                if (bernstein_radius(tinit) > 1.5*rho_crit){
                    // Direct quad if initial guess is too far
                    uSBT = OneRPYSBTKernel(xtarg[targStart+iT],ytarg[targStart+iT],
                        ztarg[targStart+iT],nUpsample,fibUpX,fibUpY,fibUpZ,forceUpX,
                        forceUpY,forceUpZ,mu,a,1);
                } else{
                    // Compute root
                    std::complex <double> troot = tinit;
                    int converged = Onerootfinder(fibhatX,fibhatY,fibhatZ,fibhatdX,fibhatdY,fibhatdZ,
                        nCoeffs,xtarg[targStart+iT],ytarg[targStart+iT],ztarg[targStart+iT],tinit,troot);
                    if (converged){
                        sqneeded[targStart+iT] = 1;
                        allRoots[targStart+iT] = troot;
                        double tapprox = real(troot);
                        if (tapprox < -1.0){
                            tapprox = -1.0;
                        } else if (tapprox > 1.0){
                            tapprox = 1.0;
                        }
                        // Compute the closest point on the fiber
                        double dxFib = eval_cheb(fibhatX,tapprox,nCoeffs)-xtarg[targStart+iT];
                        double dyFib = eval_cheb(fibhatY,tapprox,nCoeffs)-ytarg[targStart+iT];
                        double dzFib = eval_cheb(fibhatZ,tapprox,nCoeffs)-ztarg[targStart+iT];
                        double dstar = sqrt(dxFib*dxFib+dyFib*dyFib+dzFib*dzFib);
                        dstars[targStart+iT] = dstar;
                        CLwt = std::min((dstarInterp-dstar)/(dstarInterp-dstarCL),1.0); // takes care of very close ones
                        CLwt = std::max(CLwt,0.0); // far ones
                        SBTwt = 1.0-CLwt;
                        if (dstar < dstarInterp){
                            // Centerline velocity combo
                            CLpart[0] =  eval_cheb(fibCLvhatX,tapprox,nCoeffs);
                            CLpart[1] =  eval_cheb(fibCLvhatY,tapprox,nCoeffs);
                            CLpart[2] =  eval_cheb(fibCLvhatZ,tapprox,nCoeffs);
                            if (dstar < dstarCL){ //no need to continue with special quad
                                sqneeded[targStart+iT] = 0; 
                            }
                        }
                        if (bernstein_radius(troot) > rho_crit){
                            // Direct quad if possible to 3 digits
                            uSBT = OneRPYSBTKernel(xtarg[targStart+iT],ytarg[targStart+iT],
                                ztarg[targStart+iT],nUpsample,fibUpX,fibUpY,fibUpZ,forceUpX,
                                forceUpY,forceUpZ,mu,a,1);
                            sqneeded[targStart+iT] = 0;
                        }
                        if (sqneeded[targStart+iT] > 0 && dstar > dstar2panels){
                            // Compute the weights for 1 panel direct
                            std::vector <double> w1(nUpsample,0.0);
                            std::vector <double> w3(nUpsample,0.0);
                            std::vector <double> w5(nUpsample,0.0);
                            specialWeights(nUpsample,tnodes,troot,w1,w3,w5,Lf);
                            std::vector <double> target(3);
                            target[0] = xtarg[targStart+iT];
                            target[1] = ytarg[targStart+iT];
                            target[2] = ztarg[targStart+iT];
                            uSBT = SBTKernelSplit(target,nUpsample,fibUpX,fibUpY,fibUpZ,fibUpforceDX, 
                                fibUpforceDY, fibUpforceDZ, mu, epsilon, Lf, w1, w3, w5);
                            sqneeded[targStart+iT] = 0; 
                        } // end special quad needed
                    } else {
                        uSBT = OneRPYSBTKernel(xtarg[targStart+iT],ytarg[targStart+iT],
                        ztarg[targStart+iT],nUpsample,fibUpX,fibUpY,fibUpZ,forceUpX,
                        forceUpY,forceUpZ,mu,a,1);
                    } // end if initial root converged
                }
            }
            for (int d=0; d < 3; d++){
                allus[3*sortedTargs[targStart+iT]+d]+= SBTwt*uSBT[d] + CLwt*CLpart[d] -uRPY[d];
            }
        }
        targStart+= nT;
    }
    return std::make_tuple(allus,sqneeded,allRoots, dstars);
}

std::tuple<int, std::vector<double>> findQtype(std::vector<double> targ, int Npts, std::vector<double> xfib,
                    std::vector<double> yfib,std::vector<double> zfib, double g, double Lx, double Ly,
                    double Lz, double q1cut, double q2cut){
    std::vector<double> rvec(3,0.0);
    std::vector<double> shift(3,0.0);
    int qtype=0;
    double nr;
    double rmin = q1cut;
    for (int iPt=0; iPt < Npts; iPt++){
        rvec[0]=targ[0]-xfib[iPt];
        rvec[1]=targ[1]-yfib[iPt];
        rvec[2]=targ[2]-zfib[iPt];
        rvec = calcShifted(rvec,g,Lx,Ly,Lz);
        nr = sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
        if (nr < q2cut){
            qtype=2;
            shift[0]=targ[0]-xfib[iPt]-rvec[0];
            shift[1]=targ[1]-yfib[iPt]-rvec[1];
            shift[2]=targ[2]-zfib[iPt]-rvec[2];
            return std::make_tuple(qtype,shift);
        } else if (nr < rmin){
            rmin=nr;
            qtype=1;
            shift[0]=targ[0]-xfib[iPt]-rvec[0];
            shift[1]=targ[1]-yfib[iPt]-rvec[1];
            shift[2]=targ[2]-zfib[iPt]-rvec[2];
        }
    }
    return std::make_tuple(qtype,shift);
}

// Method to determine the quadrature for Chebyshev and uniform points. 
// Inputs: numNeighbs = vector with the number of uniform neighbors for each Chebyshev point, 
// sortedNeighbs = stacked vector of neighbors sorted for each Chebyshev point,
// xCheb, yCheb, zCheb = Chebyshev point coordinates, NCheb = number of Chebyshev points,
// xUni, yUni, zUni = uniform point coordinates, NuniperFib = number of uniform points on each fiber, 
// g = strain in the coordinate system, Lens = vector of periodic Lenghts, q1cut = cutoff for type 1
// special quad, q2cut = cutoff for type 2 special quad. 
std::tuple <std::vector<int>, std::vector<int>, std::vector<int>, std::vector<double>>
determineCorQuad(std::vector<int> numNeighbs, std::vector<int> sortedNeighbs, std::vector<double> xCheb,
    std::vector<double> yCheb,std::vector<double> zCheb, int NCheb, std::vector<double> xUni,
    std::vector<double> yUni,std::vector<double> zUni, int NuniperFib, double g, std::vector<double> Lens,
    double q1cut, double q2cut){
    std::vector <int> targets;
    std::vector <int> fibers;
    std::vector <int> methods;
    std::vector <double> shifts;
    std::vector <double> rvec(3,0.0);
    double xShift, yShift, zShift;
    int qtype = 0;
    int NoTwoFound = 1;
    int nePt, neFib, iFib;
    int neighbStart=0;
    double nr;
    double rmin = q1cut;
    for (int iPt=0; iPt < NCheb; iPt++){ // loop over points
        iFib = iPt / NuniperFib;
        for (int iNe=0; iNe < numNeighbs[iPt]; iNe++){ // loop over neighbors
            nePt = sortedNeighbs[neighbStart+iNe];
            neFib = nePt/NuniperFib;
            if (NoTwoFound && neFib!=iFib){
                rvec[0] = xCheb[iPt]-xUni[nePt];
                rvec[1] = yCheb[iPt]-yUni[nePt];
                rvec[2] = zCheb[iPt]-zUni[nePt];
                rvec = calcShifted(rvec,g,Lens[0],Lens[1],Lens[2]);
                nr = sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
                if (nr < q2cut){
                    NoTwoFound = 0;
                    qtype = 2;
                    xShift = xCheb[iPt] - xUni[nePt] - rvec[0];
                    yShift = yCheb[iPt] - yUni[nePt] - rvec[1];
                    zShift = zCheb[iPt] - zUni[nePt] - rvec[2];
                } else if (nr < rmin){
                    rmin = nr;
                    qtype = 1;
                    xShift = xCheb[iPt] - xUni[nePt] - rvec[0];
                    yShift = yCheb[iPt] - yUni[nePt] - rvec[1];
                    zShift = zCheb[iPt] - zUni[nePt] - rvec[2];
                }
            } // end determine quad
            // Stop once we reach the end of the neFib uniform points
            if (iNe==numNeighbs[iPt]-1 || sortedNeighbs[neighbStart+iNe+1]/NuniperFib != neFib){
                if (qtype > 0){
                    // Add to the lists
                    //std::cout << "Doint point " << iPt << " and fiber " << neFib << "  with qtype " << qtype << "\n";
                    targets.push_back(iPt);
                    fibers.push_back(neFib);
                    methods.push_back(qtype);
                    shifts.push_back(xShift);
                    shifts.push_back(yShift);
                    shifts.push_back(zShift);
                }
                // Reset variables
                rmin = q1cut;
                qtype = 0;
                NoTwoFound = 1;
            }
        }
        neighbStart+=numNeighbs[iPt];
    }
    return std::make_tuple(targets,fibers,methods,shifts);
}
                                            


    
// Module for python
PYBIND11_MODULE(EwaldUtils, m) {
    m.doc() = "The C++ version of the functions for Ewald RPY"; // optional module docstring

    m.def("Fnear", &Fnear, "F part of the near field");
	m.def("Gnear", &Gnear, "G part of the near field");
    m.def("RPYNKer", &RPYNKer, "Near field kernel for the RPY tensor");
    m.def("RPYTot", &RPYTot, "Total kernel for the RPY tensor");
    m.def("RPYSBTKernel", &RPYSBTKernel, "RPY/SBT quadrature");
    m.def("RPYSBTAllFibers", &RPYSBTAllFibers, "Many fiber");
    m.def("calcShifted", &calcShifted, "Shift in primed variables to [-L/2, L/2]^3");
    m.def("RPYNKerPairs", &RPYNKerPairs, "RPY near sum done in pairs");
    m.def("findQtype", &findQtype, "select the near field quad");
    m.def("SBTKernelSplit",&SBTKernelSplit,"Compute SBT kernel with special wts");
    m.def("determineCorQuad",&determineCorQuad, "Make list of needed quadrature routines");
}
