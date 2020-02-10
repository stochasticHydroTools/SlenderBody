#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>

//This is a pybind11 module of C++ functions that are being used
//for non-local velocity calculations in the SBT code.

//The RPY near field can be written as M_near = F(r,xi,a)*I+G(r,xi,a)*(I-RR)
//The next 2 functions are those F and G functions.
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
std::vector<double> RPYSBTKernel(int Ntarg, std::vector<double> xtarg, std::vector<double> ytarg,
                            std::vector<double> ztarg, int Nsrc, std::vector<double> xsrc,
                            std::vector<double> ysrc, std::vector<double> zsrc,
                            std::vector<double> fx, std::vector<double> fy, std::vector<double> fz,
                            double mu, double a, int SBT){
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
                                            


    
// Module for python
PYBIND11_MODULE(EwaldUtils, m) {
    m.doc() = "The C++ version of the functions for Ewald RPY"; // optional module docstring

    m.def("Fnear", &Fnear, "F part of the near field");
	m.def("Gnear", &Gnear, "G part of the near field");
    m.def("RPYNKer", &RPYNKer, "Near field kernel for the RPY tensor");
    m.def("RPYTot", &RPYTot, "Total kernel for the RPY tensor");
    m.def("RPYSBTKernel", &RPYSBTKernel, "RPY/SBT quadrature");
    m.def("calcShifted", &calcShifted, "Shift in primed variables to [-L/2, L/2]^3");
    m.def("RPYNKerPairs", &RPYNKerPairs, "RPY near sum done in pairs");
    m.def("findQtype", &findQtype, "select the near field quad");
}
