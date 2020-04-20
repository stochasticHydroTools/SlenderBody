#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iterator>
#include "SpecialQuadratures.cpp"
#include "utils.h"
#include "RPYKernels.cpp"

/**
    ManyFiberMethods.cpp
    C++ functions to compute the velocity due to many fibers. 
    This includes: the finite part integral, all integral kernel evaluations, 
    and the routine to correct the velocity from Ewald splitting. 
    2 main public methods
    1) CorrectNonLocalVelocity - correct the velocity from Ewald
    2) FinitePartVelocity - get the velocity due to the finite part integral
**/

// User-defined OMP reduction for a vector
#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

//===========================================
// GLOBAL VARIABLES AND THEIR INITIALIZATION
//===========================================

double epsilon, L;
int NFib, NChebperFib, NuniperFib, Nupsample;
vec NormalChebNodes, upsampledNodes, NormalChebWts, UpsampledChebWts;
double dstarCL,dstarInterp, dstar2panels;
double upsampdistance, specialdistance;
vec FinitePartMatrix, DifferentiationMatrix;
vec upsamplingMatrix, TwoPanelUpsampMatrix, ValuestoCoeffsMatrix;

void initFiberVars(double muin,vec3 Lengths, double epsIn, double Lin, 
                   int NfibIn,int NChebIn, int NuniIn){
    /**
    Initialize variables relating to each fiber
    @param muin = viscosity of fluid
    @param Lengths = 3 array of periodic lengths
    @param epsIn = fiber epsilon
    @param Lin = fiber length
    @param NfibIn = number of fibers
    @param NChebIn = number of Chebyshev points on each fiber
    @param NuniIn = number of uniform points on each fiber 
    **/
    initRPYVars(sqrt(1.5)*epsIn*Lin, muin, NfibIn*NChebIn, Lengths);
    // Set domain lengths
    initLengths(Lengths[0],Lengths[1],Lengths[2]);
    epsilon = epsIn;
    L = Lin;
    NuniperFib = NuniIn;
    NFib = NfibIn;
}

void initSpecQuadParams(double rcritIn, double dsCLin, double dInterpIn, 
                      double d2panIn, double upsampdIn, double specDIn){
    /**
    Initialize variables relating to special quadrature
    @param rcritIn = critical Bernstein radius for special quadrature
    @param dsCLin = non-dimensional distance d*=d/(epsilon*L) at which velocity = CL velocity
    @param dInterpIn = non-dimension distance d*=d/(epsilon*L) at which we STOP interpolating with CL velocity
    @param d2panIn = non-dimension distance d*=d/(epsilon*L) below which we need 2 panels for special quad
    @param upsampdIn = critical distance d*=d/L below which we need upsampled quadrature
    @param specDIn = critical distance d*=d/L below which we need special quadrature
    **/
    setRhoCrit(rcritIn);
    dstarCL = dsCLin;
    dstarInterp = dInterpIn;
    dstar2panels = d2panIn;
    upsampdistance = upsampdIn;
    specialdistance = specDIn;
}

void initNodesandWeights(const vec &normalNodesIn, const vec &normalWtsIn, const vec &upNodes, const vec &upWeights){
    /**
    Initialize nodes and weights (see documentation in pyManyFiberMethods.cpp)
    **/
    NormalChebNodes = normalNodesIn;
    NChebperFib = normalNodesIn.size();
    NormalChebWts = normalWtsIn;
    upsampledNodes = upNodes;
    Nupsample = upsampledNodes.size();
    UpsampledChebWts = upWeights;
}

void initResamplingMatrices(const vec &UpsampMatIn, const vec &Upsamp2PanMat, const vec &ValstoCoeffs){
    /**
    Initialize resampling matrices (see documentation in pyManyFiberMethods.cpp)
    **/
    upsamplingMatrix = UpsampMatIn;
    TwoPanelUpsampMatrix = Upsamp2PanMat;
    ValuestoCoeffsMatrix = ValstoCoeffs;
}

void initFinitePartMatrix(const vec &FPIn, const vec &DMatIn){
    /**
    Initialize matrices for finite part integration (see documentation in pyManyFiberMethods.cpp)
    **/
    FinitePartMatrix = FPIn;
    DifferentiationMatrix = DMatIn;
}

//===========================================
// PRIVATE METHODS CALLED ONLY IN THIS FILE
//===========================================
void OneRPYKernel(const vec3 &targ, const vec &sourcePts, const vec &Forces, vec3 &utarg){
    /**
    Compute the RPY kernel at a specific target due to a fiber. 
    @param targ = the target position (3 array)
    @param sourcePts = fiber points along which we are summing the kernel (row stacked vector)
    @param Forces = forces (NOT FORCE DENSITIES) at the fiber points (row stacked vector)
    @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
    **/
    vec3 rvec;
    vec3 force;
    utarg = {0,0,0};
    double r, rdotf, F, G, co2;
    double outFront = 1.0/mu;
    int Nsrc = sourcePts.size()/3;
    for (int iSrc=0; iSrc < Nsrc; iSrc++){
        for (int d=0; d < 3; d++){
            rvec[d]=targ[d]-sourcePts[3*iSrc+d];
            force[d] = Forces[3*iSrc+d];
        }
        r = normalize(rvec);
        rdotf = dot(rvec,force);
        F = FtotRPY(r);
        G = GtotRPY(r);
        co2 = rdotf*G-rdotf*F;
        for (int d=0; d<3; d++){
            utarg[d]+=outFront*(F*force[d]+co2*rvec[d]);
        }
    }
}

void RPYFiberKernel(const vec &Targets, const vec &FibPts, const vec &Forces, vec &targVels){
    /**
    Compute the integral of the RPY kernel along a fiber with respect to some targets
    @param Targets = the target positions (row stacked vector)
    @param FibPts = fiber points along which we are summing the kernel (row stacked vector)
    @param Forces = forces (NOT FORCE DENSITIES) at the fiber points (row stacked vector)
    @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
    **/
    int Ntarg = Targets.size()/3;
    vec3 uadd;
    for (int iTarg=0; iTarg < Ntarg; iTarg++){
        vec3 targPt = {Targets[3*iTarg],Targets[3*iTarg+1],Targets[3*iTarg+2]};
        OneRPYKernel(targPt,FibPts,Forces,uadd);
        for (int d=0; d<3; d++){
            targVels[iTarg*3+d]+=uadd[d];
        }
    }
}

void OneRPYKernel(const vec3 &targ, const vec &sourcePts, const vec &ForceDs, const vec &wts, 
                  int first, int last, vec3 &utarg){
    /**
    Donev: What is the difference with the previous OneRPYKernel routine above?
    I would suggest against naming routines the same thing as it leads to confusion for someone reading the code
    The compiler can read protypes/signatures of routines and figure out which to call but humans like to be able to search for a name
    and use names to understand what things are
    
    Compute the RPY kernel at a specific target due to a fiber. 
    @param targ = the target position (3 array)
    @param sourcePts = fiber points along which we are summing the kernel (row stacked vector)
    @param ForceDs = forces DENSITIES at the fiber points (row stacked vector)
    @param wts = quadrature weights for the integration
    @param first = where to start kernel the kernel (row in SourcePts)
    @param last = index of sourcePts where we stop adding the kernel (row in sourcePts)
    @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
    **/
    vec3 rvec;
    vec3 force;
    utarg = {0,0,0};
    double r, rdotf, F, G, co2;
    double outFront = 1.0/mu;
    for (int iSrc=first; iSrc < last; iSrc++){
        for (int d=0; d < 3; d++){
            rvec[d]=targ[d]-sourcePts[3*iSrc+d];
            force[d] = ForceDs[3*iSrc+d]*wts[iSrc-first];
        }
        r = normalize(rvec);
        rdotf = dot(rvec,force);
        F = FtotRPY(r);
        G = GtotRPY(r);
        co2 = rdotf*G-rdotf*F;
        for (int d=0; d<3; d++){
            utarg[d]+=outFront*(F*force[d]+co2*rvec[d]);
        }
    }
}

// Method to compute the SBT kernel for a single fiber and target. 
void OneSBTKernel(const vec3 &targ, const vec &sourcePts, const vec &ForceDs, const vec &wts, 
                  int first, int last, vec3 &utarg){
    /**
    Compute the SBT kernel at a specific target due to a fiber. 
    @param targ = the target position (3 array)
    @param sourcePts = fiber points along which we are summing the kernel (row stacked vector)
    @param ForceDs = forces DENSITIES at the fiber points (row stacked vector)
    @param wts = quadrature weights for the integration
    @param first = where to start kernel the kernel (row in SourcePts)
    @param last = index of sourcePts where we stop adding the kernel (row in sourcePts)
    @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
    **/
    vec3 rvec;
    vec3 force;
    double a = sqrt(1.5)*epsilon*L;
    double r, rdotf, F, G, co2;
    double outFront = 1.0/mu;
    for (int iSrc=first; iSrc < last; iSrc++){
        for (int d=0; d < 3; d++){
            rvec[d]=targ[d]-sourcePts[3*iSrc+d];
            force[d] = ForceDs[3*iSrc+d]*wts[iSrc-first];
        }
        r = normalize(rvec);
        rdotf = dot(rvec,force);
        F = (2.0*a*a+3.0*r*r)/(24*M_PI*r*r*r);
        G = (-2.0*a*a+3.0*r*r)/(12*M_PI*r*r*r);
        co2 = rdotf*G-rdotf*F;
        for (int d=0; d<3; d++){
            utarg[d]+=outFront*(F*force[d]+co2*rvec[d]);
        }
    }
}

void SBTKernelSplit(const vec3 &targpt, const vec &FibPts, const vec &ForceDens, 
                    const vec &w1, const vec &w3, const vec &w5, vec3 &utarg){
    /**
    Method for SBT quadrature when the weights are different for different powers of r in the denominator.
    (as they are in special quadrature)
    @param targpt = the target position (3 array)
    @param FibPts = fiber points along which we are summing the kernel (row stacked vector)
    @param ForceDens = forces DENSITIES at the fiber points (row stacked vector)
    @param w1 = quadrature weights for 1/R power 
    @param w3 = quadrature weights for 1/R^3 power
    @param w5 = quadrature weights for 1/R^5 power
    @param utarg = velocity at the target due to the fiber (using RPY). 3 array, Passed by reference and modified
    **/
    vec3 rvec;
    vec3 forceDensity;
    double r, rdotf, u1, u3, u5;
    double outFront = 1.0/(8.0*M_PI*mu);
    int N = FibPts.size()/3;
    for (int iPt=0; iPt < N; iPt++){
        for (int d=0; d < 3; d++){
            rvec[d] = targpt[d]-FibPts[3*iPt+d];
            forceDensity[d] = ForceDens[3*iPt+d];
        }
        rdotf = dot(rvec,forceDensity);
        r = normalize(rvec);
        for (int d=0; d < 3; d++){
            u1 = forceDensity[d]/r;
            u3 = (r*rvec[d]*rdotf+(epsilon*L)*(epsilon*L)*forceDensity[d])/(r*r*r);
            u5 = -3.0*(epsilon*L)*(epsilon*L)*rvec[d]*r*rdotf/pow(r,5);
            utarg[d]+=outFront*(u1*w1[iPt]+u3*w3[iPt]+u5*w5[iPt]);
        }
    }
}

// Method to find the necessary quadrature type for a given target and fiber. 
// Inputs: targ = target points, Npts = number of points on fiber, 
// (xfib,yfib,zfib) = vectors of fiber pints, g = strain, (Lx,Ly,Lz) = periodic lengths
// q1cut = distance where we need upsampled direct, q2cut = distance where we need special quadrature. 
int determineQuadratureMethod(const vec &UniformFiberPoints, int iFib, double g, vec3 &targetPoint){
    /**
    Method to find the necessary quadrature type for a given target and fiber. 
    (as they are in special quadrature)
    @param UniformFiberPoints = row stacked vector of ALL uniform points on fibers
    @param iFib = fiber number 
    @param g = strain in the coordinate system 
    @param targetPoint = the target position (3 array) 
    @return 0 if NChebperFib quadrature is acceptable (that's what comes out of Ewald), 1
    if upsampled quadrature is needed, 2 if special quadrature is needed. The target point is 
    also modified with a periodic shift. 
    **/
    int qtype=0;
    vec3 rvec;
    vec3 rvecmin={0,0,0};
    double rmin = upsampdistance*L; // cutoff for upsampled quad
    int iPtMin=0;
    for (int iPt=0; iPt < NuniperFib; iPt++){
        for (int d=0; d < 3; d++){
            rvec[d] = targetPoint[d]-UniformFiberPoints[3*(iFib*NuniperFib+iPt)+d];
        }
        calcShifted(rvec,g); // periodic shift wrt strain 
        double nr = sqrt(rvec[0]*rvec[0]+rvec[1]*rvec[1]+rvec[2]*rvec[2]);
        if (nr < specialdistance*L){
            for (int d=0; d < 3; d++){
                targetPoint[d] = rvec[d] + UniformFiberPoints[3*(iFib*NuniperFib+iPt)+d];
            }
            return 2;
        } else if (nr < rmin){
            rmin=nr;
            qtype=1;
            rvecmin = {rvec[0],rvec[1],rvec[2]};
            iPtMin = iPt;
        }
    }
    for (int d=0; d < 3; d++){ // periodic shift in the target point
        targetPoint[d] = rvecmin[d] + UniformFiberPoints[3*(iFib*NuniperFib+iPtMin)+d];
    }
    return qtype;
}

double FindClosestFiberPoint(double tapprox, const vec &PositionCoefficients, const vec3 &target, vec3 &ClosestPoint){
    /**
    Find the closest point on a fiber to a target. 
    @param tapprox = closest centerline coordinate (on [-1,1] to the target)
    @param PositionCoefficients = Chebyshev coefficients of the centerline position (row stacked vector)
    @param target = the targetPoint (a 3 array)
    @param ClosestPoint = the closest point on the fiber (modified here)
    @return the non-dimensional closest distance d*=distance from fiber /(epsilon*L). 
    **/
    // Compute closest fiber point
    ClosestPoint = {0,0,0};
    eval_Cheb3Dirs(PositionCoefficients, tapprox, ClosestPoint);
    vec3 dfromFib;
    for (int d=0; d< 3; d++){
        dfromFib[d] = ClosestPoint[d]-target[d];
    }
    double dstar = sqrt(dot(dfromFib,dfromFib))/(epsilon*L); // non-dimensional distance
    return dstar;
}
  
void initFullFiberForSpecial(const vec &ChebFiberPoints, const vec &FibForceDensities,const vec &CenterlineVelocities, int iFib,
    vec &UpsampledPos, vec &UpsampledForceDs, vec &PositionCoefficients, vec &DerivCoefficients, vec &CenterlineVelCoefficients, 
    vec &forceDCoefficients){
    /**
    "PRIVATE" method to initialize the vectors for a particular fiber
    @param ChebFiberPoints = points on all fibers 
    @param FibForceDensities = fiber force densities on all fibers
    @param CenterlineVelocities = finite part velocities of the centerline on all fibers
    @param iFib = fiber number
    @return This is a void method, but it modifies the following arrays: the upsampled positions, upsampled force densities,
    coefficients of the fiber position, tangent vectors, finite part (centerline) velocities, and force densities. 
    **/
    int start = NChebperFib*iFib;
    // Upsample points and forces to Nupsample point grid
    MatVec(Nupsample, NChebperFib, 3, upsamplingMatrix, ChebFiberPoints, start, UpsampledPos);
    MatVec(Nupsample, NChebperFib, 3, upsamplingMatrix, FibForceDensities, start, UpsampledForceDs);
    // Compute coefficients
    MatVec(NChebperFib, NChebperFib, 3, ValuestoCoeffsMatrix, ChebFiberPoints, start, PositionCoefficients);
    DifferentiateCoefficients(PositionCoefficients,3,DerivCoefficients);
    MatVec(NChebperFib, NChebperFib, 3, ValuestoCoeffsMatrix, FibForceDensities, start, forceDCoefficients);
    // Centerline velocity coefficients
    MatVec(NChebperFib, NChebperFib, 3, ValuestoCoeffsMatrix, CenterlineVelocities, start, CenterlineVelCoefficients);
}

void init2PanelsForSpecial(const vec &ChebFiberPoints, const vec &FibForceDensities, int iFib, vec &Pan1Pts, vec &Pan2Pts, 
           vec &Pan1FDens, vec &Pan2FDens, vec &Pan1Coeffs, vec &Pan1DCoeffs, vec &Pan2Coeffs, vec &Pan2DCoeffs){
    /**
    "PRIVATE" method to initialize 2 panels for a particular fiber
    @param ChebFiberPoints = points on all fibers 
    @param FibForceDensities = fiber force densities on all fibers
    @param iFib = fiber number
    @return This is a void method, but it modifies the following arrays: the positions, force densities, and coefficients
    on each of the 2 panels. 
    **/
    int start = NChebperFib*iFib;
    // Points
    vec TwoPanelsPoints(2*3*NChebperFib,0.0);
    MatVec(2*NChebperFib, NChebperFib, 3, TwoPanelUpsampMatrix, ChebFiberPoints, start, TwoPanelsPoints);
    MatVec(Nupsample, NChebperFib, 3, upsamplingMatrix, TwoPanelsPoints, 0, Pan1Pts);
    MatVec(Nupsample, NChebperFib, 3, upsamplingMatrix, TwoPanelsPoints, NChebperFib, Pan2Pts);
    // Forces
    vec TwoPanelsForceDs(2*3*NChebperFib,0.0);
    MatVec(2*NChebperFib, NChebperFib, 3, TwoPanelUpsampMatrix, FibForceDensities, start, TwoPanelsForceDs);
    MatVec(Nupsample, NChebperFib, 3, upsamplingMatrix, TwoPanelsForceDs, 0, Pan1FDens);
    MatVec(Nupsample, NChebperFib, 3, upsamplingMatrix, TwoPanelsForceDs, NChebperFib, Pan2FDens);
    // 2 panel coefficients
    MatVec(NChebperFib, NChebperFib, 3, ValuestoCoeffsMatrix, TwoPanelsPoints, 0, Pan1Coeffs);
    DifferentiateCoefficients(Pan1Coeffs,3,Pan1DCoeffs);
    MatVec(NChebperFib, NChebperFib, 3, ValuestoCoeffsMatrix, TwoPanelsPoints, NChebperFib, Pan2Coeffs);
    DifferentiateCoefficients(Pan2Coeffs,3,Pan2DCoeffs);
}

void subtractAllRPY(const vec &ChebFiberPoints, const vec &FibForceDensities, vec &correctionUs){
    /**
    Method to subtract the RPY kernel at all targets due to the fiber that target belongs to. 
    First step in correcting the Ewald velocities. 
    @param ChebFiberPoints = points on all fibers 
    @param FibForceDensities = fiber force densities on all fibers
    @param correctionUs = correction velocities that are incremented
    **/
    for (int iPt=0; iPt < NChebperFib*NFib; iPt++){
        int fibNum = iPt/NChebperFib;
        vec3 uRPYSelf = {0,0,0};
        vec3 target = {ChebFiberPoints[3*iPt], ChebFiberPoints[3*iPt+1], ChebFiberPoints[3*iPt+2]};
        OneRPYKernel(target, ChebFiberPoints, FibForceDensities, NormalChebWts,NChebperFib*fibNum, NChebperFib*(fibNum+1), uRPYSelf);
        for (int d =0; d< 3; d++){
            correctionUs[3*iPt+d]-= uRPYSelf[d]; // SUBTRACT the RPY kernel
        }
    }
}

void calcCLVelocity(const vec &FinitePartCoefficients, const vec &DerivCoefficients, 
                    const vec &forceDCoefficients, double tapprox, vec3 &CLpart){
    /**
    Method to calculate the centerline velocity of a fiber at coordinate tapprox in [-1,1]
    @param FinitePartCoefficients = coefficients of the Chebyshev series of the FP velocity
    @param DerivCoefficients = coefficients of the Chebyshev series of the tangent vectors
    @param forceDCoefficients = coefficients of the Chebyshev series of the force densities
    @param tapprox = centerline coordinate on [-1,1]
    @param CLpart = centerline velocity (modified here)
    **/
    eval_Cheb3Dirs(FinitePartCoefficients, tapprox, CLpart); // evaluate finite part velocity and add to CLpart
    // Compute velocity due to local drag
    vec3 Xs = {0,0,0};
    vec3 forceDen = {0,0,0};
    eval_Cheb3Dirs(DerivCoefficients, tapprox, Xs);
    eval_Cheb3Dirs(forceDCoefficients, tapprox, forceDen);
    double Xsdotf = dot(Xs,forceDen);
    double c = log((2.0*(1.0-tapprox*tapprox)+2.0*sqrt((1.0-tapprox*tapprox)*(1.0-tapprox*tapprox)+
        16.0*epsilon*epsilon))/(4.0*epsilon*epsilon)); // regularized leading order coefficient
    for (int d =0; d < 3; d++){
        CLpart[d] += 1.0/(8*M_PI*mu)*(c*(forceDen[d]+Xs[d]*Xsdotf)+(forceDen[d]-3*Xs[d]*Xsdotf));
    }
}

//===========================================
// PUBLIC METHODS CALLED FROM OUTSIDE
//===========================================
void CorrectNonLocalVelocity(const vec &ChebFiberPoints, const vec &UniformFiberPoints, const vec &FibForceDensities, 
                             const vec &FinitePartVelocities, double g,const intvec &numTargsbyFib, 
                             const intvec &allTargetNums, vec &correctionUs, int nThreads){
    /**
    Donev: Sorry, this is too complicated/long for me to follow. As long as you checked that it gives the same results (for a given input) as the previous python code I am fine with it...
    Method to correct the velocity from Ewald via special quadrature (or upsampled quadrature)
    @param ChebFiberPoints = vector (row stacked) of Chebyshev points on ALL fibers
    @param UniformFiberPoints = vector (row stacked) of uniform points on ALL fibers
    @param FibForceDensities = vector (row stacked) of force densities on ALL fibers
    @param FinitePartVelocities = vector (row stacked) of velocities due to the finite part integral on 
        ALL fibers (necessary when the fibers are close togther and the centerline velocity is used)
    @param g = strain in coordinate system
    Donev: The wording "on each fiber" appears several times but it confuses me -- I believe you mean "for each fiber"
    That is, for each fiber there is a list of targets (on other fibers) for which we need to subtract RPY and then do special quad, right?
    @param numTargsbyFib = int vector of the number of targets that require correction on each fiber
    @param allTargetNums = vector of the target indices that need correction in sequential order
    @param correctionUs = vector (row stacked) of correction velocities (modified here) . 
    @param nThreads = number of threads to use in parallel processing
    **/
    // Subtract the self RPY for a single fiber
    subtractAllRPY(ChebFiberPoints, FibForceDensities, correctionUs);
    // Cumulative sum of the number of targets by fiber to figure out where to start
    intvec endTargNum(numTargsbyFib.begin(),numTargsbyFib.end());
    std::partial_sum(endTargNum.begin(), endTargNum.end(),endTargNum.begin());
    omp_set_num_threads(nThreads); // Donev: I believe that you can do this inside the omp parallel pragma, instead of like this. Then no need to reset
    // Donev: Why is the schedule here fixed to dynamic?
    #pragma omp parallel for schedule(dynamic)
    for (int iFib=0; iFib < NFib; iFib++){
        // Initialize fiber specific quantities
        vec UpsampledPos(3*Nupsample,0.0), UpsampledForceDs(3*Nupsample,0.0), PositionCoefficients(3*NChebperFib,0.0);
        vec DerivCoefficients(3*NChebperFib,0.0), FinitePartCoefficients(3*NChebperFib,0.0), forceDCoefficients(3*NChebperFib,0.0);
        initFullFiberForSpecial(ChebFiberPoints,FibForceDensities,FinitePartVelocities,iFib, UpsampledPos, 
            UpsampledForceDs, PositionCoefficients, DerivCoefficients,FinitePartCoefficients, forceDCoefficients);
        // 2 Panels
        vec Pan1Pts(3*Nupsample,0.0), Pan2Pts(3*Nupsample,0.0), Pan1FDens(3*Nupsample,0.0), Pan2FDens(3*Nupsample,0.0);
        vec Pan1Coeffs(3*NChebperFib,0.0), Pan2Coeffs(3*NChebperFib,0.0), Pan1DCoeffs(3*NChebperFib,0.0), Pan2DCoeffs(3*NChebperFib,0.0);
        init2PanelsForSpecial(ChebFiberPoints, FibForceDensities, iFib, Pan1Pts, Pan2Pts, Pan1FDens, Pan2FDens, Pan1Coeffs, 
            Pan1DCoeffs, Pan2Coeffs, Pan2DCoeffs);
        // Loop over targets and do corrections
        for (int iT=endTargNum[iFib]-numTargsbyFib[iFib]; iT < endTargNum[iFib]; iT++){
            int ptNumber = allTargetNums[iT]; // target number (from 0 to Nfib*NptsbyFib
            vec3 uRPY={0,0,0}; // RPY subtraction from the velocity
            vec3 uSBT={0,0,0}; // SBT componenent
            vec3 CLpart={0,0,0}; // component coming from the centerline
            double CLwt = 0.0; // weight of centerline (for close points)
            double SBTwt = 1.0; // weight of SBT
            vec3 targetPoint;
            for (int d=0; d< 3; d++){
                targetPoint[d]= ChebFiberPoints[3*ptNumber+d];
            }
            // First determine what quadrature is necessary
            int qtype = determineQuadratureMethod(UniformFiberPoints, iFib, g, targetPoint);
            if (qtype > 0){ // only do correction if necessary
                // Subtract RPY kernel
                OneRPYKernel(targetPoint,ChebFiberPoints,FibForceDensities,NormalChebWts,iFib*NChebperFib,(iFib+1)*NChebperFib, uRPY);
                if (qtype==1){
                    // Correct with upsampling (SBT kernel)
                    OneSBTKernel(targetPoint,UpsampledPos,UpsampledForceDs,UpsampledChebWts, 0, Nupsample, uSBT);
                } else { // special quad
                    // Calculate root
                    complex troot;
                    int sqneeded = calculateRoot(upsampledNodes, UpsampledPos, PositionCoefficients,DerivCoefficients,targetPoint, troot);
                    // Estimate distance from fiber (tapprox = 1 if real(troot > 1), -1 if real(troot < -1.0))
                    double tapprox = std::max(std::min(real(troot),1.0),-1.0);
                    vec3 closestFibPoint;
                    double dstar = FindClosestFiberPoint(tapprox,PositionCoefficients,targetPoint,closestFibPoint);
                    if (dstar < dstarInterp){
                         // Compute weight and value of CL velocity assigned to nearest CL velocity
                        CLwt = std::min((dstarInterp-dstar)/(dstarInterp-dstarCL),1.0); // takes care of very close ones
                        calcCLVelocity(FinitePartCoefficients, DerivCoefficients, forceDCoefficients, tapprox, CLpart);
                    } 
                    SBTwt = 1.0-CLwt;
                    if (sqneeded==1){ // special quadrature
                        // Here is where we differentiate between one panel and two
                        vec w1(Nupsample), w3(Nupsample), w5(Nupsample);
                        if (dstar > dstar2panels){ // 1 panel of Nupsample
                            specialWeights(upsampledNodes,troot,w1,w3,w5,L);
                            SBTKernelSplit(targetPoint,UpsampledPos,UpsampledForceDs, w1, w3, w5, uSBT);
                        } else if (dstar > dstarCL){ // 2 panels of Nupsample
                            sqneeded = calculateRoot(upsampledNodes, Pan1Pts,Pan1Coeffs,Pan1DCoeffs,targetPoint, troot);
                            int sqneeded1 = sqneeded;
                            if (sqneeded){
                                specialWeights(upsampledNodes,troot,w1,w3,w5,L);
                                SBTKernelSplit(targetPoint,Pan1Pts,Pan1FDens, w1, w3, w5, uSBT);
                            } else {
                                OneSBTKernel(targetPoint,Pan1Pts,Pan1FDens,UpsampledChebWts, 0, Nupsample, uSBT);
                            }
                            sqneeded = calculateRoot(upsampledNodes, Pan2Pts,Pan2Coeffs,Pan2DCoeffs,targetPoint, troot);
                            if (sqneeded==1 && sqneeded1==1){
                                throw std::runtime_error("Special quad needed for both panels!"); // Donev: Why can't we do this?
                            }
                            if (sqneeded){
                                specialWeights(upsampledNodes,troot,w1,w3,w5,L);
                                SBTKernelSplit(targetPoint,Pan2Pts,Pan2FDens, w1, w3, w5, uSBT);
                            } else {
                                OneSBTKernel(targetPoint,Pan2Pts,Pan2FDens,UpsampledChebWts, 0, Nupsample, uSBT);
                            }
                            SBTwt*=0.5; // 2 panels, the weights are twice what they should be. Compensate here. 
                        }
                    } else{ // special quad not needed, do full fiber direct
                        OneSBTKernel(targetPoint,UpsampledPos,UpsampledForceDs,UpsampledChebWts, 0, Nupsample, uSBT);
                    } 
                } // end special quad
                // Add velocity to total velocity
                for (int d=0; d < 3; d++){
                    #pragma omp atomic update
                    correctionUs[3*ptNumber+d]+= SBTwt*uSBT[d] + CLwt*CLpart[d] -uRPY[d];
                }
            } // end if correction needed
        } // end loop over targets
    } // end parallel loop over fibers
    omp_set_num_threads(1); // todo: check if re-setting nThreads is necessary 
    // Donev: I think you should just set n_threads in the omp pragma itself -- I believe you can use variables inside the pragma too, not just constants. Try it
}

void FinitePartVelocity(const vec &ChebPoints, const vec &FDens,const vec &Xs,vec &uFP){
    /**
    Method to compute the velocity due to the finite part integral
    @param ChebPoints = vector (row stacked) of Chebyshev points on ALL fibers
    @param FDens = vector (row stacked) of uniform points on ALL fibers
    @param Xs = vector (row stacked) of tangent vectors on ALL fibers
    @param uFP = vector (row stacked) of velocities due to the finite part integral on 
        ALL fibers (modified here)
    **/
    int NfibIn = ChebPoints.size()/(3*NChebperFib); // determine how many fiber we are working with
    for (int iFib=0; iFib < NfibIn; iFib++){
        int start = iFib*NChebperFib;
        vec Xss(3*NChebperFib,0.0);
        vec fprime(3*NChebperFib,0.0);
        // Compute Xss, fprime
        MatVec(NChebperFib, NChebperFib, 3, DifferentiationMatrix, Xs, start, Xss);
        MatVec(NChebperFib, NChebperFib, 3, DifferentiationMatrix, FDens, start, fprime);
        for (int iPt=0; iPt < NChebperFib; iPt++){
            int iPtInd = start+iPt;
            double nr, rdotf, oneoverr, ds, oneoverds;
            vec3 rvec;
            double Xsdotf = Xs[3*iPtInd]*FDens[3*iPtInd]+Xs[3*iPtInd+1]*FDens[3*iPtInd+1]+Xs[3*iPtInd+2]*FDens[3*iPtInd+2];
            double Xssdotf = Xss[3*iPt]*FDens[3*iPtInd]+Xss[3*iPt+1]*FDens[3*iPtInd+1]+Xss[3*iPt+2]*FDens[3*iPtInd+2];
            double Xsdotfprime = Xs[3*iPtInd]*fprime[3*iPt]+Xs[3*iPtInd+1]*fprime[3*iPt+1]+Xs[3*iPtInd+2]*fprime[3*iPt+2];
            for (int jPt=0; jPt < NChebperFib; jPt++){
                int jPtInd = start+jPt;
                for (int d=0; d < 3; d++){
                    rvec[d] = ChebPoints[3*iPtInd+d]-ChebPoints[3*jPtInd+d];
                }
                nr = sqrt(dot(rvec,rvec));
                oneoverr = 1.0/nr;
                ds = NormalChebNodes[jPt]-NormalChebNodes[iPt];
                oneoverds = 1.0/ds;
                rdotf = rvec[0]*FDens[3*jPtInd]+rvec[1]*FDens[3*jPtInd+1]+rvec[2]*FDens[3*jPtInd+2];
                for (int d = 0; d < 3; d++){
                    // Compute the density g from Tornberg's paper and multiply by FinitePartMatrix to get 
                    // the velocity due to the FP integral.
                    if (iPt==jPt){
                        uFP[3*iPtInd+d] += (0.5*(Xs[3*iPtInd+d]*Xssdotf+Xss[3*iPt+d]*Xsdotf)+
                            fprime[3*iPt+d]+Xs[3*iPtInd+d]*Xsdotfprime) *FinitePartMatrix[NChebperFib*iPt+iPt];
                    } else{
                        uFP[3*iPtInd+d] += ((FDens[3*jPtInd+d] + rvec[d]*rdotf*oneoverr*oneoverr)*oneoverr*std::abs(ds)-\
                            (FDens[3*iPtInd+d]+Xs[3*iPtInd+d]*Xsdotf))*oneoverds*FinitePartMatrix[NChebperFib*iPt+jPt]; 
                    }
                }
            }
        }
    }
}
