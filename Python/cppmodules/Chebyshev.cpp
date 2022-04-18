#include "types.h"

/**
    Chebyshev.cpp
    This file contains functions for Chebyshev support. 
**/

void eval_Cheb3Dirs(const vec &fhat, int N, const complex &evalpt, compvec3 &results){
    /**
    Evaluate three Chebyshev polynomials in each of 3 directions (x,y,z)
    This method is used to evaluate the series on the COMPLEX
    plane. This is used in rootfinding.
    @param fhat = the coefficients of the Chebyshev serieses (as a row stacked vector)
    @param N = number of coefficients at which to truncate the expansion. The cap is 
    used because Barnett and af Klinteberg suggest it for stability 
    @param evalpt = the complex number where we evaluate the Chebyshev series
    @param results = the value of the Chebyshev series as a 3 vector at the complex value evalpt
    (modified here)
    **/
    // Compute the complex theta
    complex theta = std::acos(evalpt);
    // Series
    for (int d=0; d < 3; d++){
        results[d]=complex(0.0,0.0);
    }
    for (int j=0; j < N; j++){
        for (int d=0; d < 3; d++){
            results[d]+=fhat[3*j+d]*std::cos(1.0*j*theta);
        }
    }
}

// Second method to evaluate the series on the real line. 
// No cap on the number of coefficients. 
void eval_Cheb3Dirs(const vec &fhat, double t, vec3 &results){
    /**
    Evaluate three Chebyshev polynomials in each of 3 directions (x,y,z)
    This method is used to evaluate centerline velocities and force densties. 
    @param fhat = the coefficients of the Chebyshev serieses (as a row stacked vector)
    @param t = the double on [-1,1] where we evaluate the Chebyshev series
    @param results = the value of the Chebyshev series as a 3 vector at the coordinate t 
    (modified here)
    **/
    // Compute the real valued theta
    double theta = std::acos(t);
    int N = fhat.size()/3;
    // Series
    for (int j=0; j < N; j++){
        for (int d=0; d < 3; d++){
            results[d]+=fhat[3*j+d]*std::cos(1.0*j*theta);
        }
    }
}

void DifferentiateCoefficients(const vec &Coefficients, int numDirs, vec &DCoefficients){
    /**
    Differentiate Chebyshev coefficients using the recursive
    formula. 
    @param coefficients (an N x numDirs) array of the series X
    @param numDirs = number of directions (usually 3)
    @param DCoefficients = coefficients of the series X' (modified here)
    **/
    int N = Coefficients.size()/numDirs;
    for (int d = 0; d < numDirs; d++){
        DCoefficients[numDirs*(N-2)+d]=2.0*(N-1)*Coefficients[numDirs*(N-1)+d];
        for (int a = N-3; a > -1; a--){
            DCoefficients[numDirs*a+d]=DCoefficients[numDirs*(a+2)+d]+2.0*(a+1)*Coefficients[numDirs*(a+1)+d];
        }
        DCoefficients[d]*=0.5;
    }
}

void IntegrateCoefficients(const vec &incoefs, int numDirs, double L, vec &intcoefs){
    int N = incoefs.size()/numDirs;
    for (int d = 0; d < numDirs; d++){
        intcoefs[numDirs+d]=incoefs[d]-0.5*incoefs[2*numDirs+d];
        for (int j = 2; j < N-1; j++){
            intcoefs[numDirs*j+d] = 1.0/(2.0*j)*(incoefs[numDirs*(j-1)+d]-incoefs[numDirs*(j+1)+d]);
        }
        intcoefs[(N-1)*numDirs+d] = 1.0/(2*(N-1))*incoefs[numDirs*(N-2)+d];
    }
    for (uint j=0; j < intcoefs.size(); j++){
      intcoefs[j]*=0.5*L;
    }
}
