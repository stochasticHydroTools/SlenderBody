#include <stdio.h>
#include "types.h"
#pragma once // only include once 

// Some standard vector methods that are helpful in the
// RPY calculations
double dot(const vec3 &a, const vec3 &b){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

double normalize(vec3 &rvec){
    double r = sqrt(dot(rvec,rvec));
    if (r > 1e-10){
        double rinv = 1.0/r;
        rvec[0]*=rinv;
        rvec[1]*=rinv;
        rvec[2]*=rinv;
    }
    return r;
}

void MatVec(int m, int p, int n, const vec &M, const vec &V, int start, vec &result){
    /**
    Matrix vector product 
    // @param m = number of rows of M, 
    // @param p = number of columns of M = number of rows in V
    // @param n = number of cols in V (usually 3)
    // @param M = the matrix M
    // @param V = the column vector (or vectors) V 
    // start = where to start in the rows of V
    // result = the resulting mat-vec 
    **/
    for (int i=0; i < m; i++){ 
        for (int j=0; j < n; j++){ 
            for (int k=0; k < p; k++){
                result[i*n+j]+=M[i*p+k]*V[(k+start)*n+j];
            }
        }
    }
}
