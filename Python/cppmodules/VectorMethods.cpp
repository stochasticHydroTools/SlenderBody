#include <stdio.h>
#include "types.h"
#include "cblas.h"
#include "lapacke.h"
//#include <mkl.h>
#pragma once // only include once

// Some standard vector methods that are helpful in the
// RPY calculations


double dot(const vec3 &a, const vec3 &b){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

void cross(const vec3 &a, const vec3 &b, vec3 &result){
    result = {a[1]*b[2]-a[2]*b[1],-a[0]*b[2]+a[2]*b[0],a[0]*b[1]-b[0]*a[1]};
}

void plus(const vec &a, double alpha, const vec &b, double beta, vec &result){
    // Return alpha*A+beta*B
    for (int i=0; i < a.size(); i++){
        result[i]=alpha*a[i]+beta*b[i];
    }
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

void BlasMatrixProduct(int m, int p, int n,double alpha, double beta,const vec &a, bool transA, const vec &b, vec &c){
    /* C-> alpha*A*B+beta*C
    A = m x p
    B = p x n 
    C = m x n
    */
    int LDA = p;
    int LDB = n;
    int LDC = n;
    if (transA){
        LDA = m;
        cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans,m,n,p,alpha, & *a.begin(), LDA, & *b.begin(), LDB, beta, & *c.begin(), LDC);
    } else {
        cblas_dgemm(CblasRowMajor,CblasNoTrans, CblasNoTrans,m,n,p,alpha, & *a.begin(), LDA, & *b.begin(), LDB, beta, & *c.begin(), LDC);
    }
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
