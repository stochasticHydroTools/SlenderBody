#include <stdio.h>
#include "types.h"
#include "cblas.h"
#include "lapacke.h"
#include<exception>


void throwLapackeError(int info, std::string functionName){
  throw std::runtime_error("LAPACKE Failed in function "+
			   functionName +
			   " with error code "+std::to_string(info));
}

#define LAPACKESafeCall(info) {if(info != 0){throwLapackeError(info, __func__);}}

//#include <mkl.h> // Use when compiling with intel
#pragma once // only include once

// Some standard vector methods 
double dot(const vec3 &a, const vec3 &b){
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

void cross(const vec3 &a, const vec3 &b, vec3 &result){
    result = {a[1]*b[2]-a[2]*b[1],-a[0]*b[2]+a[2]*b[0],a[0]*b[1]-b[0]*a[1]};
}

void plus(const vec &a, double alpha, const vec &b, double beta, vec &result){
    // Return alpha*A+beta*B
    for (uint i=0; i < a.size(); i++){
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

/*  Solve Ax = b using pinv(A)*b. 
    In some cases, we need pinv(A)^(1/2)*b. This is what is returned when half is true. 
    IMPORTANT: there are very rare times when the LAPACK SVD does not converge. It uses 
    some iterative algorithm. Plugging the same matrix into MATLAB will give the correct SVD
    When it doesn't converge, the singular values are out of order. What we do is just go with 
    the first 2N+3 and remove any of the zero ones. This happens maybe once out of 10^6 solves,
    so the important thing is just that the code does not crash or go crazy. 
*/
void SolveWithPseudoInverse(int m, int n, vec &A, const vec &b, vec &answer,double svdtol, bool normalize,bool half, int maxModes){
    int dimS = std::min(m,n);
    int lda = n;
    int ldu = m;
    int ldvt = n;
    vec u(m*m), s(dimS), vt(n*n);

    //computing the SVD
    vec Acopy(m*n);
    std::memcpy(Acopy.data(),A.data(),A.size()*sizeof(double));
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', m, n, &*A.begin(), lda, &*s.begin(),
                   &*u.begin(), ldu, &*vt.begin(), ldvt);
    if (info > 0){ 
         std::cout << "LAPACK SVD DID NOT CONVERGE" << std::endl;
         std::cout << "This means singular values will be out of order" << std::endl;
         std::cout << "The matrix" << std::endl;
         for (uint i = 0; i < Acopy.size(); i++){
                 std::cout << Acopy[i] << std::endl;
         }
         std::cout << "The singular values returned" << std::endl;
         for (uint i = 0; i < s.size(); i++){
                 std::cout << s[i] << std::endl;
         }
         std::cout << "The U matrix returned" << std::endl;
         for (uint i = 0; i < u.size(); i++){
                 std::cout << u[i] << std::endl;
         }
         std::cout << "The V matrix returned" << std::endl;
         for (uint i = 0; i < vt.size(); i++){
                 std::cout << vt[i] << std::endl;
         }
    }
    //LAPACKESafeCall(info);
    double sZero = 1.0;
    if (normalize){
        sZero = *max_element(s.begin(), s.end());
    }
    if (info > 0){
        std::cout << "Normalizing by " << sZero << std::endl;
    }
    // Pseudo-inverse calculation
    for (int i = 0; i < dimS; i++) {
        if (s[i]/sZero > svdtol && i < maxModes){// Invert the singular values which are nonzero
            if (half){
                s[i] = 1.0 / sqrt(s[i]);
            } else {
                s[i] = 1.0 / s[i];
            }
        } else {
            if (i < maxModes && maxModes > 6){
                std::cout << "Mode " << i << " has singular value below threshold " << svdtol << std::endl;
            }
            s[i] = 0;
        }  
        answer[i]=0;
    }
    // Mat vec U^T*b
    int nb = b.size()/m;
    vec Ustarb(m*nb);
    BlasMatrixProduct(m, m, nb,1.0, 0.0,u, true, b, Ustarb);
    // pinv(S)*U^T*b
    for (int i =0; i < dimS; i++){
        for (int iC=0; iC < nb; iC++){
            Ustarb[i*nb+iC]*=s[i];
        }
    }
    // Resize Ustarb by adding/removing rows at end
    Ustarb.resize(n*nb,0.0);
    // Multiply by V to get V*S^(-1)*U^T*b
    BlasMatrixProduct(n, n, nb,1.0, 0.0,vt, true, Ustarb, answer);
}

// Apply A^(1/2)*b and A^(-1/2)*b using eigenvalue decomposition
// Only implemented for square matrices at the moment
void ApplyMatrixHalfAndMinusHalf(int n, const vec &A, const vec &b, vec &PlusHalf, vec &MinusHalf){
    vec s(n);
    vec Acopy(A.size());
    for (int i =0; i < n; i++){ // make copy of symmetric half of A. Has to be in column major just this once.
        for (int j = 0; j <= i; j++){
            Acopy[i*n+j]=A[i*n+j];
        }
    }
    //computing the eigenvalue decomp
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U',n, &*Acopy.begin(), n, &*s.begin());
    LAPACKESafeCall(info);
    
    vec sminushalf(n);
    // Halving the eigenvalues
    for (int i = 0; i < n; i++) {
        if (s[i] > 0){// square root of nonzero eigenvalues
            s[i] = sqrt(s[i]);
            sminushalf[i] = 1.0/s[i];
        } else {
            s[i] = 0;
            sminushalf[i] = 0;
        }  
    }
    // Here A is now overwritten by V^T
    // Mat vec V^T*b
    vec Vstarb(n);
    BlasMatrixProduct(n, n, 1,1.0, 0.0, Acopy, false, b, Vstarb);
    vec CopyOfV(Acopy.size());
    std::memcpy(CopyOfV.data(),Acopy.data(),Acopy.size()*sizeof(double));
    // Overwrite V^T -> S^(1/2)*V^T
    for (int i = 0; i < n; i++) {
        cblas_dscal(n,s[i],&Acopy[i*n],1);
        cblas_dscal(n,sminushalf[i],&CopyOfV[i*n],1);
    }
    BlasMatrixProduct(n, n, 1,1.0, 0.0, Acopy, true, Vstarb, PlusHalf);
    BlasMatrixProduct(n, n, 1,1.0, 0.0, CopyOfV, true, Vstarb, MinusHalf);
}

// Eig value decomp with adjustment for negative eigs
void SymmetrizeAndDecomposePositive(int n, const vec &A, double threshold, vec &V, vec &s){
    for (int i =0; i < n; i++){ // make copy of symmetric half of A. Has to be in column major just this once.
        for (int j = 0; j <= i; j++){
            V[i*n+j]=0.5*(A[i*n+j]+A[j*n+i]);
        }
    }
    //computing the SVD
    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U',n, &*V.begin(), n, &*s.begin());
    LAPACKESafeCall(info);
    
    // Filter the eigenvalues
    for (int i = 0; i < n; i++) {
        if (s[i] < threshold){// square root of nonzero eigenvalues
            s[i] = threshold;
        }  
    }
}

// Apply a power of a matrix from the eigenvalue decomposition. 
void ApplyMatrixPowerFromEigDecomp(int n, double power, const vec &V, const vec &s, const vec &b, vec &result){
    vec sPow(n);
    int nb = b.size()/n;
    // Halving the eigenvalues
    for (int i = 0; i < n; i++) {
        if (s[i] <= 0){// square root of nonzero eigenvalues
            throw std::runtime_error("Eigenvalues should be positive always here!");
        } else {
            sPow[i] = pow(s[i],power);
        }  
    }
    // Here A is now overwritten by V^T
    // Mat vec V^T*b
    vec Vstarb(n*nb);
    BlasMatrixProduct(n, n, nb,1.0, 0.0, V, false, b, Vstarb);
    vec CopyOfV(V.size());
    std::memcpy(CopyOfV.data(),V.data(),V.size()*sizeof(double));
    // Overwrite V^T -> S^(pow)*V^T
    for (int i = 0; i < n; i++) {
        cblas_dscal(n,sPow[i],&CopyOfV[i*n],1);
    }
    // Overwrite V^T -> S^(-1)*V^T
    BlasMatrixProduct(n, n, nb,1.0, 0.0, CopyOfV, true, Vstarb, result);
}
