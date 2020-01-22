import numpy as np
from numpy import pi
from numpy import cos
from scipy.sparse import diags

"""
    This file is a list of Chebyshev related functions
    for python that are needed in the slender fiber 
    code
"""

def chebpts(N,dom,kind=2):
    """
    Chebyshev points on the interval dom of kind kind. 
    Second kind Chebyshev points include the endpoints, 
    Inputs: number of points and size of domain (a 1 x 2) 
    array, and kind of points. 
    Outputs: the Chebyshev points x and the weights w. 
    """
    if kind==1:
        th=th1(N);
        m=2.0 / np.concatenate(([1],1-np.arange(2,N,2)**2));
        if np.mod(N,2):
            c=np.concatenate((m,-m[(N+1)/2-1:0:-1]));
        else:
            c=np.concatenate((m,[0],-m[N/2-1:0:-1]))
        v=np.exp(1j*np.arange(N)*np.pi/N);
        c=c*v;
        w = np.real(np.fft.ifft(c));
    elif kind==2:
        th=th2(N);
        c=2.0/np.concatenate(([1],1-np.arange(2,N,2)**2));
        c=np.concatenate((c,c[N//2-1:0:-1]));
        w = np.real(np.fft.ifft(c));
        w[0]/=2;
        w=np.concatenate((w,[w[0]]))
    x = cos(th);
    # Rescale
    x = float(dom[1]-dom[0])/2*(x+1)+dom[0];
    w *= float(dom[1]-dom[0])/2;
    return x, w

def RSMat(Ntarg,Nsrc,typtarg=1,typsrc=2,start=None,end=None):
    """
    Resampling matrix. This function computes the resampling matrix from 
    Nsrc points of type typsrc to Ntarg points of type typtarg. 
    """
    if typsrc==2:
        ths=th2(Nsrc);
    elif typsrc==1:
        ths=th1(Nsrc);
    if typtarg==1:
        tht=th1(Ntarg);
    elif typtarg==2:
        tht=th2(Ntarg);
    elif typtarg=='u':
        tht=thu(Ntarg);
    elif typtarg==232:
        tht=th2pan1(Ntarg);
    elif typtarg=='cl':
        tht=thu(Ntarg,start,end);
    Lsrc = Lmat(Nsrc,ths);
    Ltarg = Lmat(Nsrc,tht);
    RS = np.dot(Ltarg,np.linalg.inv(Lsrc));
    return RS
    
def diffMat(numDs,dom,Ntarg,Nsrc,typtarg,typsrc):
    """
    Differentiation matrix. Inputs are the number of sources 
    and the type of grid typsrc. The differentiation matrix is
    computed (numDs derivatives) at Ntarg target points on a grid
    of type typtarg. The answer is scaled by the domain size, which
    comes from dom. 
    The differentiation matrix is returned.
    """
    if typsrc==2:
        ths=th2(Nsrc);
    elif typsrc==1:
        ths=th1(Nsrc);
    if typtarg==1:
        tht=th1(Ntarg);
    elif typtarg==2:
        tht=th2(Ntarg);
    Lsrc = Lmat(Nsrc,ths);
    Ltarg = Lmat(Nsrc,tht);
    # Compute coefficients in the basis
    CfromD=np.linalg.inv(Lsrc);
    # Differentiate the Chebyshev series numDs times
    for iD in range(numDs):
        CfromD=coeffDiff(CfromD,Nsrc);
    # Multiply by the matrix of values at the targets
    dMat = np.dot(Ltarg,CfromD);
    J = float(dom[1]-dom[0])/2;
    return dMat/(J**numDs);
        
def coeffDiff(coefs,N):
    """
    Differentiate Chebyshev coefficients using the recursive
    formula. Input: coefficints (an N x ?) array, N = number of
    coefficients
    Outputs: coefficients of the derivative series.
    """
    Dcos = 0*coefs;
    Dcos[N-2,:]=2.0*(N-1)*coefs[N-1,:];
    for a in range(N-3,-1,-1):
        Dcos[a,:]=Dcos[a+2,:]+2.0*(a+1)*coefs[a+1,:];
    Dcos[0,:]=Dcos[0,:]/2;
    return Dcos;

def intMat(f,N,dom):
    """
    Function that applies the Chebyshev integration matrix to the series of
    coefficients on domain dom.
    """
    fint=0*f;
    # just a convention to make the unknown constant 0
    fint[1,:]=f[0,:]-0.5*f[2,:];
    for j in range(2,N-1):
        fint[j,:]=1.0/(2*j)*(f[j-1,:]-f[j+1,:]);
    fint[N-1,:]=1.0/(2*(N-1))*f[N-2,:];
    fint*=float(dom[1]-dom[0])/2;
    return fint;

def th2(N):
    """
    Theta values for a type 2 Chebyshev grid with N pts
    """
    th=np.flipud(pi*np.arange(N))/(N-1);
    return th;

def th1(N):
    """
    Theta values for a type 1 Chebyshev grid with N pts
    """
    th=np.flipud((2.0*np.arange(N)+1))*pi/(2*N);
    return th;

def thu(N,start=-1.0,end=1.0):
    """
    Theta values for a uniform grid with N pts
    """
    xu = np.linspace(start,end,N);
    thu=np.arccos(xu);
    return thu;

def th2pan1(N):
    """
    Theta values for 2 panels of type 1 N points each
    """
    x1p = (np.cos(th1(N))+1)/2.0;
    th2p2 = np.arccos(np.concatenate((x1p-1,x1p)));
    return th2p2;

def Lmat(N,th):
    """ 
    The matrix that maps coefficients to values for N 
    points. The other input are the theta values.
    """
    L = cos(np.outer(th,np.arange(N)));
    return L;

def coeffs(N,th,data):
    """
    Compute the coefficients of the Chebyshev series
    from the data. 
    Inputs = angles theta (N vector), number of points N, 
    and the data to compute the series of (an N x ? vector).
    Output: the coefficients (as an N x ? vector).
    """
    cos = np.linalg.solve(Lmat(N,th),data);
    return cos;

def evalSeries(coeffs,th,N):
    """
    Evaluate some number of Chebyshev series at an angle theta.
    Inputs: coeffs = coefficients of the series (array may
    columns, but has N rows). th = angle theta to evaluate the 
    series at, N = number of series coefficients.
    Output: value(s) of the Chebyshev series at th
    """
    polys = np.reshape(np.cos(np.arange(N)*th),(1,N));
    vals = np.dot(polys,coeffs);
    return vals;


