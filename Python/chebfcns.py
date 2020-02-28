import numpy as np
from numpy import pi
from numpy import cos
import numba as nb

"""
    This file is a list of Chebyshev related functions
    for python that are needed in the slender fiber 
    code
"""

def chebPts(N,dom,kind,numPanels=1):
    """
    Chebyshev points on the interval dom of kind kind. 
    Second kind Chebyshev points include the endpoints, 
    Inputs: number of points and size of domain (a 1 x 2
    array), and kind of points.
    Outputs: the Chebyshev points x.
    """
    x = cos(theta(N,kind,numPanels));
    # Rescale
    x = float(dom[1]-dom[0])/2*(x+1)+dom[0];
    return x;

def chebWts(N,dom,kind):
    """
    Chebyshev integration weights on the interval dom of kind kind. 
    Inputs: number of points, and size of domain (1 x 2 array),
    kind of points (only 1 and 2 supported). 
    Outputs: the weights as an N array
    """
    if kind==1:
        m=2.0 / np.concatenate(([1],1-np.arange(2,N,2)**2));
        if np.mod(N,2):
            c=np.concatenate((m,-m[(N+1)//2-1:0:-1]));
        else:
            c=np.concatenate((m,[0],-m[N//2-1:0:-1]))
        v=np.exp(1j*np.arange(N)*np.pi/N);
        c=c*v;
        w = np.real(np.fft.ifft(c));
    elif kind==2:
        c=2.0/np.concatenate(([1],1-np.arange(2,N,2)**2));
        c=np.concatenate((c,c[N//2-1:0:-1]));
        w = np.real(np.fft.ifft(c));
        w[0]/=2;
        w=np.concatenate((w,[w[0]]))
    else:
        raise ValueError('Invalid kind of points for Chebyshev weights: 1 or 2 supported')
    w *= float(dom[1]-dom[0])/2;
    return w;

def theta(N,type,numPanels):
    """
    Theta values for a Chebyshev grid with N pts and 
    type "type" (1, 2, or 'u' for uniform points).
    Multiple panels are allowed; although only type 1
    points are supported with multiple panels.
    """
    if (type==2):
        th=np.flipud(pi*np.arange(N))/(N-1);
    elif (type==1):
        th=np.flipud((2.0*np.arange(N)+1))*pi/(2*N);
    elif (type=='u'):
        xu = np.linspace(-1.0,1.0,N);
        th=np.arccos(xu);
    else:
        raise ValueError('Invalid Chebyshev grid type; 1, 2 or uniform are supported');
    if (numPanels > 1):
        if (type!=1):
            raise ValueError('Cannot have more than 1 panel with anything but type 1 grid')
        x=cos(th);
        panSize = 2.0/numPanels;
        panStarts = np.arange(numPanels)*panSize-1;
        EachPan = (x+1)*panSize/2.0;
        allpans = np.reshape(np.array([panStarts]).T+EachPan,numPanels*N);
        th = np.arccos(allpans);
    return th;

def CoeffstoValuesMatrix(Ncoefs,Ntarg,typetarg,nPantarg=1):
    """ 
    The matrix that maps Ncoefs coefficients to Ntarg values on a typetarg grid.
    Optional argument is to have multiple panels on the target.
    """
    thtarg = theta(Ntarg,typetarg,nPantarg);
    CtoVMat = cos(np.outer(thtarg,np.arange(Ncoefs)));
    return CtoVMat;

def diffMat(numDs,dom,Ntarg,Nsrc,typtarg,typsrc):
    """
    Differentiation matrix. Inputs are the number of source points
    and the type of source grid typsrc. The differentiation matrix is
    computed (numDs derivatives) at Ntarg target points on a grid
    of type typtarg. The answer is scaled by the domain size, which
    comes from dom (a 2 array).
    The differentiation matrix is returned.
    """
    CtoV_src = CoeffstoValuesMatrix(Nsrc,Nsrc,typsrc); #square matrix
    CtoV_targ = CoeffstoValuesMatrix(Nsrc,Ntarg,typtarg); # rectangular matrix Ntarg x Nsrc
    # Compute coefficients in the basis
    VtoC_src=np.linalg.inv(CtoV_src);
    # Differentiate the Chebyshev series numDs times
    for iD in range(numDs):
        VtoC_src=diffCoefficients(VtoC_src,Nsrc);
    # Multiply by the matrix of values at the targets
    NderivMat = np.dot(CtoV_targ,VtoC_src);
    Jacobian = 0.5*(dom[1]-dom[0]);
    return NderivMat/(Jacobian**numDs);

@nb.njit(nb.float64[:,:](nb.float64[:,:],nb.int64))
def diffCoefficients(coefs,N):
    """
    Differentiate Chebyshev coefficients using the recursive
    formula. Input: coefficients (an N x ?) array, N = number of
    coefficients
    Outputs: coefficients of the derivative series.
    """
    Dcos = np.zeros(coefs.shape);
    Dcos[N-2,:]=2.0*(N-1)*coefs[N-1,:];
    for a in range(N-3,-1,-1):
        Dcos[a,:]=Dcos[a+2,:]+2.0*(a+1)*coefs[a+1,:];
    Dcos[0,:]=0.5*Dcos[0,:];
    return Dcos;

def intCoefficients(incoefs,N,dom):
    """
    Function that applies the Chebyshev integration matrix to the series of
    N coefficients incoefs on domain dom.
    There is an unknown constant since this is indefinite integration
    that we set to zero here.
    """
    intcoefs = np.zeros(incoefs.shape);
    intcoefs[1,:]=incoefs[0,:]-0.5*incoefs[2,:];
    for j in range(2,N-1):
        intcoefs[j,:]=1.0/(2*j)*(incoefs[j-1,:]-incoefs[j+1,:]);
    intcoefs[N-1,:]=1.0/(2*(N-1))*incoefs[N-2,:];
    intcoefs*=0.5*(dom[1]-dom[0]);
    return intcoefs;

def ResamplingMatrix(Ntarg,Nsrc,typtarg,typsrc,nPantarg=1,nPansrc=1):
    """
    Resampling matrix. This function computes the resampling matrix from 
    Nsrc points of type typsrc to Ntarg points of type typtarg. 
    Number of panels for source and target are keyword arguments.
    """
    CtoV_src = CoeffstoValuesMatrix(Nsrc,Nsrc,typsrc,nPansrc);
    CtoV_targ = CoeffstoValuesMatrix(Nsrc,Ntarg,typtarg,nPantarg);
    ResamplingMat = np.dot(CtoV_targ,np.linalg.inv(CtoV_src));
    return ResamplingMat

def evalSeries(coeffs,th):
    """
    Evaluate some number of Chebyshev series at an angle theta.
    Inputs: coeffs = coefficients of the series (array may have
    multiple columns, but has N rows). th = angle theta to evaluate 
    the series at (can be complex). 
    Output: value(s) of the Chebyshev series at th
    """
    N,_= coeffs.shape;
    polys = np.cos(np.outer(np.arange(N),th)).T;
    vals = np.dot(polys,coeffs);
    return vals;


