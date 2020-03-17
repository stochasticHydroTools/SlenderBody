import numpy as np
import SpecQuadUtils as sq
import EwaldNumba as ewNum
import EwaldUtils as ewc

# Definitions
dstarCenterLine = 2.2; # dstarCenterLine*eps*L is distance where we set v to centerline v
dstarInterpolate = 4.4; # dstarInterpolate*eps*L is distance where we start blending centerline
                        # and quadrature result
dstar2panels = 8.8; #dstar2panels*eps*L is distance below which we need 2 panels for special quad
rootfinderExpansionCap = 16; # number of coefficients for root finder
rho_crit = 1.114; # critical Bernstein radius for special quad

""" 
These are the functions for the target and fiber velocity corrections, 
meaning it handles all of the special quadrature stuff. This class
only does free space corrections, it therefore DOES NOT need geometric
information.
"""


def SpecQuadVel(Ntarg,tpts,fiber,Xup,X2pan,forcesUp,fDup,f2pan,centerVels,cvelup,\
                wup,epsilon,Lf,aRPY,mu,N,nUpsample):
    """
    In progress. 
    """
    SBTvels = np.zeros((Ntarg,3));
    # Find the complex root using the domain for centerline t in [-1,1]
    troots, converged, cdists, clvels = calcRoots(Ntarg,tpts,fiber,Xup,cvelup,N,nUpsample);
    # Separate targets into those needing special and not needing special
    allInds = np.arange(Ntarg);
    DirectInds = allInds[converged==0];
    SpecialInds = allInds[converged > 0];
    # Now we know we have converged to the root
    # How far are we from the fiber in a non-dimensional sense?
    dstars = np.zeros(Ntarg);
    dstars[DirectInds] = float("inf");
    dstars[SpecialInds] = cdists[SpecialInds]/(epsilon*Lf);
    # If the point is inside the fiber "cross section," which we define as
    # dstar < dstarCenterLine, compute the centerline velocity at the approximate
    # closest point and return (in fact we need it when cdist/(epsilon*L) < dstarInterpolate;
    # in that case we are going to interpolate. How this is implemented is to set
    # wtCL > 0 if we need to interpolate
    wtsCL = np.zeros(Ntarg);
    wtsCL[SpecialInds]+= np.logical_and(dstars[SpecialInds] < dstarInterpolate,dstars[SpecialInds] > dstarCenterLine)\
            *(dstarInterpolate-dstars[SpecialInds])/(dstarInterpolate-dstarCenterLine);  # points that get an interpolation 
    wtsCL[SpecialInds]+= (dstars[SpecialInds] < dstarCenterLine)*1.0; # points that get only centerline vel
    wtsSBT = 1.0-wtsCL;
    # Append to directInds the ones that don't need special quad
    bradius = bernstein_radius(troots);
    sqneeded = np.logical_and(converged,bradius < rho_crit);
    AllDirectInds = allInds[np.logical_not(sqneeded)]; 
    # Do direct ones with direct
    SBTvels[AllDirectInds,:]+=np.reshape(ewc.RPYSBTKernel(len(AllDirectInds),tpts[AllDirectInds,0],\
        tpts[AllDirectInds,1],tpts[AllDirectInds,2],nUpsample,Xup[:,0],Xup[:,1],Xup[:,2],\
        forcesUp[:,0],forcesUp[:,1],forcesUp[:,2],mu,aRPY,1),(len(AllDirectInds),3));
    # Now we are dealing with the case where we need special quad.
    # 2 possible options: 1 panel of 32 is ok or need 2 panels of 32
    specNodes = fiber.getSpecialQuadNodes();
    OnePanelInds = allInds[np.logical_and(sqneeded,dstars > dstar2panels)];
    TwoPanelsInds = allInds[np.logical_and(sqneeded,dstars <= dstar2panels)];
    for iOnePan in OnePanelInds: 
        # Special quad weights for single panel
        wts = specialWts(nUpsample,specNodes,troots[iOnePan],Lf);
        SBTvels[iOnePan,:]+= ewNum.SBTKSpl(tpts[iOnePan,:],nUpsample,Xup,fDup,\
                    mu,epsilon,Lf,wts[:,0],wts[:,1],wts[:,2]);
    for iTwoPan in TwoPanelsInds:
        # Special quad w 2 panels
        for iPan in range(2): # looping over the panels
            indpan = np.arange(nUpsample)+iPan*nUpsample;
            # Points and force densities for the panel
            Xpan = X2pan[indpan,:];
            fdpan = f2pan[indpan,:];
            # Calculate the root as before. The method will waste time
            # computing the closest point, etc, but those are trivial computations
            # and it's easier for now to put it all in one method.
            tr, conv, _, _ = calcRoots(1,np.array([tpts[iTwoPan,:]]),fiber,Xpan,cvelup,nUpsample,nUpsample);
            br = bernstein_radius(tr);
            # Do we need special quad for this panel? (Will only need for 1 panel,
            # whatever one has the fiber section closest to the target).
            sqneeded = conv and (br < rho_crit);
            if (not sqneeded):
                # Directly do the integral for 1 panel (weights have to be halved because
                # we cut the fiber in 2)
                forcePan = fdpan*np.reshape(wup,(nUpsample,1))/2.0;
                SBTvels[iTwoPan,:]+= np.reshape(ewNum.RPYSBTK(1,np.array([tpts[iTwoPan,:]]),nUpsample,\
                        Xpan,forcePan,mu,aRPY,sbt=1),3);
            else:
                # Compute special quad weights (divide weights by 2 for 2 panels)
                wts = specialWts(nUpsample,specNodes,tr,Lf)/2.0;
                SBTvels[iTwoPan,:]+= ewNum.SBTKSpl(tpts[iTwoPan,:],nUpsample,Xpan,fdpan,mu,epsilon,\
                            Lf,wts[:,0],wts[:,1],wts[:,2]);  
    return np.reshape(wtsSBT,(Ntarg,1))*SBTvels+np.reshape(wtsCL,(Ntarg,1))*clvels;

def SpecQuadVel2(Ntarg,tpts,fiber,X2pan,f2pan,wup,dstars,epsilon,Lf,aRPY,mu,nUpsample):
    """
    In progress. 
    """
    SBTvels = np.zeros((Ntarg,3));
    # How far are we from the fiber in a non-dimensional sense?
    dstars = dstars/(epsilon*Lf);
    # If the point is inside the fiber "cross section," which we define as
    # dstar < dstarCenterLine, compute the centerline velocity at the approximate
    # closest point and return (in fact we need it when cdist/(epsilon*L) < dstarInterpolate;
    # in that case we are going to interpolate. How this is implemented is to set
    # wtCL > 0 if we need to interpolate
    wtsCL = np.zeros(Ntarg);
    wtsCL+= np.logical_and(dstars < dstarInterpolate,dstars > dstarCenterLine)\
            *(dstarInterpolate-dstars)/(dstarInterpolate-dstarCenterLine);  # points that get an interpolation 
    wtsCL+= (dstars < dstarCenterLine)*1.0; # points that get only centerline vel
    wtsSBT = 1.0-wtsCL;
    # Now we are dealing with the case where we need special quad.
    # 2 possible options: 1 panel of 32 is ok or need 2 panels of 32
    specNodes = fiber.getSpecialQuadNodes();
    for iTwoPan in range(Ntarg):
        # Special quad w 2 panels
        for iPan in range(2): # looping over the panels
            indpan = np.arange(nUpsample)+iPan*nUpsample;
            # Points and force densities for the panel
            Xpan = X2pan[indpan,:];
            fdpan = f2pan[indpan,:];
            # Calculate the root as before. The method will waste time
            # computing the closest point, etc, but those are trivial computations
            # and it's easier for now to put it all in one method.
            tr, conv = calcRoot(np.array([tpts[iTwoPan,:]]),fiber,Xpan,nUpsample);
            br = bernstein_radius(tr);
            # Do we need special quad for this panel? (Will only need for 1 panel,
            # whatever one has the fiber section closest to the target).
            sqneeded = conv and (br < rho_crit);
            if (not sqneeded):
                # Directly do the integral for 1 panel (weights have to be halved because
                # we cut the fiber in 2)
                forcePan = fdpan*np.reshape(wup,(nUpsample,1))/2.0;
                SBTvels[iTwoPan,:]+= np.reshape(ewNum.RPYSBTK(1,np.array([tpts[iTwoPan,:]]),nUpsample,\
                        Xpan,forcePan,mu,aRPY,sbt=1),3);
            else:
                # Compute special quad weights (divide weights by 2 for 2 panels)
                wts = specialWts(nUpsample,specNodes,tr,Lf)/2.0;
                SBTvels[iTwoPan,:]+= ewNum.SBTKSpl(tpts[iTwoPan,:],nUpsample,Xpan,fdpan,mu,epsilon,\
                            Lf,wts[:,0],wts[:,1],wts[:,2]);  
    return np.reshape(wtsSBT,(Ntarg,1))*SBTvels;

def calcRoots(Ntarg,tpts,fiber,X,centerVels,Ncos,nUpsample):
    """
    Compute the root troot using special quadrature. 
    Inputs: Ntarg = number of targets, tpt = Ntarg x 3 array target point 
    where we seek the velocity due to the fiber, 
    fiber = fiber object we seek the velocity due to, X = nptsUpsample x 3 array of fiber points,
    centerVels = nptsUpsample x 3 array of velocities on the fiber centerline (for close points).
    Outputs: troot = complex root for special quad, converged = 1 if we actually
    converged to that root, 0 if we didn't (or if the initial guess was so far that
    we didn't need to), cdist = approximate distance from the fiber centerline 
    to the point, clvel_close = 3 array of velocity at the closest centerline point
    """
    tinits = np.zeros(Ntarg,dtype=np.complex128);
    converged = -np.ones(Ntarg,dtype=int);
    troots = np.zeros(Ntarg,dtype=np.complex128);
    cdists = 1000*np.ones(Ntarg);
    cvels_close = np.zeros((Ntarg,3));
    # Initial guess (C++ function)
    specNodes = fiber.getSpecialQuadNodes();
    # C++ function to compute the roots
    tinits = np.array(sq.rf_iguess(specNodes,X[:,0],X[:,1],X[:,2],nUpsample,\
                        tpts[:,0],tpts[:,1],tpts[:,2],Ntarg));
    # If initial guess is too far, do nothing
    converged[bernstein_radius(tinits) > 1.5*rho_crit] = 0;
    # Now filter to only do targets with tinit close enough
    TargNums = np.arange(Ntarg);
    TargNumsLeft = TargNums[converged < 0];
    remainingTargs = tpts[converged < 0];
    # Compute the Chebyshev coefficients and the coefficients of the derivative
    pos_coeffs, deriv_cos = fiber.upsampledCoefficients(X);
    # C++ function to compute the roots and convergence
    trconv = sq.rootfinder(pos_coeffs[:,0],pos_coeffs[:,1],pos_coeffs[:,2],\
                deriv_cos[:,0],deriv_cos[:,1],deriv_cos[:,2],Ncos, \
                tpts[TargNumsLeft,0], tpts[TargNumsLeft,1], tpts[TargNumsLeft,2],len(TargNumsLeft),\
                tinits[TargNumsLeft]);
    troots[TargNumsLeft] = np.array(trconv[0]);
    converged[TargNumsLeft] = np.array(trconv[1]);
    # Use troot to get the approximate closest point on the fiber to the target, 
    # which I call tapprox
    tapprox = np.real(troots.copy());
    # For roots off the fiber centerline in tangental dir, push them to the endpoints
    # of the fiber
    tapprox[tapprox < -1.0] = -1.0;
    tapprox[tapprox > 1.0] = 1.0;
    # Estimate approximate distances
    cdistVectors = fiber.evaluatePosition(tapprox,pos_coeffs)-tpts;
    cdists = np.linalg.norm(cdistVectors,axis=1);
    # Evaluate centerline velocity at tapprox
    clv_cos, _ = fiber.upsampledCoefficients(centerVels);
    clvel_close = fiber.evaluatePosition(tapprox,clv_cos);
    return troots, converged, cdists, clvel_close

def calcRoot(tpt,fiber,X,nUpsample):
    """
    Compute the root troot using special quadrature. 
    Inputs: Ntarg = number of targets, tpt = Ntarg x 3 array target point 
    where we seek the velocity due to the fiber, 
    fiber = fiber object we seek the velocity due to, X = nptsUpsample x 3 array of fiber points,
    Outputs: troot = complex root for special quad, converged = 1 if we actually
    converged to that root, 0 if we didn't (or if the initial guess was so far that
    we didn't need to), cdist = approximate distance from the fiber centerline 
    to the point, clvel_close = 3 array of velocity at the closest centerline point
    """
    # Initial guess (C++ function)
    specNodes = fiber.getSpecialQuadNodes();
    # C++ function to compute the roots
    tinit = np.array(sq.rf_iguess(specNodes,X[:,0],X[:,1],X[:,2],nUpsample, tpt[:,0],tpt[:,1],tpt[:,2],1));
    # If initial guess is too far, do nothing
    if (bernstein_radius(tinit) > 1.5*rho_crit):
        return 0+0*1j, 0;
    # Compute the Chebyshev coefficients and the coefficients of the derivative
    pos_coeffs, deriv_cos = fiber.upsampledCoefficients(X);
    # From Ludvig: cap the expansion at 16 coefficients for root finding (this seems
    # to be more stable near the boundary). Use this expansion to compute the root
    # (C++ code). Returns a tuple with elements (troot, converged).
    pos_coeffsRfind = pos_coeffs[:rootfinderExpansionCap,:];
    deriv_cosRfind = deriv_cos[:rootfinderExpansionCap,:];
    # C++ function to compute the roots and convergence
    trconv = sq.rootfinder(pos_coeffsRfind[:,0],pos_coeffsRfind[:,1],pos_coeffsRfind[:,2],\
                deriv_cosRfind[:,0],deriv_cosRfind[:,1],deriv_cosRfind[:,2],rootfinderExpansionCap, \
                tpt[:,0], tpt[:,1], tpt[:,2],1,tinit);
    troot = np.reshape(np.array(trconv[0]),1);
    converged = np.reshape(np.array(trconv[1]),1);
    return troot, converged;

def specialWts(nUpsample,tnodes,troot,Lf):
    """
    Weights for the special quadrature scheme. 
    Inputs: fiber object that is contributing to velocity at the target, 
    complex root troot. 
    Outputs: the quadrature weights as a nptsUpsample x 3 array. The first column has the weights
    for the R^-1 integral, second column R^-3, third column R^-5.
    """
    # Call the C++ code to get the integrals for the root
    wts = np.reshape(sq.specialWeights(nUpsample,tnodes,troot,Lf),(3,nUpsample)).T;
    # Fiber special weights from integrals
    return wts;

def bernstein_radius(z):
    return np.abs(z + np.sqrt(z - 1.0)*np.sqrt(z+1.0));
