import numpy as np
import SpecQuadUtils as sq

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

def correctVel(tpt,fiber,fibpts,forces,forceDs,centerVels,method):
    """
    Method to correct the velocity when fibers are close together. 
    Inputs: tpt = target point (3 array), fiber = the fiber object that
    is close to the target, fibpts = Npts x 3 array of the (shifted)
    fiber locations that are close to the target, forces = the forces on the 
    fiber (not force densities; these are the force densities*weights), 
    forceDs = Npts x 3 array of force densities on the fiber, 
    centerVels = nPts x 3 array of the velocity on the fiber centerline
    (for use when points get really close to the fiber), and 
    the correction method (1 for nptsUpsample direct quadrature), otherwise this will 
    do special quadrature 
    Outputs: the correction to the velocity at the target as a 3 array
    """
    # Subtract the Ewald with Npf points on the fiber
    cvel = -fiber.RPYSBTKernel(tpt,fibpts,forces);
    # If doing upsampling, upsample, do the free space quad sum, and stop
    nUpsample = fiber.getNumUpsample();
    wup = fiber.getUpsampledWeights();
    Xup = fiber.upsampleGlobally(fibpts);
    fDup = fiber.upsampleGlobally(forceDs);
    cvelup = fiber.upsampleGlobally(centerVels);
    forcesup = fDup*np.reshape(wup,(nUpsample,1));
    if (method==1): # free space sum for N = nptsUpsample (32)
        cvel+= fiber.RPYSBTKernel(tpt,Xup,forcesup,sbt=1);
        return cvel;
    # If doing special quad, find the complex root using the domain for centerline
    # t in [-1,1]
    troot, converged, cdist, clvel = calcRoot(tpt,fiber,Xup,cvelup,nUpsample);
    # If not converged, point must be far, just do direct with N = nUpsample and stop
    if (not converged):
        cvel+= fiber.RPYSBTKernel(tpt,Xup,forcesup,sbt=1);
        return cvel;
    # Now we know we have converged to the root
    # How far are we from the fiber in a non-dimensional sense?
    epsilon, Lf = fiber.getepsilonL();
    dstar = cdist/(epsilon*Lf);
    # If the point is inside the fiber "cross section," which we define as
    # dstar < dstarCenterLine, compute the centerline velocity at the approximate
    # closest point and return (in fact we need it when cdist/(epsilon*L) < dstarInterpolate;
    # in that case we are going to interpolate. How this is implemented is to set
    # wtCL > 0 if we need to interpolate
    wtCL = 0.0;
    if (dstar < dstarInterpolate):
        wtCL = (dstarInterpolate-dstar)/dstarCenterLine;
    wtSBT = 1.0-wtCL;
    if (dstar < dstarCenterLine): # return the centerline velocity and stop
        print('Target close to fiber - setting velocity = centerline velocity')
        cvel+= clvel;
        return cvel;
    # Now we are dealing with the case where we need the free space slender integral.
    # 3 possible options:
    # 1. Don't need special quad (determined by Bernstein radius)
    # 2. Special quad needed, but 1 panel of 32 is ok
    # 3. Special quad needed, but need 2 panels of 32
    # Bernstein radius
    bradius = bernstein_radius(troot);
    sqneeded = (bradius < rho_crit);
    if (not sqneeded):
        cvel+= wtSBT*fiber.RPYSBTKernel(tpt,Xup,forcesup,sbt=1)+wtCL*clvel;
        return cvel;
    specNodes = fiber.getSpecialQuadNodes();
    if (dstar > dstar2panels): # Ok to proceed with 1 panel of nptsUpsample
        # Special quad weights
        wts = specialWts(fiber,troot,nUpsample);
        cvel+= fiber.SBTKernelSplit(tpt,Xup,fDup,wts[:,0],wts[:,1],wts[:,2])
        return cvel;
    # dstar < dstar2panels, need to redo special quad with 2 panels of 32
    # Resample at 2 panels of 32
    X2pan = fiber.upsample2Panels(fibpts);
    f2pan = fiber.upsample2Panels(forceDs);
    SBTvel = np.zeros(3);
    for iPan in range(2): # looping over the panels
        indpan = np.arange(nUpsample)+iPan*nUpsample;
        # Points and force densities for the panel
        Xpan = X2pan[indpan,:];
        fdpan = f2pan[indpan,:];
        # Calculate the root as before. The method will waste time
        # computing the closest point, etc, but those are trivial computations
        # and it's easier for now to put it all in one method.
        troot, converged, _, _ = calcRoot(tpt,fiber,Xpan,cvelup,nUpsample);
        bradius = bernstein_radius(troot);
        # Do we need special quad for this panel? (Will only need for 1 panel,
        # whatever one has the fiber section closest to the target).
        sqneeded = converged and (bradius < rho_crit);
        if (not sqneeded):
            # Directly do the integral for 1 panel (weights have to be halved because
            # we cut the fiber in 2)
            SBTvel+=fiber.RPYSBTKernel(tpt,Xpan,fdpan*\
                                np.reshape(wup,(nUpsample,1))/2.0,sbt=1);
        else:
            # Compute special quad weights (divide weights by 2 for 2 panels)
            wts = specialWts(fiber,troot,nUpsample)/2.0;
            SBTvel+= fiber.SBTKernelSplit(tpt,Xpan,fdpan,wts[:,0],wts[:,1],wts[:,2]);
    cvel+= wtSBT*SBTvel+wtCL*clvel;
    return cvel;

def calcRoot(tpt,fiber,X,centerVels,nUpsample):
    """
    Compute the root troot using special quadrature. 
    Inputs: tpt = 3 array target point where we seek the velocity due to the fiber, 
    fiber = fiber object we seek the velocity due to, X = nptsUpsample x 3 array of fiber points,
    centerVels = nptsUpsample x 3 array of velocities on the fiber centerline (for close points).
    Outputs: troot = complex root for special quad, converged = 1 if we actually
    converged to that root, 0 if we didn't (or if the initial guess was so far that
    we didn't need to), cdist = approximate distance from the fiber centerline 
    to the point, clvel_close = 3 array of velocity at the closest centerline point
    """
    # Initial guess (C++ function)
    specNodes = fiber.getSpecialQuadNodes();
    tinit = sq.rf_iguess(specNodes,X[:,0],X[:,1],X[:,2],nUpsample,\
                            tpt[0],tpt[1],tpt[2]);
    # If initial guess is too far, return so we can do direct with N = 32
    if (bernstein_radius(tinit) > 1.5*rho_crit):
        converged=0;
        troot=0+0j;
        cdist = 1000; # a big number
        return troot, converged, cdist, np.zeros(3);
    # Compute the Chebyshev coefficients and the coefficients of the derivative
    pos_coeffs, deriv_cos = fiber.upsampledCoefficients(X);
    # From Ludvig: cap the expansion at 16 coefficients for root finding (this seems
    # to be more stable near the boundary). Use this expansion to compute the root
    # (C++ code). Returns a tuple with elements (troot, converged).
    pos_coeffsRfind = pos_coeffs[:rootfinderExpansionCap,:];
    deriv_cosRfind = deriv_cos[:rootfinderExpansionCap,:];
    trconv = sq.rootfinder(pos_coeffsRfind[:,0],pos_coeffsRfind[:,1],pos_coeffsRfind[:,2],\
                deriv_cosRfind[:,0],deriv_cosRfind[:,1],deriv_cosRfind[:,2],rootfinderExpansionCap, \
                tpt[0], tpt[1], tpt[2],tinit);
    troot = trconv[0];
    converged = trconv[1];
    # Use troot to estimate the distance from the fiber
    tapprox = troot;
    if (np.real(troot) < -1): # root is off the fiber centerline in tangental dir
        tapprox = -1.0+1j*np.imag(tapprox);
    elif (np.real(troot) > 1): # root is off the fiber centerline in tangental dir
        tapprox = 1.0+1j*np.imag(tapprox);
    # Evaluate the Chebyshev series at tapprox
    cdist = np.linalg.norm(np.real(fiber.evaluatePosition(tapprox,pos_coeffs)-tpt));
    # Evaluate centerline velocity at tapprox
    clv_cos, _ = fiber.upsampledCoefficients(centerVels);
    clvel_close = np.reshape(fiber.evaluatePosition(np.real(tapprox),clv_cos),3);
    return troot, converged, cdist, clvel_close

def specialWts(fiber,troot,nUpsample):
    """
    Weights for the special quadrature scheme. 
    Inputs: fiber object that is contributing to velocity at the target, 
    complex root troot. 
    Outputs: the quadrature weights as a nptsUpsample x 3 array. The first column has the weights
    for the R^-1 integral, second column R^-3, third column R^-5.
    """
    # Call the C++ code to get the integrals for the root
    integrals = np.reshape(sq.spec_ints(troot,nUpsample),(3,nUpsample)).T;
    # Fiber special weights from integrals
    wts = fiber.specialWeightsfromIntegrals(integrals,troot);
    return wts;

def bernstein_radius(z):
    return np.abs(z + np.sqrt(z - 1.0)*np.sqrt(z+1.0));
