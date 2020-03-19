import numpy as np
import copy

# Definitions (for fixed point iteration)
itercap = 1;
fixedpointtol=1e-6;

class TemporalIntegrator(object):
    """
    Class to do the temporal integration. 
    Child classes: BackwardEuler and CrankNicolsonLMM
    Abstract class: does first order explicit
    """
    
    def __init__(self,fibCol,CLNetwork):
        self._allFibers = fibCol;
        self._CLNetwork = CLNetwork;
        self._impco = 0; #coefficient for linear solves in FibCollocationDiscretization.alphaLamSolve
    
    def getXandXsNonLoc(self):
        return self._allFibers.getX().copy(), self._allFibers.getXs().copy();

    def getLamNonLoc(self,iT,iters):
        return self._allFibers.getLambdas().copy();

    def getMaxIters(self,iT):
        if (iT==0):
            return itercap;
        return 1;

    def gettval(self,iT,dt):
        return iT*dt;
    
    def getFirstNetworkStep(self,dt):
        return dt;
    
    def getSecondNetworkStep(self,dt):
        return 0;
        
    def NetworkUpdate(self,Dom,t,tstep):
        Dom.setg(self._allFibers.getg(t));
        self._CLNetwork.updateNetwork(self._allFibers,Dom,tstep);
    
    def getLambdaN(self,iT):
        return self._allFibers.getLambdas();

    def updateAllFibers(self,iT,dt,Dom,Ewald,outfile,write=1):
        """
        The main update method. 
        Inputs: the timestep number as iT, the timestep dt, the domain
        object we are solving on, Dom, Ewald = the Ewald splitter object
        we need to do the Ewald calculations, outfile = handle to the 
        output file to write to, write=1 to write to it, 0 otherwise.  
        """           
        # Update the network (first time)
        self.NetworkUpdate(Dom,iT*dt,self.getFirstNetworkStep(dt));
        # Set domain strain and fill up the arrays of points
        tvalSolve = self.gettval(iT,dt);
        Dom.setg(self._allFibers.getg(tvalSolve));
        converged = 0;
        iters = 0;
        itmax = self.getMaxIters(iT);
        # Get relevant arguments
        XforNL, XsforNL = self.getXandXsNonLoc();
        forceExt = self._CLNetwork.CLForce(self._allFibers._fiberDisc,XforNL,Dom);
        while (not converged and iters < itmax):
            lamforNL = self.getLamNonLoc(iT,iters);  
            uNonLoc = self._allFibers.nonLocalBkgrndVelocity(XforNL,XsforNL,lamforNL, tvalSolve, \
                        forceExt,Dom,Ewald);
            self._allFibers.linSolveAllFibers(XsforNL,uNonLoc,forceExt,dt,self._impco);
            converged = self._allFibers.converged(lamforNL,fixedpointtol);
            iters+=1;
        # Converged, update the fiber positions
        self._allFibers.updateAllFibers(dt,XsforNL,exactinex=1);
        # Copy individual fiber objects into large arrays of points, forces
        self._allFibers.fillPointArrays();
        # Update the network (second time)
        self.NetworkUpdate(Dom,(iT+1)*dt,self.getSecondNetworkStep(dt));
        if (write):
            self._allFibers.writeFiberLocations(outfile);
        # Compute stress at time n+1
        #fibStress = self._allFibers.FiberStress(self.getLambdaN(iT),Dom.getLens());
        #CLstress = self._CLNetwork.CLStress(self._allFibers,self._allFibers.getFiberDisc(),Dom);
        #return fibStress[1,0]+CLstress[1,0];

class BackwardEuler(TemporalIntegrator):
 
    def __init__(self,fibCol,CLNetwork):
        self._allFibers = fibCol;
        self._CLNetwork = CLNetwork;
        self._impco = 1.0; # coefficient for implicit solves

class CrankNicolsonLMM(TemporalIntegrator):
    
    def __init__(self,fibCol,CLNetwork):
        self._allFibers = fibCol;
        self._CLNetwork = CLNetwork;
        self._allFibers.fillPointArrays();
        self._allFibersPrev = copy.deepcopy(fibCol);
        self._impco = 0.5; # coefficient for implicit solves
      
    def getXandXsNonLoc(self):
        X = 1.5*self._allFibers.getX() - 0.5*self._allFibersPrev.getX();
        Xs = 1.5*self._allFibers.getXs() - 0.5*self._allFibersPrev.getXs();
        return X, Xs;

    def getLamNonLoc(self,iT,iters):
        if (iters==0 and iT > 1):
            lamarg = 2.0*self._allFibers.getLambdas()-1.0*self._allFibersPrev.getLambdas();
        else: # more than 1 iter, lambda should be previous lambda from fixed point
            lamarg = self._allFibers.getLambdas().copy();
        if (iters==0):
            # Copy all information over on the first iteration
            self._allFibersPrev = copy.deepcopy(self._allFibers);
        return lamarg;
    
    def getLambdaN(self,iT):
        prevWt = 0.5-0.5*(iT==0);
        return (1+prevWt)*self._allFibers.getLambdas()-prevWt*self._allFibersPrev.getLambdas();

    def getMaxIters(self,iT):
        if (iT < 2):
            return itercap;
        return 1;

    def gettval(self,iT,dt):
        return (iT+0.5)*dt;
    
    def getFirstNetworkStep(self,dt):
        return dt*0.5;
    
    def getSecondNetworkStep(self,dt):
        return dt*0.5;

