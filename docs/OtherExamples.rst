Examples for numerical tests
===============================
    - CheckStability.py: this example is Section 7.4.3 in `Maxian's PhD thesis <https://www.proquest.com/docview/2832979813?pq-origsite=gscholar&fromopenview=true>`_
      and simulates 1000 fibers in an oscillatory shear flow. This can be used to look at convergence of GMRES at t = 0, or to look at the stability of the hydrodynamics
      over time. This is for deterministic suspensions. 
    - EndToEndFlucts.py: this example is in Section 8.4-5 of Maxian's PhD thesis, and is for fluctuating fibers. 
      The idea is to simulate the equilibrium fluctuations of 100 (or more/less) fibers, and 
      look at the equilibrium distribution of end-to-end distance, to check if it matches MCMC. Thus, this file allows you to take an arbitrary mobility and simply
      simulate free fluctuations until a certain end time. It will write 2 outputs: the locations of the fibers (for you to analyze end-to-end distance offline), and the 
      number of GMRES iterations required at each time step to solve the saddle point system (for the constraint forces and mobility). This will be zero if the mobility
      is local (on each fiber individually). 
    - LinearizedFluctuations.py: this example is Section 8.3 of Maxian's PhD thesis, and simulates equilibrium fluctuations of a curved fiber around a base state. 
      There is a penalty energy which holds the fiber in the base state. It will write the locations of the fiber(s) at every time step, to confirm that the covariance
      of the (small) fluctuations is equivalent to what we would expect theoretically (though this comparison would happen offline). 
    - ThreeShearedFibs.py: this example is Section 7.4.2 in Maxian's PhD thesis
      and simulates three fibers being advected by a shear flow. The fibers are not rigid, and so the disturbance flow created from advecting the central fiber
      is enough to deform the two fibers outside of it. This represents a good example to test the accuracy of nonlocal hydrodynamic interactions (as is presented 
      in that section of the thesis). It will write the output file ThreeShearedLocations.txt, which gives the collocation points of each of the three fibers at every 
      time step. This can be used for convergence plots, etc. This is for deterministic suspensions. 
