"""
BundledFibers.py
Plot fibers with bundles different colors
ffmpeg -framerate 20 -start_number 1 -i Ld3BundlingEvery2_%d.png -pix_fmt yuv420p -r 14 -vcodec libx264 -an -vf "scale=1109:985:force_original_aspect_ratio=increase,crop=1109:985" Ld3BundlingNoBrown.mp4

ffmpeg -framerate 20 -i RelaxationTurnover5_%d.png -pix_fmt yuv420p -r 14 -vcodec libx264 -an -vf "scale=1109:985:force_original_aspect_ratio=increase,crop=1109:985" RelaxationTurn5.mp4

for file in *Ld2Brownian4Movie*; do convert $file -trim "T_$file"; done
ffmpeg -framerate 20 -start_number 1 -i T_Ld2Brownian4Movie%d.png -pix_fmt yuv420p -r 14 
-vcodec libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" Ld2RigidBrownian.mp4

"""

import numpy as np
from plotoptix import TkOptiX
from plotoptix.utils import map_to_colors  # feature to color conversion
import chebfcns as cf
from Domain import PeriodicShearedDomain
from math import pi, sin
import sys
from functools import partial

import sys
sys.path.append("/Configs/")  

def save_image(rt,iT,name):
    rt.save_image("BundlingPaperPlots/Lp1/"+name+"_"+str(iT)+".png")
    #rt.save_image("StrainedTop.png")
    print("rt completed!")
    rt.close()  

def main():

    # Make some data first:
    iT = int(float(sys.argv[1]))
    t = 0.05*iT;
    ChebGridKind = 2
    #t = 0.01*iT;
    #if (iT < 25):
    #    
    #else:
    #    g = 0.2;
    g = 0;#0.2*sin(2*pi*t);
    nFib = 200;
    N = 13;
    Nuniform = 1000;
    Lp = 2;
    Dom = PeriodicShearedDomain(Lp,Lp,Lp);
    Dom.setg(g);
    U = cf.ResamplingMatrix(Nuniform,N,'u',ChebGridKind)
    name = 'SFBendLd2_Lp1.0_Dt0.0001';
    Chebpts = np.loadtxt('BundlingPaperPlots/Lp1/'+name+'_'+str(iT+1)+'.txt');
    Bundles = np.loadtxt('BundlingPaperPlots/Lp1/NewLabels_'+name+'.txt');
    Labels = Bundles[iT,:];
    NumBundles = np.amax(Labels);
    print('Number of bundles %d' %(NumBundles))
    AllColors = np.loadtxt('Colormap.txt',delimiter=',');
    #print(AllColors)
    if (NumBundles > 1):
        Spacing = int(len(AllColors)//(NumBundles-1))-1;
    else:
        Spacing = len(AllColors)-1;
    print(Spacing)
    BundleColors = AllColors[::Spacing,:];
    print(BundleColors)
    xyz = np.zeros((nFib*Nuniform,3));
    colors = np.zeros((nFib*Nuniform,3));
    c1 = np.random.rand(3);
    c2 = np.random.rand(3);
    for iFib in range(nFib):
        xyz[iFib*Nuniform:(iFib+1)*Nuniform,:]=np.dot(U,Chebpts[iFib*N:(iFib+1)*N,:])
        colors[iFib*Nuniform:(iFib+1)*Nuniform,:] = np.array([0.9, 0.9, 0.99]);
        if (Labels[iFib] > 0):
            colors[iFib*Nuniform:(iFib+1)*Nuniform,:] = BundleColors[int(Labels[iFib])-1,:]   
    # Translate particles in z and y until they are on [0,L] in those directions
    xyz[:,2]-=np.floor(xyz[:,2]/Lp)*Lp
    yshift = np.floor(xyz[:,1]/Lp)*Lp
    xyz[:,1]-=yshift;
    xyz[:,0]-=g*yshift;
    # Translate particles in x until they sit in parallelogram
    xyz[:,0]-=np.floor(xyz[:,0]/Lp)*Lp # Now they are sitting on [0,L]
    #print(np.max(np.abs(xyz)))
    #np.savetxt('FibersDiameterApart.txt',xyz)
    if (g > 0):
        inds = xyz[:,1] > 1/g*xyz[:,0]
        xyz[inds,:]+=[Lp,0,0]; 
    elif (g < 0):
        inds = xyz[:,1] > 1/g*(xyz[:,0]-Lp)
        xyz[inds,:]-=[Lp,0,0]; 
     
    r = 0.004*np.ones(Nuniform*nFib);
    
    particles = xyz
    print(particles[10,:])
    # Shift so they're in front of the camera
    camshift = -np.array([Lp/2, Lp/2, 3*Lp/4]);
    xyz+=camshift;
    
    allshift = camshift + [Lp/2,Lp/2,Lp/2];

    # Swap y and z
    #temp = particles[:,1].copy();
    #particles[:,1] = particles[:,2].copy();
    #particles[:,2] = temp;
    rp = r
    # Use xyz positions to calculate RGB color components:

    # Map Y-coordinate to matplotlib's color map RdYlBu: map_to_colors()
    # function is automatically scaling the data to fit <0; 1> range.
    # Any other mapping is OK, just keep the result in shape
    # (n-data-points, 3), where 3 stands for RGB color
    # components.
    
    # Create plots:

    optix = TkOptiX(on_rt_accum_done=partial(save_image,iT=int(iT),name=name),width=1024,height=1024) # create and configure, show the window later

    # accumulate up to 30 frames (override default of 4 frames)

    # white background
    optix.set_background(0.99)
    optix.set_ambient(0.15)     # dim ambient light (0.05)

    # add plots, ParticleSet geometry is default
    optix.set_data("particles", pos=particles, r=rp, c=colors)#np.array([1,0,0]))
    # and use geom parameter to specify cubes geometry;
    # Parallelepipeds can be described precisely with U, V, W vectors,
    # but here we only provide the r parameter - this results with
    # randomly rotated cubes of U, V, W lenghts equal to r 

    # tetrahedrons look good as well, and they are really fast on RTX devices:
    #optix.set_data("tetras", pos=cubes, r=rc, c=cc, geom="Tetrahedrons")

    # if you prefer cubes aligned with xyz:
    #optix.set_data("cubes", pos=cubes, r=rc, c=cc, geom="Parallelepipeds", rnd=False)

    # or if you'd like some edges fixed:
    #v = np.zeros((rc.shape[0], 3)); v[:,1] = rc[:]
    #optix.set_data("cubes", pos=cubes, u=[0.05,0,0], v=v, w=[0,0,0.05], c=cc, geom="Parallelepipeds")

    # show coordinates box
    #optix.set_coordinates()

    # Do the CLs
    if (True):
        NCLuniform = 40;
        U = cf.ResamplingMatrix(NCLuniform,N,'u',ChebGridKind)
        CBindingSites = np.zeros((nFib*NCLuniform,3));
        for jFib in range(nFib):
             CBindingSites[jFib*NCLuniform:(jFib+1)*NCLuniform,:]=np.dot(U,Chebpts[jFib*N:(jFib+1)*N,:])
        Links = np.loadtxt('BundlingPaperPlots/Lp1/Step'+str(iT)+'Links'+name+'_1.txt')
        nLinks = int(Links[0,0]);
        print(nLinks)
        PlotLinks = Links[1:,:];
        for iLink in range(nLinks):
            ThisPt = int(PlotLinks[iLink,0]);
            OtherPt = int(PlotLinks[iLink,1]);
            d = CBindingSites[ThisPt,:]-CBindingSites[OtherPt,:];
            dNew = Dom.calcShifted(d);
            shift = d - dNew;
            ThisLink = np.zeros((2,3));
            ThisLink[0,:]=CBindingSites[ThisPt,:];
            ThisLink[1,:]=CBindingSites[OtherPt,:]+shift
            # Form a line of the spheres
            NumCLSph = 100;
            CLSph = np.zeros((NumCLSph,3));
            dLink = ThisLink[1,:]-ThisLink[0,:];
            for iSph in range(NumCLSph):
                CLSph[iSph,:]= ThisLink[0,:]+(iSph)/(NumCLSph-1)*dLink
            # Translate particles in z and y until they are on [0,L] in those directions
            CLSph[:,2]-=np.floor(CLSph[:,2]/Lp)*Lp
            yshift = np.floor(CLSph[:,1]/Lp)*Lp
            CLSph[:,1]-=yshift;
            CLSph[:,0]-=g*yshift;
            # Translate particles in x until they sit in parallelogram
            CLSph[:,0]-=np.floor(CLSph[:,0]/Lp)*Lp # Now they are sitting on [0,L]
            #print(np.max(np.abs(CLSph)))
            if (g > 0):
                inds = CLSph[:,1] > 1/g*CLSph[:,0]
                CLSph[inds,:]+=[Lp,0,0]; 
            elif (g < 0):
                inds = CLSph[:,1] > 1/g*(CLSph[:,0]-Lp)
                CLSph[inds,:]-=[Lp,0,0]; 
            optix.set_data('link'+str(iLink), pos=CLSph.copy()+camshift, r=r[0], c=[0.2,0.2,0.2])
           
            
    
    ## THIS IS PLOTTING THE BOX. He draws it using a bunch of long thin parallelograms. 
    if (True):
        ax_width = 0.004;
        optix.set_data("ax1", pos=[-Lp/2,-Lp/2,-Lp/2]+allshift, u=[ax_width, 0, 0], v=[g*Lp,Lp,0], w=[0, 0, ax_width], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax2", pos=[Lp/2,-Lp/2,-Lp/2]+allshift, u=[ax_width, 0, 0], v=[g*Lp,Lp,0], w=[0, 0, ax_width], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax3", pos=[-Lp/2,-Lp/2,Lp/2]+allshift, u=[ax_width, 0, 0], v=[g*Lp,Lp,0], w=[0, 0, ax_width], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax4", pos=[Lp/2+g*Lp,Lp/2,Lp/2]+allshift, u=[ax_width, 0, 0], v=[-g*Lp,-Lp,0], w=[0, 0, ax_width], c=[0, 0, 0], geom="Parallelepipeds")
        
        optix.set_data("ax5", pos=[-Lp/2,-Lp/2,-Lp/2]+allshift, u=[ax_width, 0, 0], v=[0, ax_width, 0], w=[0,0,Lp], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax6", pos=[-Lp/2,-Lp/2,-Lp/2]+allshift, u=[Lp,0,0], v=[0, 0, ax_width], w=[0, ax_width, 0], c=[0, 0, 0], geom="Parallelepipeds")
        
        optix.set_data("ax7", pos=[Lp/2,-Lp/2,Lp/2]+allshift, u=[-Lp,0,0], v=[0, 0, ax_width], w=[0, ax_width, 0], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax8", pos=[Lp/2,-Lp/2,-Lp/2]+allshift, u=[-ax_width, 0, 0], v=[0, -ax_width, 0], w=[0,0,-Lp], c=[0, 0, 0], geom="Parallelepipeds")
        
        optix.set_data("ax9", pos=[Lp/2+g*Lp,Lp/2,Lp/2]+allshift, u=[-Lp,0,0], v=[0, 0, -ax_width], w=[0, -ax_width, 0], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax10", pos=[g*Lp+Lp/2,Lp/2,Lp/2]+allshift, u=[ax_width, 0, 0], v=[0, -ax_width, 0], w=[0,0,-Lp], c=[0, 0, 0], geom="Parallelepipeds")
        
        optix.set_data("ax11", pos=[-Lp/2+g*Lp,Lp/2,-Lp/2]+allshift, u=[ax_width, 0, 0], v=[0, ax_width, 0], w=[0,0,Lp], c=[0, 0, 0], geom="Parallelepipeds")
        optix.set_data("ax12", pos=[-Lp/2+g*Lp,Lp/2,-Lp/2]+allshift, u=[Lp,0,0], v=[0, 0, ax_width], w=[0, ax_width, 0], c=[0, 0, 0], geom="Parallelepipeds")
    
    optix.set_param(min_accumulation_step=15, max_accumulation_frames=2000) #,light_shading="Hard"
    optix.set_uint("path_seg_range", 15, 40) # Just from the examples. Something to do with how far 
    
    # show the UI window here - this method is calling some default
    # initialization for us, e.g. creates camera, so any modification
    # of these defaults should come below (or we provide on_initialization
    # callback)
    optix.show()
    
    
    # camera and lighting configured by hand
    # Corner view
    optix.update_camera(eye=allshift+[0,0,2.5*Lp],target=allshift)
    # optix.update_camera(eye=0.9*np.array([-1.25*Lp,Lp,-2.5*Lp]),target=[0,0,0])
    # Top view
    #optix.update_camera(eye=np.array([0,-3*Lp,-Lp/8])+Lp/2,target=np.array([Lp/2,Lp/2,Lp/2]))
    #optix.update_camera(eye=np.array([0,-3*Lp,-Lp/16])+Lp/2,target=np.array([0,0,0])+Lp/2)
    TopView = False;
    #
    if (TopView):
        # Ld = 3
        #optix.setup_light("light1", color=5*np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=False,pos=np.array([Lp/2,2.5*Lp,0.75*Lp])+allshift)
        #optix.setup_light("light2", color=np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=True,pos=np.array([Lp/4,-1.75*Lp,0.75*Lp])+allshift)
        #optix.setup_light("light3", color=10*np.array([1,1,1]), radius=1.5, in_geometry=True,pos=1.5*np.array([0,0,Lp]))
        #optix.setup_light("light4", color=3*np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=False,pos=np.array([Lp/4,Lp/2,2.5*Lp])+allshift)
        # Ld = 2
        optix.setup_light("light1", color=2*np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=False,pos=np.array([Lp/2,2.5*Lp,0.75*Lp])+allshift)
        optix.setup_light("light2", color=np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=True,pos=np.array([Lp/4,-1.75*Lp,0.75*Lp])+allshift)
        #optix.setup_light("light3", color=10*np.array([1,1,1]), radius=1.5, in_geometry=True,pos=1.5*np.array([0,0,Lp]))
        optix.setup_light("light4", color=np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=False,pos=np.array([Lp/4,Lp/2,2.5*Lp])+allshift)
    else:
        optix.update_camera(eye=0.8*np.array([2*Lp,1.4*Lp,2*Lp])+allshift,target=[Lp,0.65*Lp,Lp]+allshift)
        #optix.setup_light("light1", color=10*np.array([0.7, 0.9, 0.99]), radius=1.5, in_geometry=False,pos=np.array([Lp/2,2.5*Lp,0.75*Lp])+allshift)
        #optix.setup_light("light2", color=10*np.array([0.7, 0.9, 0.99]), radius=1.25, in_geometry=True,pos=np.array([Lp/4,-1.75*Lp,0.75*Lp])+allshift)
        #optix.setup_light("light3", color=10*np.array([1,1,1]), radius=1.5, in_geometry=True,pos=1.5*np.array([0,0,Lp]))
        rsmall=2;
        LpOG=Lp;
        if (Lp > 4):
            rsmall =1.75;
        Lp=3;
        optix.setup_light("light3", color=4*np.array([0.7, 0.9, 0.99]), radius=rsmall, in_geometry=False,pos=np.array([-Lp,2*Lp,2*Lp])+allshift)
        optix.setup_light("light4", color=4*np.array([0.7, 0.9, 0.99]), radius=rsmall, in_geometry=False,pos=1.2*np.array([1.5*Lp,1.5*Lp,Lp])+allshift)
        optix.setup_light("light2", color=4*np.array([0.7, 0.9, 0.99]), radius=rsmall, in_geometry=False,pos=1.2*np.array([-Lp,-Lp,-Lp])+allshift)
        optix.setup_light("light1", color=4*np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=False,pos=1.2*np.array([1.5*Lp,-1.5*Lp,-Lp])+allshift)
        if (LpOG > 4):
            optix.setup_light("light5", color=4*np.array([0.7, 0.9, 0.99]), radius=2, in_geometry=False,pos=1.2*np.array([0*Lp,-1.5*Lp,Lp])+allshift)
    
    print("done")

if __name__ == '__main__':
    main()
