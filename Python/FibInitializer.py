from math import pi
from DiscretizedFiber import DiscretizedFiber
import chebfcns as cf
import numpy as np


def makeCurvedFiber(Lf,N,fibDisc):
    s=cf.chebPts(N,[0,Lf],1);
    w=cf.chebPts(N,[0,Lf],1);
    # Stuff for the single relaxing fiber
    Xs = np.concatenate(([np.cos(s**3*(s-Lf)**3)],[np.sin(s**3*(s-Lf)**3)],\
        [np.ones(N)]),axis=0).T;
    Xs/=np.sqrt(2);
    Xs = np.reshape(Xs,3*N);
    raise ValueError('Relaxing fib not working - wrong configs at t=0')
    if (N== 8 and Lf == 2):
        X = np.array([0.0135868585233040,-1.88367626667077e-07,0.0135868585254738,
        0.119161084940861,-0.000925883980571755,0.119168979976868,
        0.311205286133222,-0.0306091557580084,0.314259301992996,
        0.514904756895834,-0.177928020388073,0.569157091545076,
        0.672418531390246,-0.404262502645662,0.845056470828019,
        0.876118002155044,-0.551581367277032,1.09995426038010,
        1.06816220334974,-0.581264639054238,1.29504458239623,
        1.17373642976460,-0.582190334670287,1.40062670384762]);
    elif (N==8 and Lf==1):
        X=np.array([0.00679342926272737,-1.47160719678204e-09,0.00679342926273690,
        0.0595844890218645,    -7.23389437645757e-06,    0.0595844899884341,
        0.157129276282544,    -0.000241025467463159,    0.157129650996498,
        0.284571643323847,    -0.00148771540855491,    0.284578545772538,
        0.422505704948698,    -0.00356291974301512,    0.422528235414009,
        0.549948071990455,    -0.00480960967881386,    0.549977130190049,
        0.647492859248864,    -0.00504340125536853,    0.647522291198113,
        0.700283919010473,    -0.00505063367751204,    0.700313351923811]);
    elif (N == 16 and Lf == 2):
        X=np.array([0.00340491242334965,-7.55939526115110e-10,    0.00340491242335634,
        0.0304477798090872,    -4.61506714542881e-06,    0.0304477805993711,
        0.0834935216250868,    -0.000237845135100219,    0.0834942746932120,
        0.160452114835654,    -0.00282449415571885,    0.160505847685669,
        0.257524257602402,    -0.0157320808646905,    0.258522988015229,
        0.365989737859111,    -0.0537488417770661,    0.373778951947674,
        0.469605378652775,    -0.128195896095909,    0.501844517425369,
        0.555660066937668,    -0.233136559573819,    0.637798196587002,
        0.631663221352522,    -0.349053963463132,    0.776415365786093,
        0.717717909634932,    -0.453994626942068,    0.912369044947726,
        0.821333550425869,    -0.528441681254251,    1.04043461042542,
        0.929799030687314,    -0.566458442171906,    1.15569057435787,
        1.02687117345458,    -0.579366028879417,    1.25370771468743,
        1.10382976665745,    -0.581952677896902,    1.33071928767988,
        1.15687550847917,    -0.582185907970421,    1.38376578177372,
        1.18391837586371,    -0.582190522279972,    1.41080864994974]);
    else:
        raise ValueError('Do not have the configuration of a curved fiber loaded for that N, L')
    fibList = [None];
    fibList[0] = DiscretizedFiber(fibDisc,X,Xs);
    return fibList;

def makeFallingFibers(Lf,N,fibDisc):
    s=cf.chebPts(N,[0,Lf],1);
    # Falling fibers
    Xs = np.concatenate(([np.zeros(N)],[np.zeros(N)],[np.ones(N)]),axis=0).T;
    Xs = np.reshape(Xs,3*N);
    d = 0.2;
    X1 = np.concatenate(([np.zeros(N)+d],[np.zeros(N)],[s-1]),axis=0).T;
    X2 = np.concatenate(([np.zeros(N)],[np.zeros(N)+d],[s-1]),axis=0).T;
    X3 = np.concatenate(([np.zeros(N)-d],[np.zeros(N)],[s-1]),axis=0).T;
    X4 = np.concatenate(([np.zeros(N)],[np.zeros(N)-d],[s-1]),axis=0).T;
    fibList = [None]*4;
    fibList[0] = DiscretizedFiber(fibDisc,np.reshape(X1,3*N),Xs);
    fibList[1] = DiscretizedFiber(fibDisc,np.reshape(X2,3*N),Xs);
    fibList[2] = DiscretizedFiber(fibDisc,np.reshape(X3,3*N),Xs);
    fibList[3] = DiscretizedFiber(fibDisc,np.reshape(X4,3*N),Xs);
    return fibList;

def makeThreeSheared(Lf,N,fibDisc):
    s=cf.chebPts(N,[0,Lf],1);
    # Falling fibers
    Xs13 = np.concatenate(([np.ones(N)],[np.zeros(N)],[np.zeros(N)]),axis=0).T;
    Xs2 = np.concatenate(([np.zeros(N)],[np.ones(N)],[np.zeros(N)]),axis=0).T;
    Xs13 = np.reshape(Xs13,3*N);
    Xs2 = np.reshape(Xs2,3*N);
    X1 = np.concatenate(([s-2.01],[np.zeros(N)-0.8],[np.zeros(N)]),axis=0).T;
    X2 = np.concatenate(([np.zeros(N)],[s-1],[np.zeros(N)]),axis=0).T;
    X3 = np.concatenate(([s+0.01],[np.zeros(N)+0.8],[np.zeros(N)]),axis=0).T;
    fibList = [None]*3;
    fibList[0] = DiscretizedFiber(fibDisc,np.reshape(X1,3*N),Xs13);
    fibList[1] = DiscretizedFiber(fibDisc,np.reshape(X2,3*N),Xs2);
    fibList[2] = DiscretizedFiber(fibDisc,np.reshape(X3,3*N),Xs13);
    return fibList;

