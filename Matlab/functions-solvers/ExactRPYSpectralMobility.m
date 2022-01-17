% RPY trans-trans mobility matrix
% There are 5 parts:
% 1-2) The local drag matrices for the Stokeslet and doublet
% 3-4) The matrices for the remaining integral terms in the Stokeslet and
% doublet (called "finite part matrices" in the code)
% 5) The matrix for R < 2a, for which there are 3 options depending on the
% input NForSmall. If NForSmall > 0, it will use NForSmall/2 Gauss-Leg
% nodes on (s-2a,s) and (s,s+2a). If NForSmall=-1, it will assume a straight
% segement from (-2a,2a), and if NForSmall=0 it will us the asymptotic
% representation of the integral from -2a to 2a
function Mtt = ExactRPYSpectralMobility(N,X,Xs,Xss,Xsss,a,L,mu,s,b,D,AllbS,AllbD,NForSmall)
    Loc_Slt = getMlocStokeslet(N,Xs,a,L,mu,s,0);
    Loc_Dblt = getMlocDoublet(N,Xs,Xss,Xsss,stackMatrix(D),a,L,mu,s,0,0);
    if (NForSmall==0)
        SmallExact = getMlocSmallParts(N,Xs,a,L,mu,s,stackMatrix(D),D*Xs,0,0);
    elseif (NForSmall==-1)
        SmallExact = 1/(8*pi*mu)*upsampleRPYSmallStraightMatrix(s,s,b,Xs,32,L,a);
    else 
        SmallExact = 1/(8*pi*mu)*upsampleRPYSmallMatrix(X,s,X,s,b,NForSmall,L,a);
    end
    SletFP = StokesletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,AllbS);
    DbletFP = DoubletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,AllbD);
    Dlet = Loc_Dblt+DbletFP;
    Mtt = Loc_Slt+SletFP+2*a^2/3*Dlet + SmallExact;
end