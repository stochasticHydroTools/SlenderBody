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
function Mtt = TransTransMobilityMatrix(X,a,L,mu,s,b,D,AllbS,AllbD,NForSmall,asymp,delta)
    [N,~]=size(X);
    Xs = D*X;
    Xss = D*Xs;
    if (~asymp && delta > 0)
        delta=0; % override user input for exact RPY
        warning('You input delta > 0 and said that you wanted exact RPY')
    end
    Loc_Slt = getMlocStokeslet(N,Xs,a,L,mu,s,delta);
    Loc_Dblt = getMlocDoublet(N,Xs,a,L,mu,s,delta);
    SletFP = StokesletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,AllbS);
    Mtt = Loc_Slt+2*a^2/3*Loc_Dblt+SletFP;
    if (asymp)
        Rest = getMlocSmallParts(N,Xs,a,L,mu,s,delta);
    else
        Rest = 1/(8*pi*mu)*upsampleRPYSmallMatrix(X,s,X,s,b,NForSmall,L,a) ...
         + 2*a^2/3*DoubletFinitePartMatrix(X,Xs,Xss,D,s,L,N,mu,AllbD);
    end
    Mtt = Mtt+Rest;
end

% Compute the integrals on |s-s'| < 2a for the trans-trans matrix Mtt. 
% It uses Nup/2 Gauss-Legendre points on the two different sides of s. 
function Matrix = upsampleRPYSmallMatrix(Targs,starg,X0,s0,b0,Nup,L,a)
    % Collocation pts
    N =length(Targs);
    Matrix = zeros(3*N,3*N);
    Nhalf = floor(Nup/2);
    %AllRS = [];
    for iT=1:N
        t = starg(iT);
        P = Targs(iT,:);
        for iD=0:1
            dom = [t min(t+2*a,L)];
            if (iD==0)
                dom=[max(t-2*a,0) t];
            end
            if (max(dom)-min(dom) > 0)
                [ssm,wsm,~]=legpts(Nhalf,dom);
                Rsm = barymat(ssm, s0, b0);
                %AllRS = [AllRS; diag(wsm)*Rsm];
                Xsm = Rsm*X0;
                stRsm = stackMatrix(diag(wsm)*Rsm);
                % Formulate as row of a matrix
                RowVec = zeros(3,3*Nhalf);
                for jPt=1:Nhalf
                    R = P-Xsm(jPt,:);
                    nR = norm(R);
                    Rhat = R/nR;
                    RowVec(:,3*(jPt-1)+1:3*jPt)=...
                        4/(3*a)*((1-9*nR/(32*a))*eye(3)+3/(32*a)*(R'*Rhat));
                end
                Matrix(3*(iT-1)+1:3*iT,:) = Matrix(3*(iT-1)+1:3*iT,:) +RowVec*stRsm;
            end
        end
    end
end

function MTT = getMlocSmallParts(N,Xs,a,L,mu,s0,delta)
    % Regularized version
    s = RegularizeS(s0,delta,L);
    MTT = zeros(3*N);
    for iPt=1:N
        inds = (iPt-1)*3+1:3*iPt;
        t = s(iPt);
        tau = Xs(iPt,:);
        nXs = norm(tau);
        Xshat = tau/nXs;
        XsXs = Xshat'*Xshat;
        if (t < 2*a)
            intI = (128*a^2-36*a^2*nXs+64*a*t-9*nXs*t^2)/(48*a^2);
            intTau = nXs*(1/4+t^2/(16*a^2));
        elseif (t > L-2*a)
            sbar = (L-t);
            intI = (128*a^2-36*a^2*nXs+64*a*sbar-9*nXs*sbar^2)/(48*a^2);
            intTau = nXs*(1/4+sbar^2/(16*a^2));
        else
            intI = 1/6*(32-9*nXs);
            intTau = nXs/2;
        end
        MTT(inds,inds)=intI*eye(3)+intTau*XsXs;
    end
    MTT = 1/(8*pi*mu)*MTT;
end