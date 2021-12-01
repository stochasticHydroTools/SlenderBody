% Important variables
NC=40;           % number of Chebyshev pts
L=2;            % fiber length
mu=1/(8*pi);    % fluid viscosity
eps=1e-3;       % r/L aspect ratio
delta = 0.01;     % fraction of length over which to taper the fiber
a= 2e-3;
eps = 4*a/(exp(1.5)*L);

% Chebyshev discreization on [delta L, L-delta L]
ChebDomain = [delta*L L*(1-delta)];
[sC,wC,bC] = chebpts(NC,ChebDomain,1); % 1st-kind grid for ODE.
D = diffmat(NC, 1, ChebDomain, 'chebkind1');
Ds = zeros(3*NC);
for iD=1:3
    Ds(iD:3:end,iD:3:end)=D;
end

% Discretization on [0,delta L]
Nend = delta*L/a+1;
sEnd = (0:Nend-1)'*a;
sLEnd = flipud(L-sEnd);

% Total s
s = [sEnd; sC; sLEnd];
[Xpts,Xs] = X(s);

% Calculate mobility matrix column by column
M = zeros(3*length(s));
for iS=1:length(s)
    for iD=1:3
        f = zeros(3*length(s),1);
        f(3*(iS-1)+iD) = 1;
        fCheb = f(3*Nend+1:3*(Nend+NC));
        % This is the velocity EVERYWHERE from the MIDDLE PART
        UCheb = zeros(3*length(s),1);
        if (sum(abs(fCheb)) > 0)
            UCheb = 1/(8*pi*mu)*upsampleRPYPartial(Xpts,s,X(sC),reshape(fCheb,3,NC)',sC,bC,200,L,a,delta);
            UCheb = reshape(UCheb',3*length(s),1);
        end
        % This is the velocity EVERYWHERE from one end
        fStart = f(1:3*Nend);
        Ustart = zeros(3*length(s),1);
        if (sum(abs(fStart)) > 0)
            % This is just the RPY kernel between iS and 
            for jS=1:length(s) 
                MRPY = 1/(8*pi*mu)*calcRPYKernel(Xpts(iS,:),Xpts(jS,:),a);
                Ustart(3*jS-2:3*jS,:) = a*MRPY(iD,:);
                if (iS==1 || iS==Nend)
                    Ustart(3*jS-2:3*jS,:) = 0.5*MRPY(iD,:)*a;
                end
            end
        end
        % This is the velocity EVERYWHERE from the other end
        fEnd = f(3*(Nend+NC)+1:end);
        Uend = zeros(3*length(s),1);
        if (sum(abs(fEnd)) > 0)
            % This is just the RPY kernel between iS and jS
            for jS=1:length(s)
                MRPY = 1/(8*pi*mu)*calcRPYKernel(Xpts(iS,:),Xpts(jS,:),a);
                Uend(3*jS-2:3*jS,:) = MRPY(iD,:)*a;
                if (iS==Nend+NC+1 || iS==2*Nend+NC)
                    Uend(3*jS-2:3*jS,:) = 0.5*MRPY(iD,:)*a;
                end
            end
        end
        M(:,3*(iS-1)+iD)=UCheb+Ustart+Uend;
    end
end

% Uniform body force
U = zeros(3*length(s),1);
U(1:3:end)=1;
lam = M \ U;
lam = reshape(lam,3,length(s))';

% Multiblob answer
sMB = (0:a:L)';
[XMB,XsMB]=X(sMB);
MRPY = zeros(length(sMB));
for iS=1:length(sMB)
    for jS=1:length(sMB)
        MRPY(3*(iS-1)+1:3*iS,3*(jS-1)+1:3*jS) = 1/(8*pi*mu)*calcRPYKernel(XMB(iS,:),XMB(jS,:),a);
    end
end
% Force to force density
MRPY=MRPY*a;
MRPY(1:3,:)=0.5*MRPY(1:3,:);
MRPY(end-2:end,:)=0.5*MRPY(end-2:end,:);
U = zeros(3*length(sMB),1);
U(1:3:end)=1;
lamMB = MRPY \ U;
lamMB = reshape(lamMB,3,length(sMB))';


        
function [fibpts,Xs] = X(s)
    fibpts = [s zeros(length(s),2)];
    Xs = [ones(length(s),1) zeros(length(s),2)];
end
        
    