runN2plot=1;
for eps=1e-2
chebers=[];
if (runN2plot)
    Ns = 45;
else
    Ns= [15 30 45 60];
end
for iN=1:length(Ns)
N = Ns(iN);
if (N == 45 && runN2plot)
     N2s = [6 4 8];
else
    N2s = 6;
end
for iN2 = 1:length(N2s)
NForSmall=N2s(iN2);
L=2;            % fiber length
mu=1;%/(8*pi);    % fluid viscosity
a=eps*L;
% Nodes for solution, plus quadrature and barycentric weights:
[s,w,b] = chebpts(N, [0 L], 1); % 1st-kind grid for ODE.
D = diffmat(N, 1, [0 L], 'chebkind1');

% Fiber initialization
q=7; 
X_s = [cos(q*s.^3 .* (s-L).^3) sin(q*s.^3.*(s - L).^3) ones(N,1)]/sqrt(2);
Nx = N+1;
[sNp1,wNp1,bNp1]=chebpts(Nx,[0 L],1); % Type 1 grid (rectangular spectral colloc)
DNp1 = diffmat(Nx,[0 L],'chebkind1');
RToNp1 = barymat(sNp1,s,b);
IntDNp1 = pinv(DNp1);
BMNp1 = stackMatrix(barymat(L/2,sNp1,bNp1));
% Construct matrix that gives X on the N+1 grid from (X_s, X_MP)
I=zeros(3*(N+1),3);
for iR=1:N+1
    I(3*(iR-1)+1:3*iR,1:3)=eye(3);   
end
XonNp1Mat = [(eye(3*(N+1))-repmat(BMNp1,N+1,1))*stackMatrix(IntDNp1*RToNp1) I];

X = XonNp1Mat*[reshape(X_s',[],1);0;0;0];
X = reshape(X,3,[])';
% Upsampled mobility
if (N==60 || runN2plot)
Mtt = zeros(3*Nx);
for iC=1:3*Nx
    fIn = zeros(3*Nx,1);
    fIn(iC)=1;
    fInput = reshape(fIn,3,Nx)';
    % Subtract singular part for each s
    Utt = 1/(8*pi*mu)*upsampleRPY(X,sNp1,X,fInput,sNp1,bNp1,200,L,a);
    Mtt(:,iC) = reshape(Utt',3*Nx,1);
end
end

AllbS = precomputeStokesletInts(sNp1,L,a,Nx,1);
AllbS_SBT = precomputeStokesletInts(sNp1,L,0,Nx,1);
AllbD = precomputeDoubletInts(sNp1,L,a,Nx,1);

Mrpy = TransTransMobilityMatrix(X,a,L,mu,sNp1,bNp1,DNp1,AllbS,AllbD,NForSmall,0,0);
Msbt = TransTransMobilityMatrix(X,a,L,mu,sNp1,bNp1,DNp1,AllbS_SBT,AllbD,NForSmall,1,0);

% Calculate error on upsampled grid
nzation = max(real(eig(Mrpy)));
if (runN2plot)
    nzation=1/(8*pi*mu);
    startplotind=2*iN2-1;
    set(gca,'ColorOrderIndex',iN2)
    semilogy(3*Nx-1:-1:0,sort(real(eig(Mrpy)))/nzation,'-');
    hold on
    set(gca,'ColorOrderIndex',iN2)
    semilogy(3*Nx-1:-1:0,-sort(real(eig(Mrpy)))/nzation,'-.o');
    xlabel('$k$','interpreter','latex')
    if (iN==length(Ns) && iN2==length(N2s))
        semilogy(3*Nx-1:-1:0,sort(real(eig(Mtt)))/nzation,'-k');
    end
else
if (N > 45)
    nzation=1/(8*pi*mu);
    semilogy(3*Nx-1:-1:0,sort(real(eig(Mtt)))/nzation,'-k');
else
    nzation=1/(8*pi*mu);
    if (N==60)
    semilogy(3*Nx-1:-1:0,sort(real(eig(Mtt)))/nzation,'-k');
    hold on
    end
    semilogy(3*Nx-1:-1:0,sort(real(eig(Mrpy)))/nzation,'-');
    hold on
    set(gca,'ColorOrderIndex',iN)
    semilogy(3*Nx-1:-1:0,-sort(real(eig(Mrpy)))/nzation,'-.o');
    set(gca,'ColorOrderIndex',iN)
    semilogy(3*Nx-1:-1:0,sort(real(eig(Msbt)))/nzation,'--');
    set(gca,'ColorOrderIndex',iN)
    semilogy(3*Nx-1:-1:0,-sort(real(eig(Msbt)))/nzation,':s');
    xlabel('$k$','interpreter','latex')
    ylabel('$8 \pi \mu \nu_k$','interpreter','latex')
end
end
end
end
end
