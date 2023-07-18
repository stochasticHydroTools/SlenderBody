L = 2;
eps = 4e-3/L;
mu = 1;
kbT = 4.1e-3;
Eb = 17*kbT; % pN*um^2 (Lp=17 um)
twmod = 10*kbT; % pN*um^2 (LpTwist = 10 um)
N = 16;
[spl,wpl,bpl] = chebpts(1000,[0 L]);
c= log(1/eps^2)/(8*pi*mu);
psi = 1.41*(2*pi/L)*(Eb/twmod);
H = chebop(0,L);
H.op = @(s,u) c*(-Eb*diff(u,4)+1i*twmod*psi*diff(u,3));
H.lbc = @(u) [diff(u,2); diff(u,3)];
H.rbc = @(u) [diff(u,2); diff(u,3)];
nEFs = N;
[V,D] = eigs(H,nEFs,'SM');
eigfunctions = V.blocks;
% Do the first two modes analyticallly (Chebfun not good at those for some
% reason)
FunctionValues = zeros(length(spl),nEFs);
eigvalues=diag(D);
eigvalues(3)
for i=1:nEFs
    FunctionValues(:,i)=eigfunctions{1,i}(spl);
end

% Check the eigenfunctions
% Nch=100; index=N-2;
% [sch,wch,bch]=chebpts(Nch,[0 L]);
% ch1 = barymat(sch,spl,bpl)*FunctionValues(:,index);
% D = diffmat(Nch,[0 L]);
% LHS = c*(-Eb*D^4*ch1+1i*twmod*psi*D^3*ch1);
% plot(sch,LHS);
% hold on
% plot(sch,eigvalues(index)*ch1);