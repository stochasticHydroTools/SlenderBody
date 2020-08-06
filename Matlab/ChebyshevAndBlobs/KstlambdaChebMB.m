% Function to convert K^T lambda from a uniform grid to a Chebyshev one
% For the conversion to "work" (give the same answer), lambda must
% integrate to 0
%%
% Initialize fiber curved fiber
rng(0);
NCheb = 16; % number of Chebyshev points
[sCheb,w]=chebpts(NCheb,[-1 1],1);
th_Cheb = acos(sCheb);
% Matrix of values of T_k on the Chebyshev grid
CoeffsToValsCheb = cos(th_Cheb.*(0:NCheb-1));
X = 1/sqrt(2)*[cos(sCheb) sin(sCheb) sCheb]; % curved & inextensible
Xs = 1/sqrt(2)*[-sin(sCheb) cos(sCheb) ones(NCheb,1)];
XsStack = reshape(Xs',3*NCheb,1);
% Initialize lambda as an N x 3 Chebyshev series
lamCoeffs = rand(NCheb,3).*exp(-0.1*(0:NCheb-1))';
lambda = CoeffsToValsCheb*lamCoeffs; % values on the Chebyshev grid
% Make sure lambda integrates to 0. This is a crude approximation so there
% will be some small error involved. 
lamCoeffs(1,:)=lamCoeffs(1,:)-w*lambda/2;
lambda = CoeffsToValsCheb*lamCoeffs; % new values on the Chebyshev grid
lamStack=reshape(lambda',3*NCheb,1);
% Our spectral method to compute K^T lambda
[K,Kt]=getKMats3D(XsStack,CoeffsToValsCheb(:,1:NCheb-1),0,w,NCheb);
% Stack lambda and compute K^T lambda by multiplication
Ktlam = [Kt*lamStack; (w*lambda)'];

%% Uniform points version
Nuni = 400;
ds = 2/Nuni;
sUni=(-1+ds:ds:1)';
% Matrix that takes coefficients of Cheb polynomials to values on the
% uniform grid. Equivalently, matrix with values of the Chebyshev
% polynomials on the uniform grid
CoeffsToValsUniform = cos(acos(sUni).*(0:NCheb-1));
% Sample lambda on the uniform grid
lamUni = CoeffsToValsUniform*lamCoeffs;
sumlam = ds*sum(lamUni); % integral of lambda (first order)
% Evaluate tangent vectors and normal vectors on the uniform grid
XsUni = 1/sqrt(2)*[-sin(sUni) cos(sUni) ones(Nuni,1)];
[theta,phi,~] = cart2sph(XsUni(:,1),XsUni(:,2),XsUni(:,3));
theta(abs((abs(phi)-pi/2)) < 1e-12) =0;
n1s=[-sin(theta) cos(theta) 0*theta];
n2s=[-cos(theta).*sin(phi) -sin(theta).*sin(phi) cos(phi)];
% Evaluate K^* lambda for Brennan (stacking it differently from the notes)
KstBlob = ds*makeKstBlob(lamUni,n1s,n2s,Nuni);
KstCheb_fromBlob = convertKstBlob(KstBlob,CoeffsToValsUniform,NCheb-1,Nuni,ds);
max(abs(Ktlam-KstCheb_fromBlob))

function KstBlob = makeKstBlob(lams,n1s,n2s,Nuni)
    % Evaluates Brennan's formula for the blobs
    % K^T lambda = [n_1(s) dot integral_s^L lambda(s)] for each s
    KstBlob = zeros(2*Nuni+3,1);
    for iPt=1:Nuni-1
        KstBlob(iPt)=dot(n1s(iPt,:),sum(lams(iPt:end,:)));
        KstBlob(iPt+Nuni)=dot(n2s(iPt,:),sum(lams(iPt:end,:)));
    end
    KstBlob(2*Nuni+1:2*Nuni+3)=sum(lams); % add integrals of lambda
end

function KstCheb = convertKstBlob(KstBlob,CoeffsToValsUniform,Npolys,Nuni,ds)
    % This is the function I wrote on paper to convert the multiblob to
    % Chebyshev.
    KstCheb = zeros(2*Npolys+3,1);
    for iPoly=1:Npolys
        n1prods = KstBlob(1:Nuni).*CoeffsToValsUniform(:,iPoly);
        n2prods = KstBlob(Nuni+1:2*Nuni).*CoeffsToValsUniform(:,iPoly);
        KstCheb(iPoly)=ds*sum(n1prods);
        KstCheb(Npolys+iPoly)=ds*sum(n2prods);
    end
    KstCheb(2*Npolys+1:2*Npolys+3)=KstBlob(2*Nuni+1:2*Nuni+3);
end


