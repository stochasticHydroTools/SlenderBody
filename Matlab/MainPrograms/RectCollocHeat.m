% Rectangular collocation to solve Poisson's equation
N=32;
[x,w,b]=chebpts(N+2,[-1 1],2);
[x0,w0,b0]=chebpts(N,[-1 1],1);
D=diffmat(N+2,2);
R=barymat(x0, x, b);
L=diffmat([2 N+2],0);
f=-pi^2*sin(pi*x0);
A=[R*D; L];
% Reverse operator
Reverse=R*A^(-1)*eye(N+2,N);
% Forward operator
B=diffmat([2 N+2],0);
Forward=R*D*[R; B]^(-1)*[eye(N); zeros(2,N)];
lams=eig(Forward*Reverse);
max(abs(Reverse*f-sin(pi*x0)))

% Rectangular collocation to solve the heat equation
% N=16;
% m=2;
% x=cos(pi*(0:N-1+m)/(N-1+m))';
% x0=cos(pi*(0:N-1)/(N-1))';
% u0=cos(pi*x0);
% A=diffmat([N+m N+m],2);
% R=diffmat([N N+m],0);
% L=-diffmat([2 N+m],1);
% E=[R;L]^-1*[eye(N); zeros(m,N)];
% E*u0-cos(pi*x);
% B=R*A*E;
% t=0.67;
% u=expm(t*B)*u0;
% B2=E*R*A;
% u2error=expm(t*B2)*cos(pi*x)-exp(-pi^2*t)*cos(pi*x)
% u1error=u-exp(-pi^2*t)*cos(pi*x0)

% Rectangular collocation to solve u_t=-u_xxxx
% N=16;
% m=4;
% x=cos(pi*(0:N-1+m)/(N-1+m))';
% x0=cos(pi*(0:N-1)/(N-1))';
% u0=cos(pi*x0);
% A=-diffmat([N+m N+m],4);
% R=diffmat([N N+m],0);
% L=[-diffmat([2 N+m],1); -diffmat([2 N+m],3)];
% E=[R;L]^-1*[eye(N); zeros(m,N)];
% E*u0-cos(pi*x);
% B=R*A*E;
% t=0.67;
% u=expm(t*B)*u0;
% err=max(abs(u-exp(-pi^4*t)*cos(pi*x0)))