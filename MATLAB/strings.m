function [string_matrix] = strings(string_parameters,pluck_parameters,fhmax,dt)

load(string_parameters)
load(pluck_parameters)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CREATE STRING MATRICES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STRING PROPERTIES 
L = string_parameters.L;
D = string_parameters.D;
mu = string_parameters.mu;
T = string_parameters.T;
B = string_parameters.B;
f0 = string_parameters.f0;
xp = pluck_parameters.xp; 
xb=L;
Ns = round(fhmax/f0-1)

rho=1.2; %air density - (Valette)
neta=1.8e-5; %air dynamic viscosity - (Valette)

c=sqrt(T/mu);

mj = @(j) L*mu/2 + 0*j; % modal masses
kj = @(j) (j.^2)*pi^2*T/(2*L) + (j.^4)*B*pi^4/(2*L^3); % modal stiffnesses
omegaj = @(j) (j.*pi/L)*sqrt(T/mu).*sqrt(1+(j.^2).*((B.*pi^2)/(T*L.^2))); % modal angular frequencies (considering inharmonic effects) 
fj = @(j) omegaj(j)/(2*pi);
%% DAMPING PARAMETERS 
R=@(j) ones(Ns,1)*2*pi*neta+2*pi*D*sqrt(pi*neta*rho.*fj(j)); %air friction coefficient
Qf=@(j) (2*pi*mu*fj(j))./R(j); %damping due to the air friction

dVETE=1e-3; %thermo and visco-elastic effects constant (Valette) - It is a problem, high uncertainty in determination!
Qvt=@(j) ((T^2)*c)./((4*pi^2)*(B)*dVETE.*fj(j).^2); % damping due to the thermo and visco-elasticity

Qd=5500; %damping due to the dislocation phenomenom (Adjusted in Pathé) - Depends strongly on the history of the material 7000 - 80000 for brass strings(Cuesta)!

Qt =@(j) (Qf(j).*Qvt(j).*(ones(Ns,1)*Qd))./(Qvt(j)*Qd+Qf(j)*Qd+Qvt(j).*Qf(j)); % Modal damping factor!
cj= @(j) (j*pi*sqrt(T*mu))./(2*Qt(j)); %Modal damping

j=[1:Ns]';   
%% STIFFNESS, DAMPING AND MASS MATRICES
MJ =  [
      L*mu/3  L*mu./(pi*j.') 
      L*mu./(pi*j)    diag(mj(j))
      ];                       
KJ = diag([T/L; kj(j)]);
CJ = diag([0; cj(j)]);   
%% MODESHAPES
PhiS0 = @(x) (x/L); 
PhiS = @(j,x) sin(j*pi*x/L); 
PhiSc = [PhiS0(xb)  PhiS(j,xb).']; % at the coupling point
PhiSe = [PhiS0(xp)  PhiS(j,xp).']; % at the excitation point  
%% RESOLUTION MATRICES
AIZ=inv(MJ/dt^2+CJ/(2*dt));
A1Z = AIZ*(2*MJ/dt^2-KJ);
A2Z = AIZ*(CJ/(2*dt)-MJ/dt^2);
A3Z = AIZ;

A1Y=A1Z;
A2Y=A2Z;
A3Y=A3Z;

A1=blkdiag(A1Z,A1Y);
A2=blkdiag(A2Z,A2Y);
A3=blkdiag(A3Z*PhiSe.',A3Y*PhiSe.');

string_matrix.A = [A1 A2 A3];
string_matrix.GS = blkdiag(A3Z*PhiSc.',A3Y*PhiSc.');
string_matrix.PHISc = blkdiag(PhiSc,PhiSc);

string_matrix.Ns = Ns;

String = [fj(j) Qt(j)];
fString = fj(j)
save String.txt String -ASCII;

end
