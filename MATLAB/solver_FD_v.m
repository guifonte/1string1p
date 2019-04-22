function [z_p,y_p,z_b1,z_b2,y_b1,y_b2,a,b] = solver_FD(body_matrix,string_matrix,string_parameters,pluck_parameters,d,Fs)
%% LOADING PARAMETERS
load(pluck_parameters)
load(string_parameters)
%% SIMULATION PARAMETERS
dt = 1/Fs;
t = [0:floor(d/dt)-1]*dt;  
%% INPUT PARAMETERS
%String
A = string_matrix.A;
A3bar = string_matrix.A3bar;
PHISc = string_matrix.PHISc;
Ns = string_matrix.Ns;
L=string_parameters.L;
PhiS0 = @(x) (x/L); 
PhiS = @(j,x) sin(j*pi*x/L); 
j=[1:Ns]; 
%Body
B = body_matrix.B;
B3bar = body_matrix.B3bar; 
PHIB = body_matrix.PHIB;
Nb = body_matrix.Nb;
xb=L;
%Pluck parameters
xp = pluck_parameters.xp; 
Ti = pluck_parameters.Ti;
dp = pluck_parameters.dp;
F0 = pluck_parameters.F0;
gamma = pluck_parameters.gamma;
Tr = Ti + dp;
%% INITIALISATION
% Modal quantities
a = NaN*zeros(2*(Ns+1),length(t));
b = NaN*zeros(Nb,length(t));
a(:,[1 2]) = 0;
b(:,[1 2]) = 0;
Fc = NaN*zeros(2,length(t));
%% ITERATIVE COMPUTATION OF SOLUTION (Finite differences scheme)
for i = 2:length(t)-1
%string
Fe = (F0/(dp))*((i)*dt-Ti)*(heaviside((i)*dt-Ti)-heaviside((i)*dt-Tr));  
FeZ = Fe*sin(gamma);
FeY = Fe*cos(gamma);
alpha = [a(:,i); a(:,i-1); FeZ; FeY];
%body
beta = [b(:,i); b(:,i-1)];
%coupling force calculation
Fc(:,i) = ((PHISc*A3bar)+(PHIB*B3bar))\(PHISc*A*alpha-PHIB*B*beta);   
% compute modal coordinates for next time-step
a(:,i+1) = A*alpha - A3bar*Fc(:,i); 
b(:,i+1) = B*beta + B3bar*Fc(:,i);
end
%% COMPUTING PHYSICAL DISPLACEMENTS
PhiSc = [PhiS0(xb)  PhiS(j,xb)]; % at the coupling point
PhiSe = [PhiS0(xp)  PhiS(j,xp)]; % at the excitation point   
%String displacement at the plucking point
z_p = real(PhiSe*a(1:(Ns+1),:))*1e3; %in mm
y_p = real(PhiSe*a((Ns+2):(2*(Ns+1)),:))*1e3; %in mm
%String displacement at the coupling point
z_b1 = real(PhiSc*a(1:(Ns+1),:))*1e3;%in mm
y_b1 = real(PhiSc*a((Ns+2):(2*(Ns+1)),:))*1e3;%in mm
%Body displacement at the coupling point
z_b2 = real(PHIB(1,:)*b)*1e3; %in mm
y_b2 = real(PHIB(2,:)*b)*1e3; %in mm
