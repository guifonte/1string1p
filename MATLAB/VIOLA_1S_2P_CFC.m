%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%  TIME DOMAIN SIMULATIONS: 1 STRING(DOUBLE POLARIZATION) + BODY  %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Hybrid model:
%I)Experimental body modes
%II)Analytical(pinned) string modes
%III)Contact force computation at each time step
%Numerical solution (Finite difference scheme)
%-----------------------------------REFERENCE------------------------------------------------------------
% M. Demoucron, On the control of virtual violins - Physical modelling and control of
% bowed string instruments, Ph.D. Thesis, Universit´e Pierre et Marie Curie Paris VI, Paris,
% France; Royal Institute of Technology, Stockholm, Sweden (2008)
%--------------------------------------------------------------------------------------------------------
%G. PAIVA, September, 2016.
% Laboratoire d'Acoustique de l'Universite du Maine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 
%close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Constants
i=sqrt(-1);
%% Input parameters
%String
string_parameters = 'C:\Users\Guilherme\Google Drive\WORK\Viola Caipira Modelling and Simulations\DATA\STRINGS\1 String\2 Polarizations\string_2.mat';
%Body
body_parameters = 'C:\Users\Guilherme\Google Drive\WORK\Viola Caipira Modelling and Simulations\DATA\BODY\LSCE\10 Coupling Points\Viola_5points_1.unv';
%Pluck parameters
pluck_parameters = 'C:\Users\Guilherme\Google Drive\WORK\Viola Caipira Modelling and Simulations\DATA\STRINGS\1 String\2 Polarizations\pluck_3.mat';

%% Simulation parameters
fhmax = 5000; %Maximum frequency of string 
Nb = 17
Fs = 5*44100; 
dt=1/Fs;
d = 3.2; % duration
t = [0:floor(d/dt)-1]*dt;
Nstr=1;
%% Output parameters
%Output controls and file names
%Movie
Movie = 0;
MovieFile_Name = 'test.avi';
Movie_duration = 0.05; 
%Sound
%Sound = 1;
%SoundFile_Name =  '1str_NoCol_1.wav';
tic
%% String
[string_matrix] = strings(string_parameters,pluck_parameters,fhmax,dt);
%% Body
[body_matrix] = body2(body_parameters,Nb,Nstr,dt);
%% Resolution
[z_p,y_p,z_b1,z_b2,y_b1,y_b2,a,b] = solver_FD(body_matrix,string_matrix,string_parameters,pluck_parameters,d,Fs);
%% Calculation of Acceleration and Velocity
[vel,acc]=calc_va(z_p,y_p,z_b1,y_b1,d,Fs);
break
%% Plots
figure(5000)
subplot(2,1,1)
set(gcf,'Color','w')
hold on
plot(t,z_p,'Color','Red')
legend('CFC')
ylabel('z_{pluck}(mm)')
xlabel('Time(s)')
box on
subplot(2,1,2)
plot(t,y_p,'Color','Red')
legend('CFC')
ylabel('y_{pluck}(mm)')
xlabel('Time(s)')
box on

figure(5001)
subplot(2,1,1)
set(gcf,'Color','w')
hold on
plot(t,z_b1,'Color','Red')
legend('CFC')
ylabel('z_{bridge}(mm)')
xlabel('Time(s)')
box on
subplot(2,1,2)
hold on
plot(t,y_b1,'Color','Red')
legend('CFC')
ylabel('y_{bridge}(mm)')
xlabel('Time(s)')
box on
toc
break
%% Displacement sound
%Plucking point
Name_dp = 'Disp_Ppoint.wav';
SOUND_disp = [z_p.' + y_p.'];
SOUND_dispdec = decimate(SOUND_disp,Fs/44100);
sound(0.95*SOUND_dispdec/max(abs(SOUND_dispdec)),44100);
audiowrite(Name_dp,SOUND_dispdec ./(max(abs(SOUND_dispdec))),44100); %Normalized
%Bridge
Name_db = 'Disp_Bpoint.wav';
SOUND_disp = [z_b1.' + y_b1.'];
SOUND_dispdec = decimate(SOUND_disp,Fs/44100);
sound(0.95*SOUND_dispdec/max(abs(SOUND_dispdec)),44100);
audiowrite(Name_db,SOUND_dispdec ./(max(abs(SOUND_dispdec))),44100); %Normalized
%% Velocity sound
%Plucking point
Name_vp = 'Vel_Ppoint.wav';
SOUND_vel = [vel.z_p_v.' + vel.y_p_v.'];
SOUND_veldec = decimate(SOUND_vel,Fs/44100);
sound(0.95*SOUND_veldec/max(abs(SOUND_veldec)),44100);
audiowrite(Name_vp,SOUND_veldec./(max(abs(SOUND_veldec))),44100); %Normalized
%Bridge 
Name_vb = 'Vel_Bpoint.wav';
SOUND_vel = [vel.z_b1_v.' + vel.y_b1_v.'];
SOUND_veldec = decimate(SOUND_vel,Fs/44100);
sound(0.95*SOUND_veldec/max(abs(SOUND_veldec)),44100);
audiowrite(Name_vb,SOUND_veldec./(max(abs(SOUND_veldec))),44100); %Normalized
figure(100000)
plot(vel.z_b1_v)
%% Acceleration sound
%Plucking point
Name_ap = 'Acc_Ppoint.wav';
SOUND_acc = [acc.z_p_a.' + acc.y_p_a.'];
SOUND_accdec = decimate(SOUND_acc,Fs/44100);
audiowrite(Name_ap,0.95*SOUND_accdec./(max(abs(SOUND_accdec))),44100); %Normalized
sound(0.95*SOUND_accdec/max(abs(SOUND_accdec)),44100);
%Bridge
Name_ab = 'Acc_Bpoint.wav';
SOUND_acc = [acc.z_b1_a.' + acc.y_b1_a.'];
SOUND_accdec = decimate(SOUND_acc,Fs/44100);
audiowrite(Name_ab,0.95*SOUND_accdec./(max(abs(SOUND_accdec))),44100); %Normalized
sound(0.95*SOUND_accdec/max(abs(SOUND_accdec)),44100);
%% Plots
figure(5001)
subplot(3,1,1)
set(gcf,'Color','w')
hold on
plot(t,SOUND_dispdec./(max(abs(SOUND_dispdec))),'Color','Red')
ylabel('Disp')
xlabel('Time(s)')
box on
subplot(3,1,2)
plot(t,SOUND_veldec./(max(abs(SOUND_veldec))),'Color','Red')
ylabel('Vel')
xlabel('Time(s)')
box on
subplot(3,1,3)
plot(t,SOUND_accdec./(max(abs(SOUND_accdec))),'Color','Red')
ylabel('Acc')
xlabel('Time(s)')
box on

%% Movie
if Movie    
animation(a,z_b1,y_b1,z_p,y_p,Movie_duration,MovieFile_Name,string_parameters,Fs)
end
%% Calcul spectre 
s1=real(z_b1);
s1=s1(:,0.01*Fs:0.41*Fs); % 
Nfft=length(s1);          % Nombre de points de calcul FFT 
[module_Tf,freq2]=spectre(s1,Fs,Nfft);
%% Visu spectre
figure(3)
%plot(freq2,20*log10(module_Tf))
semilogx(freq2,20*log10(module_Tf));
hold on
box on; 
xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
legend('f_s = 441kHz')
% ajustement de l'affichage
%f_debut=50; f_fin=24000; xlim([f_debut f_fin]); ylim([-60 20]);
set(gcf,'Color','w')
%% Calcul spectrogramme
Fs=5*44100;
SOUND_vel=real([vel.z_b1_v + vel.y_b1_v]);
SOUND_vel(1,1:2)=0;
SOUND_vel=decimate(SOUND_vel,Fs/44100);
SOUND_vel=SOUND_vel(floor(0.010*44100):floor(3.2*44100));
SOUND_vel=[zeros(1,floor(0.010*44100)) SOUND_vel];
Length_window=80e-3;    % largeur de la fenetre d'analyse en ms
Delta=0.4*5e-3;              % décalage entre les fenetres d'analyse en ms 
Nfft=length(SOUND_vel);          % Nombre de points de calcul FFT
Dyn=90; %dynamique utilisée pour le seuillage de S_module % 
Fs1=44100;
[S_module,f2,t2]=spectro(SOUND_vel,Fs1,Nfft,Length_window, Delta, Dyn);
%%% Visu spectrogramme
figure(30033);
clf;
imagesc(t2,f2,S_module)
% ajustement de l'affichage 
axis xy ; xlabel('Time [s]');ylabel('Frequency [Hz]')
cb1 = colorbar;
xlabel(cb1,'dB')
t_debut=0.04;t_fin=t2(end); xlim([t_debut t_fin]);
f_debut=0; f_fin=6000; ylim([f_debut f_fin]);
set(gcf,'Color','w')
