function [mode] = read_LSCE(body_parameters,Nb)

%Import modal TestLab modal data (only complex modes)
%Nb = Number of identified modes + Upper and Lower residuals.
%Modal filtering is not performed.

%INPUT
%data_path                   : data path
%Filename                    : file name where is stored the data
%Nb                          : number of body modes
%simul_FenetreExponentielle  : option for damping correction

%OUTPUT
%mode(k).freq   : frequency of damped system
%mode(k).shape  : complex mode shape
%mode(k).xsi    : modal damping factor
%mode(k).mk     : modal mass   

DS_init = 2; % nombre de DATASET à écarter en début de fichier unv %Discard DATASET 151 and 164... 

for k = 1:Nb
    
    [Data, Info, errmsg] = readuff(body_parameters,[k+DS_init],[55]); 
    
    % tri des données par noeud (bug export unv de LMS sur testlab 13A)
    [Data,noeuds_tries] = classe_TestLab_nodes(Data);    
      
        R3 = Data{1}.r3
        modalA  = Data{1}.modalA;
        A = modalA; 
        lambda =  Data{1}.eigVal;         
        Wn = abs(lambda); %pulsation propre du système conservatif  
        Wd = A/(2i); %pulsation du système dissipatif
        Freq = Wd/(2*pi); % frequence du système dissipatif
        Xsi = -real(lambda)./Wn; % amortissement modal
        Mk = A/(2*Wd*1i);         
    
        freq(k) = Freq;
        complexmode(k) = R3;
        xsi(k) = Xsi;
        mass(k) = Mk; 
end

 % regroupement des données dans la variable (structure) MODE
    mode.freq = freq;
    mode.shape = complexmode; % données brutes du fichier unv
    mode.xsi = xsi; % amortissement modal
    mode.mass = mass;       

 
 


