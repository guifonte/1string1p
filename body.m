function [body_matrix] = body(body_parameters,Nb)

[mode] = read_LSCE(body_parameters,Nb);

[real_mode] = conv_real_LSCE(mode);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CREATE BODY MATRICES %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mk = real_mode.mass;
wdk = 2*pi*real_mode.freq;
xsi = real_mode.xsi;
wnk = wdk./sqrt(1-xsi.^2);
ck = 2.*mk.*xsi.*wnk;  
PhiB = real_mode.shape;
     
body_matrix.MK=diag(mk);
body_matrix.KK=diag(mk.*wnk.^2);            
body_matrix.CK=diag(ck);

body_matrix.PhiB = PhiB;

body_matrix.Nb = length(PhiB);
fBody=real_mode.freq
BodyUnc = [real_mode.freq.' xsi.' PhiB.'];
save BodyUnc.txt BodyUnc -ASCII;

end

