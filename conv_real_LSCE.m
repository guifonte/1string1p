function [real_mode] = conv_real_LSCE(mode)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% IMPORT BODY MODAL DATA (complex modes)%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%(and convert complex modes into real ones)%%%    
cplx_shape = mode.shape;
real_shape = real(cplx_shape);
freq = mode.freq;
xsi = mode.xsi;

real_mode.shape= real_shape;
real_mode.freq = freq;
real_mode.xsi = xsi;
real_mode.mass = mode.mass;
end
    
