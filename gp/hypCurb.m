function [f, df] = hypCurb(hyp, x, y, mS, curb)

% wrapper for gpr training, penalising large SNR, extreme length-scales,
% and linear mean weights larger than 1. The penalty thresholds can be set 
% in the curb struct input argument. The penalty should keep:
%                          SNR < curb.snr
%       curb.std/curb.ls <  L  < curb.std*curb.ls
%                     -1 <  m  < 1
%
% Carl Edward Rasmussen & Andrew McHutchon, 2013-11-07

if nargin < 4, curb.snr = 500; curb.ls = 100; curb.std = 1; end % set default

p = 30;                                                     % penalty power

[f, df] = gp(hyp, x, y, mS);                               % first, call gp

f = f + sum(((hyp.l - log(curb.std'))./log(curb.ls)).^p);   % length-scales
df.l = df.l + p*(hyp.l - log(curb.std')).^(p-1)/log(curb.ls)^p;

f = f + sum(((hyp.s - hyp.n)/log(curb.snr)).^p);    % signal to noise ratio
df.s = df.s + p*(hyp.s - hyp.n).^(p-1)/log(curb.snr)^p;
df.n = df.n - p*sum((hyp.s - hyp.n).^(p-1)/log(curb.snr)^p);

if isfield(hyp,'m') && mS;               % keep linear weights between -/+1
    f = f + sum(hyp.m.^p);
    df.m = df.m + p*hyp.m.^(p-1);
end
