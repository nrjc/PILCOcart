function [f, df] = hypCurbum(hyp, x, y, curb)

% wrapper for gpr training, penalising large SNR and extreme length-scales.
%
% Carl Edward Rasmussen, 2013-07-05

if nargin < 4, curb.snr = 500; curb.ls = 100; curb.std = 1; end   % set default

p = 30;                                                         % penalty power

[f, df] = gpum(hyp, x, y);                                       % first, call gp

f = f + sum(((hyp.l - log(curb.std'))./log(curb.ls)).^p);       % length-scales
df.l = df.l + p*(hyp.l - log(curb.std')).^(p-1)/log(curb.ls)^p;

f = f + sum(((hyp.s - hyp.n)/log(curb.snr)).^p);        % signal to noise ratio
df.s = df.s + p*(hyp.s - hyp.n).^(p-1)/log(curb.snr)^p;
df.n = df.n - p*sum((hyp.s - hyp.n).^(p-1)/log(curb.snr)^p);
