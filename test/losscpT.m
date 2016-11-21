function [d dy dh] = losscpT(cost, s, delta)

% check derivatives of the cartPole/loss function, 2014-11-04

D = length(s.m);
if nargin < 3; delta = 1e-4; end
[d dy dh] = checkgrad(@tmp, [s.m s.s(tril(ones(D))==1)'], delta, cost);

function [f, df] = tmp(s, cost)
e = length(s); d = round(sqrt(8*e+9)/2-3/2);
t.m = s(1:d); t.s = zeros(d); 
t.s(tril(ones(d))==1) = s(d+1:e); t.s = t.s+t.s'-diag(diag(t.s));
[L, ~, dLd] = cost.fcn(cost, t);
f = L;
df(1:d) = dLd.m; 
z = 2*dLd.s-diag(diag(dLd.s)); df(d+1:e) = z(tril(ones(d))==1);
