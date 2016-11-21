function [f, df] = exploreUCB(expl, cc, ~, ~)
% EXPLOREUCB, is an exploration heuristic, applied to the cumulative cost.
%
% [f, df] = exploreUCB(expl, cc)
%
% expl   .    exploration structure
%   beta 1x1  the trade-off parameter between exploration (sqrt(cc.s)) and 
%             exploitation (cc.m)
% cc     .    cumulative (discounted) cost structure
%   m    1x1  mean scalar
%   s    1x1  variance scalar
% f      1x1  loss
% df     .    loss derivartive structute
%   m    1x1  loss derivative wrt cc mean
%   s    1x1  loss derivative wrt cc variance
%
% Rowan McAllister 2016-05-04

beta = expl.beta;

f = cc.m - beta*sqrt(cc.s);

df.m = 1;
df.s = -beta/2/sqrt(cc.s);
