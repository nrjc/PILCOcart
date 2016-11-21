function [f, df] = explorePI(~, cc, ccprev, ~)
% EXPLOREPI, is the negative Probability of Improvement exploration heuristic,
% applied to the cumulative cost.
%
% [f, df] = explorePI([], cc, ccprev)
%
% cc      .    cumulative (discounted) cost structure
%   m     1x1  mean scalar
%   s     1x1  variance scalar
% ccprev  .    cumulative (discounted) cost structure of previous rollout
%   m     1x1  mean scalar
%   s     1x1  variance scalar
% f       1x1  loss
% df      .    loss derivartive structute
%   m     1x1  loss derivative wrt cc mean
%   s     1x1  loss derivative wrt cc variance
%
% See also <a href="exploreMyopicGittins.pdf">explorePI.pdf</a>
% Rowan McAllister 2016-05-04

s2 = cc.s + ccprev.s;
z = (cc.m - ccprev.m) / sqrt(s2);
p = normpdf(z);
c = normcdf(z);

f = c - 1;

df.m = p/sqrt(s2);
df.s = -z*p/(2*s2);
