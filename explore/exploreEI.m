function [f, df] = exploreEI(~, cc, ccprev, ~)
% EXPLOREEI, is the negative Expected Improvement exploration heuristic,
% applied to the cumulative cost.
%
% [f, df] = exploreEI([], cc, ccprev)
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
% See also <a href="exploreMyopicGittins.pdf">exploreEI.pdf</a>
% Rowan McAllister 2016-05-04

m = cc.m - ccprev.m;
s = sqrt(cc.s + ccprev.s);
z = m / s;
p = normpdf(z);
cn = normcdf(-z);

f = cn * m - p * s;

df.m = cn;
df.s = -p/(2*s);
