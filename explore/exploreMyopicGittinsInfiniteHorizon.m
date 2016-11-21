function [f, df] = exploreMyopicGittinsInfiniteHorizon(expl, cc, ~, gamma)
% exploreMyopicGittinsInfiniteHorizon, is an exploration heuristic, applied to
% the cumulative cost. See exploreMyopicGittins.pdf document for details.
%
% [f, df] = exploreMyopicGittinsInfiniteHorizon(expl, cc, [], gamma)
%
% expl  .    exploration structure
%   on  1x1  observation noise variance parameter
% cc    .    cumulative (discounted) cost structure
%   m   1x1  mean scalar
%   s   1x1  variance scalar
% gamma      discount rate
% f     1x1  loss
% df    .    loss derivartive structute
%   m   1x1  loss derivative wrt cc mean
%   s   1x1  loss derivative wrt cc variance
%
% See also <a href="exploreMyopicGittins.pdf">exploreMyopicGittins.pdf</a>
% Rowan McAllister 2016-05-04

on = expl.on;
assert(0 <= gamma && gamma < 1, ...
  'exploreMyopicGittins: gamma (discount rate) must be in range [0,1)')

if ~exist('on','var'); on = 0; end % 0 is an optimisitc assumption
s = cc.s / sqrt(cc.s+on);

x = 0; % lambda == x in exploreMyopicGittins.pdf document
if s > 0 % else x remains as 0
  while true
    p = normpdf(x/s);
    c = normcdf(x/s);
    g  = gamma*(s*p + x*c - x) + x;
    if abs(g) < 1e-12; break; end % converged
    dg = gamma*(c - 1) + 1;
    x = x - g/dg; % Newton's method
  end
end

f = cc.m + x;

df.m = 1;
if s > 0
  dxds = -p/(c-1+1/gamma);
  dsdccs = (sqrt(cc.s+on) - s/2) / (cc.s+on);
  df.s = dxds * dsdccs;
else
  df.s = 0;
end
