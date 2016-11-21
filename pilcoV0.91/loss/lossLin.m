function [L dLdm dLds S dSdm dSds C dCdm dCds] = lossLin(cost,m,s)
% [L dLdm dLds S dSdm dSds C dCdm dCds] = lossLin(cost,m,s)
% Function to compute the expected loss and its derivatives, given an input
% distribution, under a linear loss function: L = a^T(x - b). Note, this
% loss function can give negative losses.
%
% Inputs:
%    mu        mean of input distribution, D-by-1
%    S         covariance matrix of input distribution, D-by-D
%    cost.a    gradient of linear loss function, D-by-1
%    cost.b    targets, the value of x for which there is zero loss, D-by-1
%
% Outputs:
%      L       expected loss, scalar
%      dLdm    derivative of expected loss wrt input mean, D-by-1
%      dLds    derivative of expected loss wrt input covariance, D-by-D
%      S       variance of loss, scalar
%      dSdm    derivative of variance wrt input mean, D-by-1
%      dSds    derivative of variance wrt input covariance, D-by-D
%
% Andrew McHutchon, 22/11/10

a = cost.a(:); b = cost.b(:); D = length(m);
if length(a) ~= D || length(b) ~= D;
    error('a or b not the same length as m'); end

% Mean
L = a'*(m - b);
dLdm = a';
dLds = zeros(D);

% Covariance
S = a'*s*a;
dSdm = zeros(1,D);
dSds = a*a';

% inv(s) * IO covariance
C = a;
dCdm = zeros(D); dCds = zeros(D,D^2);

