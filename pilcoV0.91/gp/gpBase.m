function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpBase(gpmodel, m, s)

% Base function for computing GP predictions with uncertain inputs. This
% function acts as a landing pad for gp calls, directing the call on to the
% appropriate function.
%
% dynmodel  dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%     .m    (optional) D-by-1 linear weights for the GP mean
%     .b    (optional) 1-by-1 biases for the GP mean
%   inputs  n-by-D, training inputs
%   targets n-by-E, training targets
%   induce  (optional) np-by-D(-by-E), inducing inputs can either be shared or
%           separate over output dimensions
%   iK      n-by-n-by-E, inverse covariance matrix
%   beta    n-by-E, iK*targets
% m         D-by-1, mean of the test distribution
% s         D-by-D, covariance matrix of the test distribution
%
% M         E-by-1, mean of pred. distribution 
% S         E-by-E, covariance of the pred. distribution             
% V         D-by-E, inv(s) times covariance between input and output
% dMdm      E-by-D, deriv of output mean w.r.t. input mean 
% dSdm      E^2-by-D, deriv of output covariance w.r.t input mean
% dVdm      D*E-by-D, deriv of input-output cov w.r.t. input mean
% dMds      E-by-D^2, deriv of ouput mean w.r.t input covariance
% dSds      E^2-by-D^2, deriv of output cov w.r.t input covariance
% dVds      D*E-by-D^2, deriv of inv(s)*input-output covariance w.r.t input cov
%
% Copyright (C) 2008-2013 by Carl Edward Rasmussen, Marc Deisenroth, 
%  Andrew McHutchon, & Joe Hall   2013-07-08

% Check if we need to recalculate precomputable matrices
if ~isfield(gpmodel,'iK') || ~isequal(unwrap(gpmodel.hyp),gpmodel.oldh)...
                                     || size(gpmodel.inputs,1) ~= gpmodel.oldn;
  gpmodel = gpPreComp(gpmodel); fprintf('Precomputable matrices computed\n');
end

switch (nargout > 3) + 2*(isfield(gpmodel,'approxS')&&gpmodel.approxS)
    case 0                                      % no derivatives, full S
        [M, S, V] = gpp(gpmodel, m, s);
    
    case 1                                      % derivatives. full S
        [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpd(gpmodel, m, s);
    
    case 2                                      % no derivatives, approx S
        [M, S, V] = gpas(gpmodel, m, s);
    
    case 3                                      % derivatives and approx S
        [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpasd(gpmodel, m, s);
end