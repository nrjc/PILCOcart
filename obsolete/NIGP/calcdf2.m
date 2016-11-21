function df2 = calcdf2(lhyp,x,preC,xs)
% df2 = calcdf(lhyp,x,preC)
% Function to calculate the derivative of the posterior mean of a GP about
% each of the training points.
% Inputs
%   lhyp          the log hyperparameters, struct with two fields
%       .seard    hyperparameters for the squared exponential kernal,
%                  D+2-by-E
%       .lsipn    log standard deviation input noise hyperparameters, one
%                 per input dimension
%   x             training inputs matrix, N-by-D
%   preC          pre computed variables
%   xs            (optional) locations to calc slopes at instead of at x
%
% Output
%   df2            matrix of squared derivatives, N-by-D
%
% Andrew McHutchon, Dec 2011

D = size(x,2); 

iell2 = exp(-2*lhyp.seard(1:D,:));      % D-by-E

if nargin < 4
    % Form the derivative covariance function
    XmXiLam = bsxfun(@times,preC.XmX,permute(iell2,[4,3,2,1])); % N-by-N-by-E-by-D
    dKdx = bsxfun(@times,XmXiLam,preC.K);                       % N-by-N-by-E-by-D

    % Compute derivative
    df2 = etprod('123',dKdx,'4123',preC.alpha,'42').^2;       % N-by-E-by-D

else
    % Test set
    % Find the derivative of the covariance function
    [d,Ks] = covSEardA(lhyp.seard,x,xs,preC);
    Xmx = permute(bsxfun(@minus,permute(x,[1,3,2]),permute(xs,[3,1,2])),[1,2,4,3]); % N-by-Ns-by-1-by-D
    XmxiLam = bsxfun(@times,Xmx,permute(iell2,[4,3,2,1])); % N-by-Ns-by-E-by-D
    dKdx = bsxfun(@times,XmxiLam,Ks);                      % N-by-Ns-by-E-by-D

    df2 = etprod('123',dKdx,'4123',preC.alpha,'42').^2;      % Ns-by-E-by-D
end
