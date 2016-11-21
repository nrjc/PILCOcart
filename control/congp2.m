  function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
                                                         = congp2(par, m, s)

% Gaussian process policy. Compute mean, variance and input output covariance of
% the action using a Gaussian process policy function, when the input is
% Gaussian. The GP is parameterized using a pseudo training set of
% N cases. Optionally, compute partial derivatives wrt the input parameters.
%
% This version sets the derivatives of the policy inputs to zero, meaning
% that only the targets and the lengthscales are trained.
%
% inputs:
% par      policy parameters (struct)
% par.hyp  GP log hyperparameters, (d+2)*D by 1
% par.inp  policy pseudo inputs, N by d
% par.tar  policy pseudo targets, N by D
% m        mean of state distribution, d by 1
% s        covariance matrix of state distribution, d by d
%
% outputs:
% M        mean of the action, D by 1
% S        variance of action, D by D
% C        covariance input and action, d by D
% dMdm     derivative of mean action wrt mean of state, D by d
% dSdm     derivative of variance of action wrt mean of state, D by D by d
% dCdm     derivative of covariance wrt mean of state, d by D by d
% dMds     derivative of mean action wrt variance, D by d by d
% dSds     derivative of action variance wrt variance, D by D by d by d
% dCds     derivative of covariance wrt variance, d by D by d by d
% dMdp     derivative of mean action wrt GP parameters, d by N by d
% dSdp     derivative of action variance wrt GP parameters, D by D by N by d
% dCdp     derivative of covariance wrt GP parameters, d by D by N by d
%
% Copyright (C) 2008-2009 by Carl Edward Rasmussen & Marc Deisenroth, 2009-10-26
% Andrew McHutchon, 18/01/2012

par.hyp(end-1,:) = 0; % set signal variance to 1
par.hyp(end,:) = log(0.01); % set noise standard dev to 0.01

if nargout < 4                                  % if no derivatives are required
  [M, S, C] = gp2(par, m, s);
else                                          % else compute derivatives, too
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdi, dSdi, dCdi, dMdt, ...
             dSdt, dCdt, dMdh, dSdh, dCdh] = gp2d(par, m, s);
    
    % signal and noise variance is fixed
    d = size(par.inputs,2); d2 = size(par.hyp,1); dimU = size(par.target,2);
    sidx = bsxfun(@plus,(d+1:d2)',(0:dimU-1)*d2);
    dMdh(:,sidx(:)) = 0; dSdh(:,:,sidx(:)) = 0; dCdh(:,:,sidx(:)) = 0;
    
    % policy inputs are fixed
    dMdi = 0*dMdi; dSdi = 0*dSdi; dCdi = 0*dCdi;
         
    % cat derivs for all parameters
    D = size(par.target,2); 
    dMdp = cat(2, reshape(dMdh, D, []), reshape(dMdi, D, []), reshape(dMdt, D, []));
    dSdp = cat(3, reshape(dSdh, D, D, []), reshape(dSdi, D, D, []), reshape(dSdt, D, D, []));
    dCdp = cat(3, reshape(dCdh, d, D, []), reshape(dCdi, d, D, []), reshape(dCdt, d, D, []));
end

%if(any(imag(S(:)))); disp('congp 1.1'); keyboard; end
%if(any(eig(S)<-1e-10)); disp('congp 1.2'); keyboard; end
%if any(diag(S)<-1e-10); keyboard; end