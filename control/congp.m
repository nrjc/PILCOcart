  function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
                                                         = congp(policy, m, s)

% Gaussian process policy. Compute mean, variance and input output covariance of
% the action using a Gaussian process policy function, when the input is
% Gaussian. The GP is parameterized using a pseudo training set of
% N cases. Optionally, compute partial derivatives wrt the input parameters.
%
% This version sets the signal variance to 1, the noise to 0.01 and their
% respective lengthscales to zero. This results in only the lengthscales,
% inputs, and outputs being trained.
%
% inputs:
% policy      policy (struct)
% policy.hyp  GP log hyperparameters (Ph = (d+2)*D)                 [ Ph      ]
% policy.inputs  policy pseudo inputs                               [ N  x  d ]
% policy.target  policy pseudo targets                              [ N  x  D ]
% m           mean of state distributiond                           [ d       ]
% s           covariance matrix of state distribution               [ d  x  d ]
%
% outputs:
% M           mean of the action                                    [ D       ]
% S           variance of action                                    [ D  x  D ]
% C           covariance input and action                           [ d  x  D ]
% dMdm        derivative of mean action w.r.t mean of state         [ D  x  d ]
% dSdm        derivative of variance of action w.r.t mean of state  [D*D x  d ]
% dCdm        derivative of covariance w.r.t mean of state          [d*D x  d ]
% dMds        derivative of mean action w.r.t variance              [ D  x d*d]
% dSds        derivative of action variance w.r.t variance          [D*D x d*d]
% dCds        derivative of covariance w.r.t variance               [d*D x d*d]
% dMdp        derivative of mean action w.r.t GP parameters         [ D  x  P ]
% dSdp        derivative of action variance w.r.t GP parameters     [D*D x  P ]
% dCdp        derivative of covariance w.r.t GP parameters          [d*D x  P ]
%
% where P = (d+2)*D + n*(d+D) is the number of policy parameters.
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth, Andrew 
% McHutchon 2012-06-25, Edited by Joe Hall 2012-07-09

policy.hyp = policy.p.hyp;
policy.inputs = policy.p.inputs;
policy.target = policy.p.target;

policy.hyp(end-1,:) = 0;                             % set signal variance to 1
policy.hyp(end,:) = log(0.01);                 % set noise standard dev to 0.01

if nargout < 4                                 % if no derivatives are required
  [M, S, C] = gp2(policy, m, s);
else                                             % else compute derivatives too
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdi, dSdi, dCdi, dMdt, ...
                            dSdt, dCdt, dMdh, dSdh, dCdh] = gp2d(policy, m, s);
    
    d = size(policy.inputs,2);             % signal and noise variance is fixed
    d2 = size(policy.hyp,1); dimU = size(policy.target,2);
    sidx = bsxfun(@plus,(d+1:d2)',(0:dimU-1)*d2);
    dMdh(:,sidx(:)) = 0; dSdh(:,sidx(:)) = 0; dCdh(:,sidx(:)) = 0;

    dMdp = [dMdh dMdi dMdt]; dSdp = [dSdh dSdi dSdt]; dCdp = [dCdh dCdi dCdt];
end

if(any(imag(S(:)))); disp('congp 1.1'); end
if(any(eig(S)<-1e-10)); disp('congp 1.2'); end
if any(diag(S)<-1e-10); disp('congp 1.3'); end
