  function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
                                                         = congpa(policy, m, s)

% Gaussian process policy with first order SE kernels. Compute mean, variance
% and input output covariance of the action using a Gaussian process policy
% function, when the input is Gaussian. The GP is parameterized using a pseudo
% training set of N cases. Optionally, compute partial derivatives w.r.t the 
% input parameters.
%
% This version sets the signal variance to 1, the noise to 0.01 and their
% respective lengthscales to zero. This results in only the lengthscales,
% inputs, and outputs being trained.
%
% inputs:
% policy   policy (struct)
%  .hyp     GP log hyperparameters                                      [q    ]
%  .inputs  policy pseudo inputs                                        [N x d]
%  .target  policy pseudo targets                                       [N x D]
% m        mean of state distribution                                   [d    ]
% s        covariance matrix of state distribution                      [d x d]
%
% outputs:
% M        mean of the action                                   [D    ]
% S        variance of action                                   [D x D]
% C        covariance input and action                          [d x D]
% dMdm     derivative of mean action w.r.t mean of state        [D     x d    ]    
% dSdm     derivative of variance of action w.r.t mean of state [D x D x d    ] 
% dCdm     derivative of covariance w.r.t mean of state         [d x D x d    ] 
% dMds     derivative of mean action w.r.t variance             [D     x d x d]
% dSds     derivative of action variance w.r.t variance         [D x D x d x d]
% dCds     derivative of covariance w.r.t variance              [d x D x d x d]
% dMdp     derivative of mean action w.r.t GP parameters        [D     x r    ]
% dSdp     derivative of action variance w.r.t GP parameters    [D x D x r    ]
% dCdp     derivative of covariance w.r.t GP parameters         [d x D x r    ]
%
% where q = (2d+1)*D and r = (2d+1)*D + (d+D)*N
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth, Andrew 
% McHutchon 2012-01-18. Edited by Joe Hall 2012-04-05

d = size(policy.inputs,2);
policy.hyp(d+1:2*d,:) = zeros(1,d);                  % set signal variance to 1
policy.hyp(end,:) = log(0.01);                 % set noise standard dev to 0.01

if nargout < 4                                 % if no derivatives are required
  [M, S, C] = gpa2(policy, m, s);
  
else                                             % else compute derivatives too
  [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdi, dSdi, dCdi, dMdt, ...
                           dSdt, dCdt, dMdh, dSdh, dCdh] = gpa2d(policy, m, s);
    
  % signal and noise variance is fixed
  d2 = size(policy.hyp,1); dimU = size(policy.target,2);
  sidx = bsxfun(@plus,(d+1:d2)',(0:dimU-1)*d2);
  dMdh(:,sidx(:)) = 0; dSdh(:,:,sidx(:)) = 0; dCdh(:,:,sidx(:)) = 0;
         
  % cat derivs for all parameters
  D = size(policy.target,2); 
  dMdp = cat(2, reshape(dMdh, D, []), reshape(dMdi, D, []), ...
                                                         reshape(dMdt, D, []));
  dSdp = cat(3, reshape(dSdh, D, D, []), reshape(dSdi, D, D, []), ...
                                                      reshape(dSdt, D, D, []));
  dCdp = cat(3, reshape(dCdh, d, D, []), reshape(dCdi, d, D, []), ...
                                                      reshape(dCdt, d, D, []));
end

if(any(imag(S(:)))); disp('congp 1.1'); keyboard; end
if(any(eig(S)<-1e-10)); disp('congp 1.2'); keyboard; end
if any(diag(S)<-1e-10); keyboard; end