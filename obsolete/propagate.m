function [s, a] = propagate(s, dyn, ctrl)

% Propagate the state distribution one time step forward.
%
% [s, a] = propagate(s, dyn, ctrl)
%
% s        .       state structure
%   m      D x 1   mean vector
%   s      D x D   covariance matrix
% a        .       action structure
%   m      U x 1   mean vector
%   s      U x U   covariance matrix
% dyn      .       dynamics model object
%   pred   @       dynamics model function
%   on     E x 1   log std dev observation noise
%   pn     E x 1   log std dev process noise
% ctrl     .       controller object
%   fcn    @       controller function
%   U              dimension of control actions
%
% Copyright (C) 2008-2015 Carl Edward Rasmussen and Rowan McAllister 2015-03-20

D = length(s.m);                                    % number of state variables
E = dyn.E;              % number of state variables the dynamics model predicts
i = 1:D;                                     % indices of input state variables
j = D + (1:ctrl.U);                                   % indices of ctrl actions
k = max(j) + (1:E); K = max(k);                   % indices of predicted states
o = ctrl.U + E + (1:D);                          % indices to select next state
M = zeros(K,1); M(i) = s.m; S = zeros(K); S(i,i) = s.s;            % init M & S

[M(j), S(j,j), C, s] = ctrl.fcn(s, dyn);  % compute distr of the control signal
S(i,j) = C; S(j,i) = C';                                   % action covariances

ij = [i j];               % input to dynamics model is state (i) and action (j)
[M(k), S(k,k), C] = dyn.pred(M(ij), S(ij,ij));    % compute distr of next state
S(k,k) = S(k,k) + diag(exp(2*dyn.pn));                      % add process noise
q = S(ij,ij)*C; S(ij,k) = q; S(k,ij) = q';             % next state covariances

s.m = M(o); s.s = (S(o,o)+S(o,o)')/2;                       % select next state
if nargout > 1, a.m = M(j); a.s = (S(j,j)+S(j,j)')/2; end
