function [s, C, a] = propagate(s, dyn, ctrl)

% Propagate the state distribution one time step forward.
%
% [s, C, a] = propagate(s, dyn, ctrl)
%
% s        .       state structure
%   m      F x 1   mean vector
%   s      F x F   covariance matrix
%   ?              possibly other fields representing additional information
% dyn      .       dynamics model object
%   E              number of predicted state variables
%   pred   @       dynamics model function
%   pn     E x 1   log std dev process noise
% ctrl     .       controller object
%   D              number of physical state variables
%   Dz             number of infomation state variables
%   F              number of state variables (physical + infomation)
%   fcn    @       controller function
%   U              number of control actions
% C        F x F   inverse input covariance times input-output covariance
% a        .       action structure
%   m      U x 1   mean vector
%   s      U x U   covariance matrix
%
% Copyright (C) 2008-2015 Carl Edward Rasmussen and Rowan McAllister 2016-01-12

D = ctrl.D; Dz = ctrl.Dz; E = dyn.E; F = ctrl.F; U = ctrl.U;       % short hand
i = 1:D;                            % indices of physical state input variables
j = D+1:F;                                       % indices of information state
k = F + (1:U);                                     % indices of control actions
l = max(k) + (1:Dz);                   % indices of predicted information state
m = max([k,l]) + (1:E);                           % indices of predicted states
ij = [i j]; ik = [i k]; kl = [k l]; ijkl = [ij kl];                % short hand
o = [ik(end-D+E+1:end) m l];                     % indices to select next state
M = zeros(max(m),1); M(ij) = s.m; S = zeros(max(m)); S(ij,ij) = s.s;     % init

[M(kl), S(kl,kl), A, s] = ctrl.fcn(s);           % control signal and inf state
q = S(ij,ij)*A; S(ij,kl) = q; S(kl,ij) = q';               % action covariances

[M(m), S(m,m), B] = dyn.pred(M(ik), S(ik,ik));    % compute distr of next state
S(m,m) = S(m,m) + diag(exp(2*dyn.pn));                      % add process noise
q = S(ijkl,ik)*B; S(ijkl,m) = q; S(m,ijkl) = q';       % next state covariances

C = [eye(F) A [eye(F,D) A(:,1:U)]*B];                 % inv input var times cov
% C_exact_ZC = [eye(F,D),A(:,U+1:end)]*C_gph;

s.m = M(o); s.s = (S(o,o)+S(o,o)')/2; C = C(ij,o);          % select next state
a.m = M(k); a.s = (S(k,k)+S(k,k)')/2;                          % control action
