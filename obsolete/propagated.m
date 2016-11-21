function [s, dsds, dsdp] = propagated(s, dyn, ctrl)

% Propagate the state distribution one time step forward with derivatives
%
% [s, dsds, dsdp] = propagated(s, dyn, ctrl)
%
% s        .       state structure
%   m      D x 1   mean vector
%   s      D x D   covariance matrix
% dyn      .       dynamics model object
%   pred   @       dynamics model function
%   on     E x 1   log std dev observation noise
%   pn     E x 1   log std dev process noise
% ctrl     .       controller object
%   fcn    @       controller function
%   is     .       struct indexing vectorized state distributions variables
%   np             number of parameters in the policy
%   ns             number of state distrib parameters (means and variances)
%   U              dimension of control actions
% dsds     S x S   vectorised derivative of output state wrt input state
% dsdp     S x P   vectorised derivative of output state wrt policy parameters
%
% Copyright (C) 2008-2015 Carl Edward Rasmussen and Rowan McAllister 2015-03-24

if nargout <= 1, s = propagate(s, dyn, ctrl); return; end
D = length(s.m);                                    % number of state variables
E = dyn.E;              % number of state variables the dynamics model predicts
i = 1:D;                                     % indices of input state variables
j = D + (1:ctrl.U);                                    % indices of ctrl signal
k = max(j) + (1:E); K = max(k);                   % indices of predicted states
o = ctrl.U + E + (1:D);                          % indices to select next state

M = zeros(K,1); M(i) = s.m; S = zeros(K); S(i,i) = s.s;            % init M & S
ii = sub2ind2(K,i,i); is = ctrl.is; ns = ctrl.ns; np = ctrl.np;
Mds = zeros(K,ns); Mds(i,is.m) = eye(D); Sds = zeros(K*K,ns);
Sds(sub2ind(size(Sds),ii(:),is.s(:))) = 1; Sds = symmetrised(Sds,1);
Mdp = zeros(K,np); Sdp = zeros(K*K,np);

% 1) Compute distribution of the control signal -------------------------------
ij = sub2ind2(K,i,j); ji = sub2ind2(K,j,i); jj = sub2ind2(K,j,j);
[M(j), S(j,j), C, s, Mds(j,:), Sds(jj,:), cds, dsds, Mdp(j,:), Sdp(jj,:), ...
                                                 cdp, dsdp] = ctrl.fcn(s, dyn);
S(i,j) = C; S(j,i) = C';                             % state-action covariances
Sds(ij,:) = cds; Sds(ji,:) = transposed(cds,D);
Sdp(ij,:) = cdp; Sdp(ji,:) = transposed(cdp,D);

% 2) Compute distribution of the next state -----------------------------------
ij = [i j];               % input to dynamics model is state (i) and action (j)
[M(k), S(k,k), C, mdm, sdm, cdm, mds, sds, cds] = dyn.pred(M(ij),S(ij,ij));
S(k,k) = S(k,k) + diag(exp(2*dyn.pn));                      % add process noise
q = S(ij,ij)*C; S(ij,k) = q; S(k,ij) = q';
IJ=sub2ind2(K,ij,ij); kk=sub2ind2(K,k,k);
ijk=sub2ind2(K,ij,k); kij =sub2ind2(K,k,ij);
Mds(k,:) = mdm*Mds(ij,:)+mds*Sds(IJ,:); Mdp(k,:) = mdm*Mdp(ij,:)+mds*Sdp(IJ,:);
Sds(kk,:)= sdm*Mds(ij,:)+sds*Sds(IJ,:); Sdp(kk,:)= sdm*Mdp(ij,:)+sds*Sdp(IJ,:);
Cds      = cdm*Mds(ij,:)+cds*Sds(IJ,:); Cdp      = cdm*Mdp(ij,:)+cds*Sdp(IJ,:);
Sds(ijk,:) = prodd(S(ij,ij),Cds) + prodd([],Sds(IJ,:),C);
Sdp(ijk,:) = prodd(S(ij,ij),Cdp) + prodd([],Sdp(IJ,:),C);
Sds(kij,:) = transposed(Sds(ijk,:),numel(ij));
Sdp(kij,:) = transposed(Sdp(ijk,:),numel(ij));

% 3) Select distribution of the next state ------------------------------------
s.m = M(o); s.s = (S(o,o)+S(o,o)')/2;                       % select next state
oo = sub2ind2(K,o,o);
dsds(is.m,:) = Mds(o,:); dsds(is.s,:) = Sds(oo,:);
dsdp(is.m,:) = Mdp(o,:); dsdp(is.s,:) = Sdp(oo,:);
