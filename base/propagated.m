function [s, C, a, dsds, dsdp, dCds, dCdp] = propagated(s, dyn, ctrl)

% Propagate the state distribution one time step forward with derivatives
%
% [s, C, a, dsds, dsdp, dCds, dCdp] = propagated(s, dyn, ctrl)
%
% s        .       state structure
%   m      F x 1   mean vector
%   s      F x F   covariance matrix
%   ?              possibly other fields representing additional information
% dyn      .       dynamics model object
%   D              dimension of the physical state
%   E              dimension of predictions from dyn model
%   pred   @       dynamics model function
%   pn     E x 1   log std dev process noise
% ctrl     .       controller object
%   fcn    @       controller function
%   is     .       struct indexing vectorized state distributions variables
%   np             number of parameters in the policy
%   ns             number of state distrib parameters (means and variances)
%   U              dimension of control actions
% C        F x F   inverse input covariance times input-output covariance
% a        .       action structure
%   m      U x 1   mean vector
%   s      U x U   covariance matrix
% dsds     S x S   vectorised derivative of output state wrt input state
% dsdp     S x P   vectorised derivative of output state wrt policy parameters
% dCds    FF x S   vectorised derivative of covariance wrt input state
% dCdp    FF x P   vectorised derivative of covariance wrt policy parameters
%
% Copyright (C) 2008-2015 Carl Edward Rasmussen and Rowan McAllister 2015-06-05

if nargout <= 3, [s, C, a] = propagate(s, dyn, ctrl); return; end
F = length(s.m); D = ctrl.D; E = dyn.E; U = ctrl.U;          % short hand names
Dz = F-D;                              % length of predicted information states
i = 1:D;                            % indices of physical state input variables
j = D+1:F;                                       % indices of information state
k = F + (1:U);                                     % indices of control actions
l = max(k) + (1:Dz);                   % indices of predicted information state
m = max([k,l]) + (1:E);                           % indices of predicted states
ij = [i j]; ik = [i k]; kl = [k l]; ijkl = [ij kl];                % short hand
o = [ik(end-D+E+1:end) m l];                        % ind. to select next state
K = max(m);

M = zeros(K,1); M(ij) = s.m; S = zeros(K); S(ij,ij) = s.s;               % init
ijij = sub2ind2(K,ij,ij); is = ctrl.is; ns = ctrl.ns; np = ctrl.np;
Mds = zeros(K,ns); Mds(ij,is.m) = eye(F); Sds = zeros(K*K,ns);
Sds(sub2ind(size(Sds),ijij(:),is.s(:))) = 1; Sds = symmetrised(Sds,1);
Mdp = zeros(K,np); Sdp = zeros(K*K,np);

% 1) Compute distribution of the control signal -------------------------------
ijkl2 = sub2ind2(K,ij,kl); klij = sub2ind2(K,kl,ij); klkl= sub2ind2(K,kl,kl);
[M(kl), S(kl,kl), A, s, Mds(kl,:), Sds(klkl,:), Ads, dsds, Mdp(kl,:), ...
                                         Sdp(klkl,:), Adp, dsdp] = ctrl.fcn(s);
q = S(ij,ij)*A; S(ij,kl) = q; S(kl,ij) = q';         % state-action covariances
Sds(ijkl2,:) = prodd(S(ij,ij),Ads) + prodd([],Sds(ijij,:),A);
Sds(klij,:) = transposed(Sds(ijkl2,:),F);
Sdp(ijkl2,:) = prodd(S(ij,ij),Adp) + prodd([],Sdp(ijij,:),A);
Sdp(klij,:) = transposed(Sdp(ijkl2,:),F);

% 2) Compute distribution of the next state -----------------------------------
[M(m), S(m,m), B, mdm, sdm, bdm, mds, sds, bds] = dyn.pred(M(ik), S(ik,ik));
S(m,m) = S(m,m) + diag(exp(2*dyn.pn));                      % add process noise
q = S(ijkl,ik)*B; S(ijkl,m) = q; S(m,ijkl) = q';
ikik = sub2ind2(K,ik,ik); mm = sub2ind2(K,m,m);
ijklik = sub2ind2(K,ijkl,ik); ijklm = sub2ind2(K,ijkl,m);
mijkl = sub2ind2(K,m,ijkl);
Mds(m,:)   = mdm*Mds(ik,:)+mds*Sds(ikik,:);
Sds(mm,:) = sdm*Mds(ik,:)+sds*Sds(ikik,:);
Bds        = bdm*Mds(ik,:)+bds*Sds(ikik,:);
Mdp(m,:)   = mdm*Mdp(ik,:)+mds*Sdp(ikik,:);
Sdp(mm,:) = sdm*Mdp(ik,:)+sds*Sdp(ikik,:);
Bdp        = bdm*Mdp(ik,:)+bds*Sdp(ikik,:);
Sds(ijklm,:) = prodd(S(ijkl,ik),Bds) + prodd([],Sds(ijklik,:),B);
Sdp(ijklm,:) = prodd(S(ijkl,ik),Bdp) + prodd([],Sdp(ijklik,:),B);
Sds(mijkl,:) = transposed(Sds(ijklm,:),numel(ijkl));
Sdp(mijkl,:) = transposed(Sdp(ijklm,:),numel(ijkl));

C = [eye(F) A [eye(F,D) A(:,1:U)]*B];                 % inv input var times cov
dCds = [zeros(F*F,ns); Ads; prodd([],[zeros(F*D,ns);Ads(1:F*U,:)],B) + ...
  prodd([eye(F,D) A(:,1:U)],Bds)];
dCdp = [zeros(F*F,np); Adp; prodd([],[zeros(F*D,np);Adp(1:F*U,:)],B) + ...
  prodd([eye(F,D) A(:,1:U)],Bdp)];

% 3) Select distribution of the next state ------------------------------------
s.m = M(o); s.s = (S(o,o)+S(o,o)')/2; C = C(ij,o);          % select next state
a.m = M(k); a.s = (S(k,k)+S(k,k)')/2;                          % control action
oo = sub2ind2(K,o,o);
dsds(is.m,:) = Mds(o,:); dsds(is.s,:) = Sds(oo,:);
dsdp(is.m,:) = Mdp(o,:); dsdp(is.s,:) = Sdp(oo,:);
dCds = dCds(sub2ind2(F,ij,o),:);
dCdp = dCdp(sub2ind2(F,ij,o),:);
