function [L, dLdm, dLds, S2] = loss(cost, m, s)
%% Brief Description and Interface
% *Summary:* Cart-pole loss function. The loss is 1 - exp(-0.5*a*d^2), where "a" is a
% (positive) constant and "d^2" is the squared (Euclidean) distance
% between the tip of the pendulum and the upright position.
%
% If the exploration parameter "b" is not present (or if it is zero) then
% the expected loss is computed, averaging over the Gaussian distribution
% of the state, with mean "mu" and covariance matrix "Sigma". If it is
% present, then sum of the average loss and "b" times the std. deviation
% of the loss is returned. Negative values of "b" are used to encourage
% exploration and positive values avoid regions of uncertainty in the
% policy. Derivatives of these quantities are computed when desired. See
% also loss.pdf.
%
% inputs:
% m       mean of state distribution
% s       covariance matrix for the state distribution
% cost    cost structure
%   cost.p       length of the pendulum
%   cost.width   array of widths of the cost (summed together)
%   cost.expl    exploration parameter
%   cost.angle   array of angle indices
%   cost.target  target state
%
% outputs:
% L     expected cost
% dLdm  derivative of expected cost wrt. state mean vector
% dLds  derivative of expected cost wrt. state covariance matrix
%
% Copyright (C) 2008-2012 by and Marc Deisenroth Carl Edward Rasmussen,
% 2012-06-26. Edited by Joe Hall 2012-10-02.


%% Code
cw = cost.width;
if ~isempty(cost.expl), b = cost.expl; else b = 0; end

% 1. Some precomputations
D0 = size(s,2); D = D0;                                  % state dimension
D1 = D0 + 2*length(cost.angle);           % state dimension (with sin/cos)

M = zeros(D1,1); M(1:D0) = m; S = zeros(D1); S(1:D0,1:D0) = s;
Mdm = [eye(D0); zeros(D1-D0,D0)]; Sdm = zeros(D1*D1,D0);
Mds = zeros(D1,D0*D0); Sds = kron(Mdm,Mdm);

% 2. Define static penalty as distance from target setpoint
ell = cost.p;
Q = zeros(D1); Q([1 D+1],[1 D+1]) = [1 ell]'*[1 ell]; Q(D+2,D+2) = ell^2;

% 3. Trigonometric augmentation
if D1-D0 > 0
  target = [cost.target(:); gTrig(cost.target(:), 0*s, cost.angle)];
    
  i = 1:D0; k = D0+1:D1;
  [M(k) S(k,k) C mdm sdm Cdm mds sds Cds] = gTrig(M(i),S(i,i),cost.angle);
  
  X = reshape(1:D1*D1,[D1 D1]); XT = X';              % vectorised indices
  I=0*X; I(i,i)=1; ii=X(I==1)'; I=0*X; I(k,k)=1; kk=X(I==1)';
  I=0*X; I(i,k)=1; ik=X(I==1)'; ki=XT(I==1)';

  Mdm(k,:)  = mdm*Mdm(i,:) + mds*Sdm(ii,:);                    % chainrule
  Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:);
  Sdm(kk,:) = sdm*Mdm(i,:) + sds*Sdm(ii,:);
  Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:);
  dCdm      = Cdm*Mdm(i,:) + Cds*Sdm(ii,:);
  dCds      = Cdm*Mds(i,:) + Cds*Sds(ii,:);

  S(i,k) = S(i,i)*C; S(k,i) = S(i,k)';                      % off-diagonal
  SS = kron(eye(length(k)),S(i,i)); CC = kron(C',eye(length(i)));
  Sdm(ik,:) = SS*dCdm + CC*Sdm(ii,:); Sdm(ki,:) = Sdm(ik,:);
  Sds(ik,:) = SS*dCds + CC*Sds(ii,:); Sds(ki,:) = Sds(ik,:);
end

% 4. Calculate loss!
L = 0; dLdm = zeros(1,D0); dLds = zeros(1,D0*D0); S2 = 0;
for i = 1:length(cw)                    % scale mixture of immediate costs
    cost.z = target; cost.W = Q/cw(i)^2;
  [r rdM rdS s2 s2dM s2dS] = lossSat(cost, M, S);
  
  L = L + r; S2 = S2 + s2;
  dLdm = dLdm + rdM(:)'*Mdm + rdS(:)'*Sdm;
  dLds = dLds + rdM(:)'*Mds + rdS(:)'*Sds;
  
  if (b~=0 || ~isempty(b)) && abs(s2)>1e-12
    L = L + b*sqrt(s2);
    dLdm = dLdm + b/sqrt(s2) * ( s2dM(:)'*Mdm + s2dS(:)'*Sdm )/2;
    dLds = dLds + b/sqrt(s2) * ( s2dM(:)'*Mds + s2dS(:)'*Sds )/2;
  end
end

% normalize
n = length(cw); L = L/n; dLdm = dLdm/n; dLds = dLds/n; S2 = S2/n;