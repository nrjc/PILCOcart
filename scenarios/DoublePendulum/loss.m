function [L, dLds, S2] = loss(cost, s, plant)
% Double pendulum loss function. The loss is 1 - exp(-0.5*a*d^2), where
% "a" is a (positive) constant and "d^2" is the squared (Euclidean)
% distance between the tip of the pendulum and the upright position.
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
% s         state structure
% cost      cost structure
%   p       lengths of the pendula [ell1; ell2]
%   width   array of widths of the cost (summed together)
%   expl    exploration parameter
%   angle   array of angle indices
%   target  target state
%
% outputs:
% L     expected cost
% dLds  derivative of expected cost wrt state structure
%
% Copyright (C) 2008-2014 by Marc Deisenroth, Carl Edward Rasmussen,
% Joe Hall, Rowan McAllister 2014-07-29

cw = cost.width;
if ~isempty(cost.expl), b = cost.expl; else b = 0; end
is = plant.is; ns = plant.ns;

% 1. Some precomputations
D0 = length(s.m); D = D0;                                % state dimension
D1 = D0 + 2*length(cost.angle);           % state dimension (with sin/cos)

if isfield(s,'s'); ss = s.s; else ss = zeros(D); end
M = zeros(D1,1); M(1:D0) = s.m; S = zeros(D1); S(1:D0,1:D0) = ss;
Mds = zeros(D1,ns); Mds(:,is.m) = eye(D1,D0);
Sds = zeros(D1*D1,ns); Sds(:,is.s) = kron(Mds(:,is.m),Mds(:,is.m));
XT = reshape(1:D1*D1,[D1 D1])'; Sds = (Sds + Sds(XT(:),:))/2;

% 2. Define static penalty as distance from target setpoint
ell1 = cost.p(1); ell2 = cost.p(2); C = [ell1 0 ell2 0; 0 ell1 0 ell2];
Q = zeros(D1); Q(D+1:D+4,D+1:D+4) = C'*C;

% 3. Trigonometric augmentation
if D1-D0 > 0
  target = [cost.target(:); gTrig(cost.target(:), 0*ss, cost.angle)];
  
  i = 1:D0; k = D0+1:D1;
  [M(k) S(k,k) C mdm sdm Cdm mds sds Cds] = gTrig(M(i),S(i,i),cost.angle);
  [S Mds Sds] = fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mds,Sds,i,k,D1);
end

% 4. Calculate loss!
L = 0; dLds = zeros(1,ns); S2 = 0;
for i = 1:length(cw)                    % scale mixture of immediate costs
  cost.z = target; cost.W = Q/cw(i)^2;
  [r rdM rdS s2 s2dM s2dS] = lossSat(cost, M, S);
  
  L = L + r; S2 = S2 + s2;
  dLds = dLds + rdM(:)'*Mds + rdS(:)'*Sds;
  
  if (b~=0 || ~isempty(b)) && abs(s2)>1e-12
    L = L + b*sqrt(s2);
    dLds = dLds + b/sqrt(s2) * ( s2dM(:)'*Mds + s2dS(:)'*Sds )/2;
  end
end

% normalize
n = length(cw); L = L/n; dLds = dLds/n; S2 = S2/n;

% Fill in covariance matrix...and derivatives ----------------------------
function [S Mds Sds] = fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mds,Sds,i,k,D)
X = reshape(1:D*D,[D D]); XT = X';                    % vectorised indices
I=0*X; I(i,i)=1; ii=X(I==1)'; I=0*X; I(k,k)=1; kk=X(I==1)';
I=0*X; I(i,k)=1; ik=X(I==1)'; ki=XT(I==1)';

Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:);                      % chainrule
Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:);
dCds      = Cdm*Mds(i,:) + Cds*Sds(ii,:);

S(i,k) = S(i,i)*C; S(k,i) = S(i,k)';                        % off-diagonal
SS = kron(eye(length(k)),S(i,i)); CC = kron(C',eye(length(i)));
Sds(ik,:) = SS*dCds + CC*Sds(ii,:); Sds(ki,:) = Sds(ik,:);