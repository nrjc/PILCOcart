function [L, dLdm, dLds, S2] = loss2(cost, m, s)
% Alternative pendulum loss function. The loss is (cos(theta)+1)/2 i.e.
% the angular distance to the upright position.
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

if ~isempty(cost.expl), b = cost.expl; else b = 0; end

% 1. Some precomputations
D0 = size(s,2); D = D0;                                  % state dimension
D1 = D0 + 2*length(cost.angle);           % state dimension (with sin/cos)

M = zeros(D1,1); M(1:D0) = m; S = zeros(D1); S(1:D0,1:D0) = s;
Mdm = [eye(D0); zeros(D1-D0,D)]; Sdm = zeros(D1*D1,D);
Mds = zeros(D1,D*D); Sds = kron(Mdm,Mdm);

% 2. Trigonometric augmentation
if D1-D0 > 0
  i = 1:D0; k = D0+1:D1;
  [M(k) S(k,k) C mdm sdm Cdm mds sds Cds] = gTrig(M(i),S(i,i),cost.angle);
  [S Mdm Mds Sdm Sds] = ...
               fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,i,k,D1);
end

% 4. Calculate loss!
a = length(M); L = (M(a)+1)/2; S2 = S(a,a)/4;
dLdm = Mdm(a,:)/2;
dLds = Mds(a,:)/2;
if (b~=0 || ~isempty(b)) && abs(S2)>1e-12
  L = L + b*sqrt(S2);
  dLdm = dLdm + b/sqrt(S2)*Sdm(a*a,:)/8;
  dLds = dLds + b/sqrt(S2)*Sds(a*a,:)/8;
end

% Fill in covariance matrix...and derivatives ----------------------------
function [S Mdm Mds Sdm Sds] = ...
                 fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,i,k,D)
X = reshape(1:D*D,[D D]); XT = X';                    % vectorised indices
I=0*X; I(i,i)=1; ii=X(I==1)'; I=0*X; I(k,k)=1; kk=X(I==1)';
I=0*X; I(i,k)=1; ik=X(I==1)'; ki=XT(I==1)';

Mdm(k,:)  = mdm*Mdm(i,:) + mds*Sdm(ii,:);                      % chainrule
Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:);
Sdm(kk,:) = sdm*Mdm(i,:) + sds*Sdm(ii,:);
Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:);
dCdm      = Cdm*Mdm(i,:) + Cds*Sdm(ii,:);
dCds      = Cdm*Mds(i,:) + Cds*Sds(ii,:);

S(i,k) = S(i,i)*C; S(k,i) = S(i,k)';                        % off-diagonal
SS = kron(eye(length(k)),S(i,i)); CC = kron(C',eye(length(i)));
Sdm(ik,:) = SS*dCdm + CC*Sdm(ii,:); Sdm(ki,:) = Sdm(ik,:);
Sds(ik,:) = SS*dCds + CC*Sds(ii,:); Sds(ki,:) = Sds(ik,:);