function [L, dLds, S2] = loss(cost, s, is)
% Robotic unicycle loss function. The loss is 1 - exp(-0.5*a*d^2), where
% "a"is a (positive) constant and "d^2" is the squared difference between
% the current z-position of the top of the unicycle and the upright
% position.
%
% Compute the expected loss, averaged over a Gaussian state distibution,
% plus cost.expl times the standard deviation of the loss (averaged wrt
% the same Gaussian), where the exploration paramater cost.expl defaults
% to zero.
%
% Negative values of the exploration parameter are used to encourage
% exploration and positive values avoid regions of uncertainty in the
% policy. Derivatives are computed when desired.
%
% inputs:
% s         state structure
% cost      cost structure
%   p       parameters: [radius of wheel, length of rod]
%   width   array of widths of the cost (summed together)
%   expl    exploration parameter
% is        state struct indexer
%
% outputs:
% L     expected cost
% dLds  derivative of expected cost wrt. state structure
%
% Copyright (C) 2009-2014 Carl Edward Rasmussen, Marc Deisenroth,
% Philipp Hennig, Joe Hall, Rowan McAllister 2014-10-08

I6 = 8;  I9 = 10;                           % coordinates of theta and psi
Ixc = 6; Iyc = 7;                               % coordinates of xc and yc

cw = cost.width; rw = cost.p(1); r = cost.p(2);
if isfield(cost,'expl'), b = cost.expl; else b = 0; end
ns = numel(unwrap(s));

% 1. Some precomputations
D = length(s.m);                                         % state dimension
D0 = D + 2;             % state dimension (augmented with I6-I9 and I6+I9)
D1 = D0 + 8;                              % state dimension (with sin/cos)

P = eye(D+2,D); P(D+1:end,I6) = [1;-1]; P(D+1:end,I9) = [1;1];

if isfield(s,'s'); ss = s.s; else ss = zeros(D); end
M = zeros(D1,1); M(1:D0) = P*s.m; S = zeros(D1); S(1:D0,1:D0) = P*ss*P';
if nargout > 1
  Mds = zeros(D1,ns); Mds(:,is.m) = [P; zeros(D1-D0,D)];
  Sds = zeros(D1*D1,ns); Sds(:,is.s) = kron(Mds(:,is.m),Mds(:,is.m));
  XT = reshape(1:D1*D1,[D1 D1])'; Sds = (Sds + Sds(XT(:),:))/2;
end

% 2. Define static penalty as distance from target setpoint
Q = zeros(D+10);
C1 = [rw r/2 r/2];
Q([D+4 D+6 D+8],[D+4 D+6 D+8]) = 8*(C1'*C1);                          % dz
C2 = [1 -r];
Q([Ixc D+9],[Ixc D+9]) = 0.5*(C2'*C2);                                % dx
C3 = [1 -(r+rw)];
Q([Iyc D+3],[Iyc D+3]) = 0.5*(C3'*C3);                                % dy
Q(9,9) = (1/(4*pi))^2;                                    % yaw angle loss

target = zeros(D1,1); target([D+4 D+6 D+8 D+10]) = 1;    % target setpoint

% 3. Trigonometric augmentation
i = 1:D0; k = D0+1:D1;
[M(k) S(k,k) C mdm sdm Cdm mds sds Cds] = ...
  gTrig(M(i),S(i,i),[I6 D+1 D+2 I9]);
if nargout > 1
  [S Mds Sds] = fillIn(i,k,D1,S,C,mdm,sdm,Cdm,mds,sds,Cds,Mds,Sds);
else
  S = fillIn(i,k,D1,S,C);
end

% 4. Calculate loss!
L = 0; dLds = zeros(1,ns); S2 = 0;
for i = 1:length(cw)                    % scale mixture of immediate costs
  cost.z = target; cost.W = Q/cw(i)^2;
  [r rdM rdS s2 s2dM s2dS] = lossSat(cost, M, S);
  
  L = L + r; S2 = S2 + s2;
  if nargout > 1; dLds = dLds + rdM(:)'*Mds + rdS(:)'*Sds; end
  
  if (b~=0 || ~isempty(b)) && abs(s2)>1e-12
    L = L + b*sqrt(s2);
    if nargout > 1
      dLds = dLds + b/sqrt(s2) * ( s2dM(:)'*Mds + s2dS(:)'*Sds )/2;
    end
  end
end

% normalize
n = length(cw); L = L/n; dLds = dLds/n; S2 = S2/n;


% Fill in covariance matrix...and derivatives ----------------------------
function [S Mds Sds] = fillIn(i,k,D,S,C,mdm,sdm,Cdm,mds,sds,Cds,Mds,Sds)
S(i,k) = S(i,i)*C; S(k,i) = S(i,k)';                        % off-diagonal
if nargout < 2; return; end

X = reshape(1:D*D,[D D]); XT = X';                    % vectorised indices
I=0*X; I(i,i)=1; ii=X(I==1)'; I=0*X; I(k,k)=1; kk=X(I==1)';
I=0*X; I(i,k)=1; ik=X(I==1)'; ki=XT(I==1)';

Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:);                      % chainrule
Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:);
dCds      = Cdm*Mds(i,:) + Cds*Sds(ii,:);

SS = kron(eye(length(k)),S(i,i)); CC = kron(C',eye(length(i)));
Sds(ik,:) = SS*dCds + CC*Sds(ii,:); Sds(ki,:) = Sds(ik,:);