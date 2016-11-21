function [M, S, V] = gpas(gpmodel, m, s)

% Compute single predictions for multiple GPs with uncertain inputs and
% obtain the joint distribution using an approximation of joint Gaussianity
% between input and output. This approximates the output covariance matrix.
% If dynmodel.nigp exists, individual noise contributions are added.
% Predictive variances contain uncertainty about the function, but no
% noise.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters                               [D+2 x  E ]
%   inputs  training inputs                                    [ n  x  D ]
%   target  training targets                                   [ n  x  E ]
%   nigp    (optional) individual noise variance terms         [ n  x  E ]
% m         mean of the test distribution                      [ D       ]
% s         covariance matrix of the test distribution         [ D  x  D ]
%
% M         mean of pred. distribution                         [ E       ]
% S         covariance of the pred. distribution               [ E  x  E ]
% V         inv(s) times covariance between input and output   [ D  x  E ]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
%                                Joe Hall and Andrew McHutchon, 2014-01-17

if isfield(gpmodel,'induce') && numel(gpmodel.induce)>0; x = gpmodel.induce; 
else x = gpmodel.inputs; end

[~, D, pE] = size(x); E = size(gpmodel.beta,2);
h = gpmodel.hyp; iK = gpmodel.iK; beta = gpmodel.beta; 

M = zeros(E,1); V = zeros(D,E); Si = zeros(E,1); M1 = zeros(E,1);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E
iL = exp(-2*[h.l]);inpiL = bsxfun(@times,inp,permute(iL,[3,1,2])); % N-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l));          % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;       % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D);               % Lambda^-1/2 * S * *Lambda^-1/2 + I
  
  t = in/B;                         % in.*t = (X-m) (S+L)^-1 (X-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il;
  c = exp(2*h(i).s)/sqrt(det(B));   % = sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  V(:,i) = tL'*lb*c + h(i).m;            % inv(s) times input-output covariance
  k = 2*h(i).s-sum(in.*in,2)/2;
  liBl = il*(B\il); xm = x(:,:,min(i,pE))'*lb*c; 
  a = diag(exp(2*h(i).l))*liBl*m*M1(i) + s*liBl*xm;

  % 2. Compute predictive variance (non-central moments) ---------------
  R = 2*s*diag(iL(:,i)) + eye(D); t = 1/sqrt(det(R));
  L = exp(bsxfun(@plus,k,k')+maha(inpiL(:,:,i),-inpiL(:,:,i),R\s/2));
  Si(i) = t*(beta(:,i)'*L*beta(:,i) - sum(sum(iK(:,:,i).*L)));  % variance
  Si(i) = Si(i) + 2*h(i).m'*(a - m*M1(i));
end
  
% 3. Compute cross covariances -------------------------------------------
S = V'*s*V; S = (S+S')/2;                    % symmetrize
S(eye(E)==1) = Si + exp(2*[h.s]') - M1.^2 + diag([h.m]'*s*[h.m]);
