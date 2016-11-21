function [M, S, C] = gpp(gpmodel, m, s)
% Compute joint predictions for multiple GPs with uncertain inputs. This version
% uses a linear + bias mean function. Predictive variances contain uncertainty 
% about the function, but no noise.
%
% dynmodel  dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%     .m    D-by-1 linear weights for the GP mean
%     .b    1-by-1 biases for the GP mean
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%   iK      n-by-n-by-E, inverse covariance matrix
%   beta    n-by-E, iK*(targets - mean function of inputs)
%
% m         D-by-1, mean of the test distribution
% s         D-by-D, covariance matrix of the test distribution
%
% M         E-by-1, mean of pred. distribution 
% S         E-by-E, covariance of the pred. distribution             
% C         D-by-E, inv(s) times covariance between input and output 
%
% Copyright (C) 2008-2014 by Carl Edward Rasmussen, Marc Deisenroth,
% Andrew McHutchon, Joe Hall, Rowan McAllister 2014-11-14

if isfield(gpmodel,'induce') && numel(gpmodel.induce)>0; x = gpmodel.induce; 
else x = gpmodel.inputs; end

[n, D, pE] = size(x); E = size(gpmodel.beta,2);
h = gpmodel.hyp; iK = gpmodel.iK; beta = gpmodel.beta;
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end
if ~isfield(h,'b'); [h.b] = deal(0); end

M = zeros(E,1); C = zeros(D,E); S = zeros(E);
k = zeros(n,E); a = zeros(D,E); M1 = zeros(E,1);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l));          % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;       % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D);               % Lambda^-1/2 * S * *Lambda^-1/2 + I
  
  t = in/B;                         % in.*t = (X-m) (S+L)^-1 (X-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il;
  c = exp(2*h(i).s)/sqrt(det(B));   % = sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  C(:,i) = tL'*lb*c + h(i).m;            % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
  liBl = il*(B\il); xm = x(:,:,min(i,pE))'*lb*c; 
  a(:,i) = (s*il*il+eye(D))\m*M1(i) + s*liBl*xm;
end

iL = exp(-2*[h.l]);inpiL = bsxfun(@times,inp,permute(iL,[3,1,2])); % N-by-D-by-E
for i=1:E                  % compute predictive covariance, non-central moments
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); t = 1/sqrt(det(R));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(inpiL(:,:,i),-inpiL(:,:,j),R\s/2));
    S(i,j) = beta(:,i)'*L*beta(:,j)*t;                  % variance of the mean
    S(i,j) = S(i,j) + h(i).m'*(a(:,j) - m*M1(j)) + h(j).m'*(a(:,i) - m*M1(i));
    S(j,i) = S(i,j);
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s) - t*sum(sum(iK(:,:,i).*L)); % last L has i==j
end

S = S - M1*M1' + [h.m]'*s*[h.m];                           % centralize moments
