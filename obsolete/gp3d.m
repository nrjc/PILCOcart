function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gp3d(dynmodel, m, s)
                                                 
% Compute joint predictions and derivatives for multiple GPs with uncertain
% inputs. If dynmodel.nigp exists, individial noise contributions are added.
% Predictive variances contain uncertainty about the function, but no noise.
%
% dynmodel  dynamics model struct
%   hyp     D+2 by E vector of log-hyper-parameters
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%   nigp    optional, n by E matrix of individual noise variance terms
% m         D by 1 vector, mean of the test distribution
% s         D by D covariance matrix of the test distribution
%
% M         E by 1 vector, mean of pred. distribution
% S         E by E matrix, covariance of the pred. distribution
% V         D by E inv(s) times covariance between input and prediction
% dMdm      E by D matrix of output mean by input mean partial derivatives
% dSdm      E by E by D matrix of output covariance by input mean derivatives
% dVdm      D by E by D matrix of input-output cov by input mean derivatives
% dMds      E by D by D matrix of ouput mean by input covariance derivatives
% dSds      E by E by D by D matrix of output cov by input covariance derivs
% dVds      D by E by D by D matrix of inv(s)*input-output covariance by input cov der
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth, 
%  Andrew McHutchon, & Joe Hall 2012-07-09

if nargout < 4; [M S V] = gp3(dynmodel, m, s); return; end     % no derivs, gp0

persistent K iK beta oldh oldn;
[n, D] = size(dynmodel.inputs);    % number of examples and dimension of inputs
[n, E] = size(dynmodel.target);      % number of examples and number of outputs
h = dynmodel.hyp; x = dynmodel.inputs;
if ~isfield(h,'m') && ~isfield(h,'b'); [M S V] = gp0d(dynmodel,m,s); return;
elseif ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); 
elseif ~isfield(h,'b'); [h.b] = deal(0); end

% if necessary: re-compute cashed variables
if isempty(iK) || any(unwrap(h) ~= oldh) || n ~= oldn
  oldh = unwrap(h); oldn = n; iK = zeros(n,n,E); K = zeros(n,n,E); beta = zeros(n,E);
  
  for i=1:E                                              % compute K and inv(K)
    inp = bsxfun(@times,dynmodel.inputs,exp(-h(i).l'));
    K(:,:,i) = exp(2*h(i).s-maha(inp,inp)/2);
    if isfield(dynmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n) + diag(dynmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n))';
    end
    iK(:,:,i) = L'\(L\eye(n));
    y = dynmodel.target(:,i) - x*h(i).m - h(i).b;
    beta(:,i) = L'\(L\y);
  end
end

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);      % initialize
dMds = zeros(E,D,D); dSdm = zeros(E,E,D); a = zeros(D,E); M1 = zeros(E,1);
dSds = zeros(E,E,D,D); dVds = zeros(D,E,D,D); T = zeros(D); dadm = zeros(D,D,E);
dM1dm = zeros(E,D); dads = zeros(D,E,D,D);

inp = bsxfun(@minus,x,m');                    % centralize inputs

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  iL = diag(exp(-h(i).l));
  in = inp*iL;
  B = iL*s*iL+eye(D); LiBL = iL/B*iL;  
  
  t = in/B;     % in.*t = (x-m) (S+L)^-1 (x-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*iL; tlb = bsxfun(@times,tL,lb);
  c = exp(2*h(i).s)/sqrt(det(B));   % sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  V1 = tL'*lb*c; V(:,i) = V1 + h(i).m;   % inv(s) times input-output covariance
  dMds(i,:,:) = c*tL'*tlb/2 - LiBL*M1(i)/2;
  for d = 1:D
    dVds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - LiBL*V1(d)/2 ...
                                                            - V1*LiBL(d,:);
  end
   k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
   xm = x'*lb*c; L2iBL = diag(exp(2*h(i).l))*LiBL; 
   LiBLxm = LiBL*xm; sLx = s*LiBL*x';
   a(:,i) = L2iBL*m*M1(i) + s*LiBLxm;
   dadm(:,:,i) = L2iBL*(M1(i)*eye(D) + m*V1') + s*LiBL*x'*bsxfun(@times,tL,lb)*c;
   dads(:,i,:,:) = -bsxfun(@times,L2iBL,permute(LiBL*m,[2,3,1]))*M1(i) + bsxfun(@times,L2iBL*m,dMds(i,:,:));
   for d=1:D; dads(d,i,d,:) = dads(d,i,d,:) + permute(LiBLxm,[2,3,4,1]); end
   tLtlb = bsxfun(@times,tL,permute(tlb,[1,3,2])); Lbc = bsxfun(@times,lb,permute(LiBL*c,[3,1,2]));
   dads(:,i,:,:) = squeeze(dads(:,i,:,:)) - bsxfun(@times,s*LiBL,permute(LiBLxm,[2,3,1]))...
        + etprod('123',sLx,'14',c*tLtlb-Lbc,'423')/2;
   
   dM1dm(i,:) = V1;
end  
dMdm = V'; dVdm = 2*permute(dMds,[2 1 3]);                  % derivatives wrt m

il = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(il,[3,1,2])); % N-by-D-by-E
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(il(:,i)+il(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    if i==j
      iKL = iK(:,:,i).*L; s1iKL = sum(iKL,1); s2iKL = sum(iKL,2);  
      S(i,j) = t*(beta(:,i)'*L*beta(:,i) - sum(s1iKL));
      zi = ii/R; 
      bibLi = L'*beta(:,i).*beta(:,i); cbLi = L'*bsxfun(@times, beta(:,i), zi);
      r = (bibLi'*zi*2 - (s2iKL' + s1iKL)*zi)*t;
      for d = 1:D
        T(d,1:d) = 2*(zi(:,1:d)'*(zi(:,d).*bibLi) + ...
            cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) ...
                                                   - zi(:,1:d)'*(iKL*zi(:,d)));
        T(1:d,d) = T(d,1:d)';
      end
    else
      zi = ii/R; zj = ij/R;
      S(i,j) = beta(:,i)'*L*beta(:,j)*t;

      bibLj = L*beta(:,j).*beta(:,i); bjbLi = L'*beta(:,i).*beta(:,j);
      cbLi = L'*bsxfun(@times, beta(:,i), zi);
      cbLj = L*bsxfun(@times, beta(:,j), zj);

      r = (bibLj'*zi+bjbLi'*zj)*t;
      for d = 1:D
        T(d,1:d) = zi(:,1:d)'*(zi(:,d).*bibLj) + ...
          cbLi(:,1:d)'*(zj(:,d).*beta(:,j)) + zj(:,1:d)'*(zj(:,d).*bjbLi) + ...
                                             cbLj(:,1:d)'*(zi(:,d).*beta(:,i));
        T(1:d,d) = T(d,1:d)';
      end
    end
    S1 = S(i,j);
    S(i,j) = S1 + h(i).m'*(a(:,j) - m*M1(j)) + h(j).m'*(a(:,i) - m*M1(i));
    S(j,i) = S(i,j);
    
    dSdm(i,j,:) = r - M1(i)*dM1dm(j,:)-M1(j)*dM1dm(i,:) + h(i).m'*(dadm(:,:,j) ...
        - M1(j)*eye(D) - m*dM1dm(j,:)) + h(j).m'*(dadm(:,:,i) - M1(i)*eye(D) - m*dM1dm(i,:));
    dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S1*diag(il(:,i)+il(:,j))/R)/2;
    T = T - reshape(M1(i)*dMds(j,:,:) + M1(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T + reshape(h(i).m'*reshape(dads(:,j,:,:),D,D^2),D,D)...
         + reshape(h(j).m'*reshape(dads(:,i,:,:),D,D^2),D,D)...
          - squeeze(h(i).m'*m*dMds(j,:,:)) - squeeze(h(j).m'*m*dMds(i,:,:)) ...
          + h(i).m*h(j).m';
    dSds(j,i,:,:) = permute(dSds(i,j,:,:),[1,2,4,3]);
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);
end

S = S - M1*M1' + [h.m]'*s*[h.m];                           % centralize moments

dMds = reshape(dMds,[E D*D]);                           % vectorise derivatives
dSds = reshape(dSds,[E*E D*D]); dSdm = reshape(dSdm,[E*E D]);
dVds = reshape(dVds,[D*E D*D]); dVdm = reshape(dVdm,[D*E D]);