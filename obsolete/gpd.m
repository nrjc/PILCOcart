function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpd(gpmodel, m, s)                      
% Compute joint predictions and derivatives for multiple GPs with uncertain
% inputs. This version uses a linear + bias mean function and also
% computes derivatives of the output moments w.r.t the input moments.
% Predictive variances contain uncertainty about the function, but no noise.
%
% gpmodel   dynamics model struct
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
% m         D by 1 vector, mean of the test distribution
% s         D by D covariance matrix of the test distribution
%
% M         E-by-1 vector, mean of pred. distribution
% S         E-by-E matrix, covariance of the pred. distribution
% V         D-by-E inv(s) times covariance between input and prediction
% dMdm      E-by-D, deriv of output mean w.r.t. input mean 
% dSdm      E^2-by-D, deriv of output covariance w.r.t input mean
% dVdm      D*E-by-D, deriv of input-output cov w.r.t. input mean
% dMds      E-by-D^2, deriv of ouput mean w.r.t input covariance
% dSds      E^2-by-D^2, deriv of output cov w.r.t input covariance
% dVds      D*E-by-D^2, deriv of inv(s)*input-output covariance w.r.t input cov
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth, 
%  Andrew McHutchon, Joe Hall, Rowan McAllister 2014-11-14

if isfield(gpmodel,'induce') && numel(gpmodel.induce)>0; x = gpmodel.induce; 
else x = gpmodel.inputs; end

[n, D, pE] = size(x); E = size(gpmodel.beta,2);
h = gpmodel.hyp; iK = gpmodel.iK; beta = gpmodel.beta;
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end
if ~isfield(h,'b'); [h.b] = deal(0); end

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);      % initialize
dMds = zeros(E,D,D); dSdm = zeros(E,E,D); a = zeros(D,E); M1 = zeros(E,1);
dSds = zeros(E,E,D,D); dVds = zeros(D,E,D,D); T = zeros(D); dadm = zeros(D,D,E);
dM1dm = zeros(E,D); dads = zeros(D,E,D,D);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  xi = x(:,:,min(i,pE));
  il = diag(exp(-h(i).l));                  % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;               % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D); liBl = il/B*il;       % liBl = (Lambda + S)^-1
  
  t = in/B;                                 % in.*t = (x-m) (S+L)^-1 (x-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il; tlb = bsxfun(@times,tL,lb);
  c = exp(2*h(i).s)/sqrt(det(B));           % sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  V1 = tL'*lb*c; V(:,i) = V1 + h(i).m;   % inv(s) times input-output covariance
  dMds(i,:,:) = c*tL'*tlb/2 - liBl*M1(i)/2;
  for d = 1:D
    dVds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - liBl*V1(d)/2 - V1*liBl(d,:);
  end
   k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
   xm = xi'*lb*c; L2iBL = eye(D)/(s*il*il+eye(D));
   LiBLxm = liBl*xm; sLx = s*liBl*xi';
   a(:,i) = L2iBL*m*M1(i) + s*LiBLxm;
   dadm(:,:,i) = L2iBL*(M1(i)*eye(D) + m*V1') + s*liBl*xi'*bsxfun(@times,tL,lb)*c;
   dads(:,i,:,:) = -bsxfun(@times,L2iBL,permute(liBl*m,[2,3,1]))*M1(i) ...
                                          + bsxfun(@times,L2iBL*m,dMds(i,:,:));
   for d=1:D; dads(d,i,d,:) = dads(d,i,d,:) + permute(LiBLxm,[2,3,4,1]); end
   tLtlb = bsxfun(@times,tL,permute(tlb,[1,3,2])); 
   Lbc = bsxfun(@times,lb,permute(liBl*c,[3,1,2]));
   dads(:,i,:,:) = squeeze(dads(:,i,:,:)) - bsxfun(@times,s*liBl,permute(LiBLxm,[2,3,1]))...
                                + etprod('123',sLx,'14',c*tLtlb-Lbc,'423')/2;
   
   dM1dm(i,:) = V1;
end  
dMdm = V'; dVdm = 2*permute(dMds,[2 1 3]);                  % derivatives wrt m

iL = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(iL,[3,1,2])); % N-by-D-by-E
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
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
    T = (t*T-S1*diag(iL(:,i)+iL(:,j))/R)/2;
    T = T - reshape(M1(i)*dMds(j,:,:) + M1(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T + reshape(h(i).m'*reshape(dads(:,j,:,:),D,D^2),D,D)...
         + reshape(h(j).m'*reshape(dads(:,i,:,:),D,D^2),D,D)...
          - squeeze(h(i).m'*m*dMds(j,:,:)) - squeeze(h(j).m'*m*dMds(i,:,:)) ...
          + h(i).m*h(j).m';
    dSds(j,i,:,:) = permute(dSds(i,j,:,:),[1,2,4,3]);
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);            % add signal variance to diagonal
end

S = S - M1*M1' + [h.m]'*s*[h.m];                           % centralize moments

dMds = reshape(dMds,[E D*D]);                           % vectorise derivatives
dSds = reshape(dSds,[E*E D*D]); dSdm = reshape(dSdm,[E*E D]);
dVds = reshape(dVds,[D*E D*D]); dVdm = reshape(dVdm,[D*E D]);

