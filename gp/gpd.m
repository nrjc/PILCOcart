function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds] = gpd(gp, m, s)
% Compute joint predictions and derivatives for multiple GPs with uncertain
% inputs. This version uses a linear + bias mean function and also
% computes derivatives of the output moments w.r.t the input moments.
% Predictive variances contain uncertainty about the function, but no noise.
%
% gp                 gaussian process model struct
%   hyp     1 x E    struct array of GP hyper-parameters
%     l     D x 1    log lengthscales
%     s     1 x 1    log signal standard deviation
%     n     1 x 1    log noise standard deviation
%     m     D x 1    linear weights for the GP mean
%     b     1 x 1    biases for the GP mean
%   inputs  n x D    matrix of training inputs
%   target  n x E    matrix of training targets
%   W     n x n x E  inverse covariance matrix
%   beta    n x E    iK*(targets - mean function of inputs)
% m         D x 1    vector, mean of the test distribution
% s         D x D    covariance matrix of the test distribution
% M         E x 1    vector, mean of pred. distribution
% S         E x E    matrix, covariance of the pred. distribution
% C         D x E    inv(s) times covariance between input and prediction
% dMdm      E x D    deriv of output mean w.r.t. input mean
% dSdm    E*E x D    deriv of output covariance w.r.t input mean
% dCdm    D*E x D    deriv of input-output cov w.r.t. input mean
% dMds      E x D*D  deriv of ouput mean w.r.t input covariance
% dSds    E*E x D*D  deriv of output cov w.r.t input covariance
% dCds    D*E x D*D  deriv of inv(s)*input-output covariance w.r.t input cov
%
% Copyright (C) 2015, Carl Edward Rasmussen, Rowan McAllister 2015-07-10

if numel(gp.induce) > 0, x = gp.induce; else x = gp.inputs; end

[n, D, pE] = size(x); DD = D*D; E = size(gp.beta,2);
h = gp.hyp; iK = gp.W; beta = gp.beta;

k = zeros(n,E); M = zeros(E,1); C = zeros(D,E); S = zeros(E);      % initialize
dMds = zeros(E,D,D); dSdm = zeros(E,E,D); T = zeros(D);
dSds = zeros(E,E,D,D); dCds = zeros(D,E,D,D);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l));                                        % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;                             % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D); liBl = il/B*il;                  % liBl = (Lambda + S)^-1
  t = in/B;                                      % in.*t = (x-m) (S+L)^-1 (x-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il; tlb = bsxfun(@times,tL,lb);
  c = exp(2*h(i).s)/sqrt(det(B));                     % sf2/sqrt(det(S*iL + I))
  M(i) = sum(lb)*c;
  C(:,i) = tL'*lb*c;                     % inv(s) times input-output covariance
  dMds(i,:,:) = c*tL'*tlb/2 - liBl*M(i)/2;
  for d = 1:D
    dCds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - liBl*C(d,i)/2 - ...
                                                              C(:,i)*liBl(d,:);
  end
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
end
dMdm = C'; dCdm = 2*permute(dMds,[2 1 3]);                  % derivatives wrt m

iL = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(iL,[3,1,2]));
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
          cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) - ...
                                                     zi(:,1:d)'*(iKL*zi(:,d)));
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
    S(j,i) = S(i,j);
    
    dSdm(i,j,:) = r - M(i)*dMdm(j,:)-M(j)*dMdm(i,:);
    dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S(i,j)*diag(iL(:,i)+iL(:,j))/R)/2;
    T = T - reshape(M(i)*dMds(j,:,:) + M(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T;
    dSds(j,i,:,:) = permute(dSds(i,j,:,:),[1,2,4,3]);
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);            % add signal variance to diagonal
end

dMds = reshape(dMds,[E DD]);                            % vectorise derivatives
dSds = reshape(dSds,[E*E DD]); dSdm = reshape(dSdm,[E*E D]);
dCds = reshape(dCds,[D*E DD]); dCdm = reshape(dCdm,[D*E D]);

% linear components
dMdm = dMdm + [h.m]';
dCtdm = transposed(dCdm,D);
dSdm = dSdm + prodd([],dCtdm,s*[h.m]) + prodd([h.m]'*s,dCdm);
dCtds = transposed(dCds,D);
dSds = dSds + prodd([],dCtds,s*[h.m]) + prodd([h.m]'*s,dCds) + ...
     prodd(C','eye',[h.m]) + prodd([h.m]','eye',C) + prodd([h.m]','eye',[h.m]);

% enforce symmetry?
dSds = symmetrised(dSds,2);
dCds = symmetrised(dCds,2);

S = S - M*M';                                              % centralize moments
M = M + [h.m]'*m + [h.b]';                           % add linear contributions
S = S + C'*s*[h.m] + [h.m]'*s*C + [h.m]'*s*[h.m];
C = C + [h.m];
