function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] = ...
                                                                  gpD(gp, m, s)
% Compute joint predictions and derivatives for multiple GPs with uncertain
% inputs. This version uses a linear + bias mean function. Predictive variances
% contain uncertainty about the function, but no noise.
%
% gp                 gaussian process struct
%   hyp     1 x E    struct array of GP hyper-parameters
%     l     D x 1    log lengthscales
%     s     1 x 1    log signal standard deviation
%     n     1 x 1    log noise standard deviation
%     m     D x 1    linear weights for the GP mean
%     b     1 x 1    biases for the GP mean
%   inputs  n x D    matrix of training inputs
%   target  n x E    matrix of training targets
%   iK    n x n x E  inverse covariance matrix
%   beta    n x E    iK*(targets - mean function of inputs)
% m         D x 1    mean of the test distribution
% s         D x D    covariance matrix of the test distribution
% M         E x 1    mean of predictive distribution 
% S         E x E    covariance of predictive distribution             
% C         D x E    inv(s) times input-output covariance matrix
% dMdm      E x D    deriv of output mean w.r.t. input mean
% dSdm    E*E x D    deriv of output covariance w.r.t input mean
% dCdm    D*E x D
% dMds      E x D*D  deriv of ouput mean w.r.t input covariance
% dSds    E*E x D*D  deriv of output cov w.r.t input covariance
% dCds    D*E x D*D
% dMdp      E x Np   deriv of output mean wrt parameters
% dSdp    E*E x Np   deriv of output covariance wrt parameters
% dCdp    D*E x Np   deriv of inv(s) times input-output covariance wrt par
%
% Copyright (C) 2014, Carl Edward Rasmussen & Andrew McHutchon, 2014-12-01

x = gp.inputs;

[n, D, pE] = size(x); DD = D*D; E = size(gp.beta,2); EE = E*E; DE = D*E;
h = gp.hyp; iK = gp.W; beta = gp.beta; idx = gp.idx; Np = length(unwrap(idx));

M = zeros(E,1); S = zeros(E); C = zeros(D,E); k = zeros(n,E); dMdm = zeros(E,D);
dSdm = zeros(E,E,D); dCdm = zeros(D,E,D); dMds = zeros(E,D,D); 
dSds = zeros(E,E,D,D); dCds = zeros(D,E,D,D); dkdl = zeros(n,D,E);
dMdp = zeros(E,Np); dSdp = zeros(E,E,Np); dCdp = zeros(D,E,Np); T = zeros(D);

inp = bsxfun(@minus,x,m');                         % x - m, either nxD or nxDxE

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  l = diag(exp(-h(i).l));                                         % DxD, L^-1/2
  z = inp(:,:,min(i,pE))*l;                               % nxD, (X - m)*L^-1/2
  A = l*s*l;
  B = A+eye(D);                                             % L^-1/2*S*L^-1/2+I
  H = l/B*l;                                                         % inv(S+L)
  G = z/B;                                                                % nxD
  q = exp(-sum(z.*G,2)/2);                                                % nx1
  qb = q.*beta(:,i);                                                      % nx1
  Gl = G*l;                                               % nxD, (X-m)*(S+L)^-1
  c = exp(2*h(i).s)/prod(diag(chol(B)));              % sf2/sqrt(det(S*iL + I))
  M(i) = c*sum(qb);
  P = bsxfun(@times,Gl,qb);                                               % nxD
  C(:,i) = c*sum(P,1);                                               % c*Gl'*qb
  dMds(i,:,:) = c*Gl'*P/2 - H*M(i)/2;
  for d = 1:D
    dCds(d,i,:,:) = c*bsxfun(@times,Gl,Gl(:,d))'*P/2-H*C(d,i)/2-C(:,i)*H(d,:);
  end
  z2 = z.^2; k(:,i) = 2*h(i).s - sum(z2,2)/2;
   
  dMdp(i,idx(i).beta) = c*q';             % Derivatives w.r.t. the parameters p
  dCdp(:,i,idx(i).beta) = c*bsxfun(@times,Gl,q)';
  dMdp(i,idx(i).s) = 2*M(i);
  dCdp(:,i,idx(i).s) = 2*C(:,i);
   
  lbdl = bsxfun(@times,G.*G,qb);                                          % nxD
  cdl = sum(A.*inv(B));
  dMdp(i,idx(i).l) = sum(lbdl,1)*c + c*sum(qb)*cdl;                       % ExD
  dkdl(:,:,i) = z2;                                                     % nxDxE
  tLdl = -2*bsxfun(@times,permute(G,[1,3,2]),permute(B\l,[3,2,1]));     % nxDxD
  dCdp(:,i,idx(i).l) = reshape(qb'*tLdl(:,:),D,D)*c + Gl'*lbdl*c + C(:,i)*cdl;
end
dMdm = C'; dCdm = 2*permute(dMds,[2 1 3]);

iL = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(iL,[3,1,2])); % N x D x E
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); 
    t = 1/sqrt(det(R)); ij = inpil(:,:,j); iRs = R\s;
    L = exp(bsxfun(@plus,k(:,i),k(:,j)') + maha(ii,-ij,iRs/2));
    bLb = beta(:,i)'*L*beta(:,j);
    
    if i==j
      iKL = iK(:,:,i).*L; s1iKL = sum(iKL,1); ssiKL = sum(s1iKL);
      S(i,j) = t*(bLb - ssiKL);
      
      zi = ii/R; s2iKL = sum(iKL,2);
      bibLi = L'*beta(:,i).*beta(:,i); cbLi = L'*bsxfun(@times, beta(:,i), zi);
      r = (bibLi'*zi*2 - (s2iKL' + s1iKL)*zi)*t;
      for d = 1:D
        T(d,1:d) = 2*(zi(:,1:d)'*(zi(:,d).*bibLi) + ...
            cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) ...
                                                   - zi(:,1:d)'*(iKL*zi(:,d)));
        T(1:d,d) = T(d,1:d)';
      end
    else
      S(i,j) = bLb*t;      
      zi = ii/R; zj = ij/R;
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
    
    dSdm(i,j,:) = r - M(i)*dMdm(j,:)-M(j)*dMdm(i,:);  % dSdm & dSds derivatives
    dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S(i,j)*diag(iL(:,i)+iL(:,j))/R)/2;
    T = T - reshape(M(i)*dMds(j,:,:) + M(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T;
    dSds(j,i,:,:) = permute(dSds(i,j,:,:),[1,2,4,3]);
    
    dSdp(i,j,idx(i).beta) = L*beta(:,j)*t;                   % dSdp derivatives
    dSdp(i,j,idx(j).beta) = permute(dSdp(i,j,idx(j).beta),[4,3,1,2]) + ...
                                                                beta(:,i)'*L*t;
    dSdp(i,j,idx(i).s) = 2*S(i,j);
    dSdp(i,j,idx(j).s) = dSdp(i,j,idx(j).s) + 2*S(i,j);
    tdli = diag(iRs).*iL(:,i)*t; tdlj = diag(iRs).*iL(:,j)*t; 
    iiij = bsxfun(@plus,permute(ii,[1,3,2]),permute(ij,[3,1,2]));
    ijiRs = reshape(reshape(iiij,n^2,D)*iRs,n,n,D);                     % nxnxD
    mahadli = -2*bsxfun(@times,permute(ii,[1,3,2]),ijiRs) ...
                    + bsxfun(@times,ijiRs.^2,permute(iL(:,i),[2,3,1])); % nxnxD
    mahadlj = -2*bsxfun(@times,permute(ij,[3,1,2]),ijiRs) ...
                            + bsxfun(@times,ijiRs.^2,permute(iL(:,j),[2,3,1]));
    dLdli = bsxfun(@times,bsxfun(@plus,permute(dkdl(:,:,i),[1,3,2]),mahadli),L); % n-by-n-by-D
    dLdlj = bsxfun(@times,bsxfun(@plus,permute(dkdl(:,:,j),[3,1,2]),mahadlj),L); % n-by-n-by-D
    dS1dli = reshape(beta(:,i)'*dLdli(:,:),n,D)'*beta(:,j)*t + bLb*tdli;
    dS1dlj = reshape(beta(:,i)'*dLdlj(:,:),n,D)'*beta(:,j)*t + bLb*tdlj;
    dSdp(i,j,idx(i).l) = dS1dli; 
    dSdp(i,j,idx(j).l) = squeeze(dSdp(i,j,idx(j).l)) + dS1dlj;
    if i==j; 
      dSdiK = -t*L; iKdn = -2*exp(2*h(i).n)*iK(:,:,i)*iK(:,:,i);
      dSdp(i,i,idx(i).n) = dSdiK(:)'*iKdn(:);
      iKdlsf = -2*iK(:,:,i)*gp.Kclean(:,:,i)*iK(:,:,i);
      dSdp(i,i,idx(i).s) = dSdp(i,i,idx(i).s) + dSdiK(:)'*iKdlsf(:) + 2*exp(2*h(i).s);
      dSdp(i,i,idx(i).l) = squeeze(dSdp(i,i,idx(i).l))' - ...
                t*reshape(iK(:,:,i),1,n^2)*reshape(dLdli+dLdlj,n^2,D) ...
                    - L(:)'*gp.iKdl(:,:,i)*t - ssiKL*(tdli + tdlj)';
    end
  end
  S(i,i) = S(i,i) + exp(2*h(i).s);            % add signal variance to diagonal
end
S = S - M*M';                                           % centralize 2nd moment

i = repmat(triu(true(E,E)),[1,1,Np]);          % fill in symmetric half of dSdp
dSdpt = permute(dSdp,[2,1,3]); dSdp(i) = dSdpt(i);

dMds = reshape(dMds,E,DD);                              % vectorise derivatives
dCdm = reshape(dCdm,DE,D); dCds = reshape(dCds,DE,DD);
dSdm = reshape(dSdm,EE,D); dSds = reshape(dSds,EE,DD);

ms = [h.m]'*s; Cs = C'*s; t = zeros(E,E,D,E); 
for i=1:D; for j=1:E; t(:,j,i,j) = ms(:,i) + Cs(:,i); end; end
dSdp(:,:,[idx.m]) = reshape(t+permute(t,[2,1,3,4]),E,E,DE);
dSdp = reshape(dSdp,EE,Np);
dSdp = dSdp - prodd(M,dMdp) + prodd(ms,reshape(dCdp,DE,Np)) - ...
       prodd([],dMdp,M') + prodd([], reshape(permute(dCdp,[2,1,3]),DE,Np),ms');

dMdm = dMdm + [h.m]';                 % add linear contributions to derivatives
for i=1:E, dMdp(i,idx(i).m) = m; dMdp(i,idx(i).b) = 1; end
dCtdm = transposed(dCdm,D);
dSdm = dSdm + prodd([],dCtdm,s*[h.m]) + prodd([h.m]'*s,dCdm);
dCtds = transposed(dCds,D);
dSds = dSds + prodd([],dCtds,s*[h.m]) + prodd([h.m]'*s,dCds) + ...
     prodd(C','eye',[h.m]) + prodd([h.m]','eye',C) + prodd([h.m]','eye',[h.m]);
for i=1:E, for j=1:D, dCdp(j,i,idx(i).m(j)) = 1; end; end
dCdp = reshape(dCdp, DE, Np);

M = M + [h.m]'*m + [h.b]';                % add linear contributions to outputs 
S = S + C'*s*[h.m] + [h.m]'*s*C + [h.m]'*s*[h.m];
C = C + [h.m];



