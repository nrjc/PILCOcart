function [M, S, C, V, dMdm, dSdm, dCdm, dVdm, dMds, dSds, dCds, dVds, ...
  dMdv, dSdv, dCdv, dVdv] = gphd_parallel(gpmodel, m, s, v, combineSV)
% Compute joint predictions and derivatives for multiple GPs with hierarchical
% uncertain inputs.
%
% gpmodel         dynamics model struct
%   hyp     1xE   struct array of GP hyper-parameters
%     l     Dx1   log lengthscales
%     s             log signal standard deviation
%     n             log noise standard deviation
%     m     Dx1   linear weights for GP mean
%     b             bias for GP mean
%   inputs  nxD   or
%           nxDxE   training inputs, possibly separate per target
%   target  nxE   training targets
%   W       nxnxE inverse noisy covariance matrices
%   beta    nxE   W*(targets - mean function of inputs)
% m         Dx1   mean     of         test input if v non-existent, else
%           Dx1   mean     of mean of test input
% s         DxD   variance of         test input if v non-existent, else
%           DxD   variance of mean of test input
% v         DxD   variance of         test input
% combineSV bool  output S (not V) where S is instead S+V
% M         Ex1   mean     or mean of mean of prediction
% S         ExE   variance or variance of mean of prediction
% C         DxE   inv(s) times mean input - mean output covariance, equivalently
%                 inv(v) times expected[input-output covariance]
% V         ExE   mean of variance of prediction
% dMdm      Ex1 x D    deriv of output mean w.r.t. input mean-of-mean
% dSdm      ExE x D    deriv of output var-of-mean w.r.t input mean-of-mean
% dCdm      DxE x D    deriv of inv(s) input-output cov w.r.t. input mean-of-mean
% dVdm      ExE x D    deriv of output mean-of-var w.r.t. input mean-of-mean
% dMds      Ex1 x DxD  deriv of output mean w.r.t. input var-of-mean
% dSds      ExE x DxD  deriv of output var-of-mean w.r.t input var-of-mean
% dCds      DxE x DxD  deriv of inv(s) input-output cov w.r.t. input var-of-mean
% dVds      ExE x DxD  deriv of output mean-of-var w.r.t. input var-of-mean
% dMdv      Ex1 x DxD  deriv of output mean w.r.t. input var
% dSdv      ExE x DxD  deriv of output var-of-mean w.r.t input var
% dCdv      DxE x DxD  deriv of inv(s) input-output cov w.r.t. input var
% dVdv      ExE x DxD  deriv of output mean-of-var w.r.t. input var
%
% See also <a href="gph.pdf">gph.pdf</a>, GPH.M, GPHT.M.
% Copyright (C) 2014 by Carl Edward Rasmussen, & Rowan McAllister 2015-09-24

if isprop(gpmodel,'induce') && numel(gpmodel.induce) > 0
  x = bsxfun(@minus, gpmodel.induce, m');            % x is either nxD or nxDxE
else
  x = bsxfun(@minus, gpmodel.inputs, m');                            % x is nxD
end
[~, D, pE] = size(x); E = size(gpmodel.beta,2); C = zeros(D,E); DD=D*D; EE=E*E;
dCdm =nan(D*E,D); dCds =nan(D*E,DD);
if nargin < 3; s = zeros(D); end
if nargin < 4; v = zeros(D); end
if nargin < 5; combineSV = false; end
if nargout <=4; [M, S, C, V] = gph(gpmodel, m, s, v, combineSV); return; end
h = gpmodel.hyp; iK = gpmodel.W; beta = gpmodel.beta;
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end; hm = [h.m];
if ~isfield(h,'b'); [h.b] = deal(0); end
ss2 = exp(2*bsxfun(@plus,[h.s]',[h.s]));

% M                                             % first, compute predicted mean
[z,c,dzdm,~,dzdsv,dcdsv] = q(x, [h.l], s+v);
qb = exp(z).*beta;
M = bsxfun(@times,exp(c+2*[h.s]),qb);         % mean without mean contribution
dMdm = permute(sum(bsxfun(@times,M,dzdm),1),[2,3,1]);
ldqds = bsxfun(@plus,permute(dcdsv,[3,1,2]),dzdsv);
dMds = permute(sum(bsxfun(@times,M,ldqds),1),[2,3,1]);
dMdv = dMds;
M = sum(M,1)';

% C                                 % then inv(s) times input-output covariance
for i=1:E
  il=diag(exp(-2*h(i).l)); il=il/(eye(D)+(s+v)*il); ilx=il*x(:,:,min(i,pE))';
  C(:,i) = ilx*qb(:,i);
  for d=1:D
    Di = sub2ind2(D,1:D,i);
    di = sub2ind2(D,d,i);
    dCdm(Di,d) = ilx*(qb(:,i).*dzdm(:,i,d)) - il(d,:)'*sum(qb(:,i));
    dCds(di,:) = reshape(-il(:,d)*C(:,i)',DD,1);
    dCds(di,:) = dCds(di,:) + ...
      permute(sum(bsxfun(@times,ilx(d,:)',bsxfun(@times,qb(:,i),ldqds(:,i,:))),1),[1,3,2]);
  end
end
chs = repmat(exp(c+2*[h.s]),D,1);
C = C.*chs;                              % covariance without mean contribution
chs = chs(:);
dCdm = bsxfun(@times, dCdm, chs);
dCds = (dCds + transposed(dCds,D,2))/2;
dCds = bsxfun(@times, dCds, chs);
dCdv = dCds;

% S                                                        variance of the mean
[bQb,~,dbQbdm,~,dbQbds,~,dbQbdv] = Q(x, [h.l], v, s, iK, beta); shm = (s*hm)';
S = ss2.*bQb - M*M' + C'*s*[h.m] + [h.m]'*s*C + [h.m]'*s*[h.m]; uss2 = ss2(:);
MdMdm = outern(@times,M,dMdm); cm = prodd(shm,dCdm);
MdMds = outern(@times,M,dMds); cs = prodd(shm,dCds);
MdMdv = outern(@times,M,dMdv); cv = prodd(shm,dCdv);
dSdm = bsxfun(@times,uss2,dbQbdm) -MdMdm +cm +transposed(-MdMdm+cm,E);
dSds = bsxfun(@times,uss2,dbQbds) -MdMds +cs +transposed(-MdMds+cs,E);
dSdv = bsxfun(@times,uss2,dbQbdv) -MdMdv +cv +transposed(-MdMdv+cv,E);
chm = outernd(@times,C',hm');
chm = chm + transposed(chm,E) + outernd(@times,hm',hm');
chm = (chm + transposed(chm,D,2))/2;
dSds = dSds + chm;

% V                                               finally, mean of the variance
V = bQb; dVdm = dbQbdm; dVds = dbQbds; dVdv = dbQbdv;
[bQb, tiKQ, dbQbdm, dtikQdm, dbQbdsv, dtikQdsv] = ...
  Q(x, [h.l], zeros(D), s+v, iK, beta); vhm = (v*hm)';
V = ss2.*(bQb - V + diag(exp(-2*[h.s])-tiKQ')) + ...
  C'*v*[h.m] + [h.m]'*v*C + [h.m]'*v*[h.m];
cm = prodd(vhm,dCdm);
cs = prodd(vhm,dCds);
cv = prodd(vhm,dCdv);
dVdm = bsxfun(@times,uss2, dbQbdm  -dVdm -diagd(dtikQdm )) + cm + transposed(cm,E);
dVds = bsxfun(@times,uss2, dbQbdsv -dVds -diagd(dtikQdsv)) + cs + transposed(cs,E);
dVdv = bsxfun(@times,uss2, dbQbdsv -dVdv -diagd(dtikQdsv)) + cv + transposed(cv,E) + chm;

M = M + [h.m]'*m + [h.b]';                  % add contribution of mean function
dMdm = dMdm + [h.m]';
C = C + [h.m];                        % add the mean contribution to covariance
if combineSV
  S = S + V;
  dSdm = dSdm + dVdm; dSds = dSds + dVds; dSdv = dSdv + dVdv;
  V=nan(E); dVdm=nan(EE,D); dVds=nan(EE,DD); dVdv=nan(EE,DD);
end


function [z,c,dzdm,dcdm,dzdv,dcdv] = q(x, L, V)
% q function used in GPH.PDF document.
%
% z     nxE    exp negative quatratics
% c       E    -log(det)/2
% dzdm  nxE x D
% dcdm    E x D
% dzdv  nxE x D*D
% dcdv    E x D*D
[n, D, pE] = size(x); E = size(L,2); z = zeros(n,E); c = zeros(1,E);
dzdm = nan(n,E,D); dcdm = zeros(E,D); dzdv = nan(n,E,D*D); dcdv = nan(E,D*D);
for i=1:E
  il = diag(exp(-L(:,i)));                                        % Lambda^-1/2
  in = x(:,:,min(i,pE))*il;                               % (X - m)*Lambda^-1/2
  B = il*V*il+eye(D);                       % Lambda^-1/2 * V * Lambda^-1/2 + I
  z(:,i) = -sum(in.*(in/B),2)/2;
  c(i) = -sum(log(diag(chol(B))));                  % -log(det(Lambda\V + I))/2
  if nargout<3, continue, end
  il = diag(exp(-2*L(:,i))); iL = il/(V*il+eye(D)); xiL = x(:,:,min(i,pE))*iL;
  dzdm(:,i,:) = xiL;
  dzdv(:,i,:) = outerd(@times,xiL,xiL)/2;
  dcdv(i,:) = -iL(:)/2;
end

function [bQb,tiKQ,dbQbdm,dtikQdm,dbQbds,dtikQds,dbQbdv,dtikQdv] = Q(...
  x, L, V, s, iK, beta)
% GPHQD implements the Q function used in GPH.PDF document.
% Note to future editors: this function is the greatest bottleneck when using a
% CtrlBF controller, and thus is written for speed, not for readability.
%
% bQb      E*E    quadratics of beta with Q
% tikQ       E    traces of the products of iK and Q
% dbQbdm   E*E x D
% dtikQdm    E x D
% dbQbds   E*E x D*D
% dtikQds    E x D*D
% dbQbdv   E*E x D*D
% dtikQdv    E x D*D
%
% See also <a href="gph.pdf">gph.pdf</a>, Q.M, GPH.M.
[n, D, pE] = size(x); E = size(L,2); DD = D*D; nn=n*n; EE = E*E;
bQb = zeros(E); tiKQ = zeros(E,1);
iL = zeros(D,D,E); iLs = zeros(D,D,E); xiL = zeros(n,D,E); xiLs = zeros(n,D,E);
xiL2s = nan(n,D*(D+1)/2,E);
dbQbdm = nan(EE,D); dtikQdm = nan(E,D); dbQbdv = nan(EE,DD);
dtikQdv = nan(E,DD); dbQbds = nan(EE,DD); dtikQds = nan(E,DD);
[z, c, dzdm, ~, dzdv, dcdv] = q(x, L, V);
dzdm = permute(dzdm,[1,3,2]); dzdv = permute(dzdv,[1,3,2]);
%persistent uiK dQdm dQds J I dQdv              % terms of unchanging size > n*n
Ln = tril(reshape(1:nn,n,n));              Ln = Ln(Ln>0);
LD = tril(reshape(1:DD,D,D));              LD = LD(LD>0);
ln = tril(reshape(1:nn,n,n),-1);           ln = ln(ln>0);
un = triu(reshape(1:nn,n,n), 1); un = un'; un = un(un>0);
lD = tril(reshape(1:DD,D,D),-1);           lD = lD(lD>0);
uD = triu(reshape(1:DD,D,D), 1); uD = uD'; uD = uD(uD>0);
dcdv = dcdv(:,LD); dzdv = dzdv(:,LD,:);
[J,I] = meshgrid(1:n,1:n);
for i=1:E
  il = diag(exp(-2*L(:,i)));
  iL(:,:,i) = il/(V*il + eye(D));
  iLs(:,:,i) = (il/(V*il + eye(D)))/sqrt(2);
  xiL(:,:,i) = x(:,:,min(pE,i))*iL(:,:,i);
  xiLs(:,:,i) = x(:,:,min(pE,i))*iLs(:,:,i);
  if nargout > 6
    xiL2s(:,:,i) = outerds(@times,xiLs(:,:,i),xiLs(:,:,i),LD);
  end
end

% Parfor prep.
all_ij = nan(1,E*(E+1)/2); K = 0;
for i=1:E
  for j=1:i
    K = K+1;
    all_ij(K) = sub2ind2(E,i,j);
  end
end
bQb_k = nan(K,1);
dbQbdm_k = nan(K,D);
dbQbds_k = nan(K,DD);
dbQbdv_k = nan(K,DD);
tiKQ_k = nan(K,1);
dtikQdm_k = nan(K,D);
dtikQds_k = nan(K,DD);
dtikQdv_k = nan(K,DD);
uiK = reshape(iK,nn,E)';
nout = nargout; % nargout cannot be used inside parfor

parfor k = 1:K;
  ij = all_ij(k);
  [i,j] = ind2sub(E,ij);
  nsym = i==j;
  iLij = iL(:,:,i)+iL(:,:,j);
  R = s*iLij+eye(D); t = exp(c(i)+c(j))/sqrt(det(R)); Y = R\s;
  Q = exp(bsxfun(@plus,z(:,i),z(:,j)')+maha(xiL(:,:,i),-xiL(:,:,j),Y/2));
  bQb_k(k) = beta(:,i)'*Q*beta(:,j)*t;
  if nsym; tiKQ_k(k) = (uiK(i,:)*Q(:))*t; end
  
  if nout<3, continue, end
  Q = Q(:);
  Ydydm = -Y*iLij;
  dQdm = bsxfun(@times,Q,outern(@plus,dzdm(:,:,i),dzdm(:,:,j))+...
    outern(@plus,xiL(:,:,i)*Ydydm, xiL(:,:,j)*Ydydm)); % dyYydm term
  dbQbdm_k(k,:) = prodd(beta(:,i)',dQdm,beta(:,j))*t;
  if nsym; dtikQdm_k(k,:)  = (uiK(i,:)*dQdm)*t; end
  
  if nout<5, continue, end
  dlc2ds = -iLij/R/2; dlc2ds = dlc2ds(LD)';
  if ~nsym
    y = outern(@plus,xiL(:,:,i),xiL(:,:,j));       % nn x D
    yiR2 = y/(R*sqrt(2));
    dQds = bsxfun(@times,Q,bsxfun(@plus,dlc2ds,... % nn x D*(D+1)/2;
      outerds(@times,yiR2,yiR2,LD))); % dyYyds term
  else
    y = outerns(@plus,xiL(:,:,i),xiL(:,:,j),Ln); % nn/2 x D
    yiR2 = y/(R*sqrt(2));
    dQds = nan(nn,D*(D+1)/2); % alloc
    dQds(Ln,:) = bsxfun(@times,Q(Ln),bsxfun(@plus,dlc2ds,...
      outerds(@times,yiR2,yiR2,LD))); % dyYyds term
    dQds(un,:) = dQds(ln,:);
  end
  dbQbds_k(k,LD) = prodd(beta(:,i)',dQds,beta(:,j))*t;
  if nsym; dtikQds_k(k,LD) = (uiK(i,:)*dQds)*t; end
  
  if nout<7, continue, end
  dlc2dv = iLs(:,:,i)*Y*iLs(:,:,i) + iLs(:,:,j)*Y*iLs(:,:,j);
  dlc2dv = dlc2dv(LD)';
  if ~nsym
    Yy = Y*y'; % D x nn
    iLYyi = (iLs(:,:,i)*(Yy - x(I,:,min(pE,i))'))'; % nn x D
    iLYyj = (iLs(:,:,j)*(Yy - x(J,:,min(pE,j))'))'; % nn x D
    dQdv = bsxfun(@times,Q, ...                     % nn x D*(D+1)/2;
      bsxfun(@plus, dlc2dv+dcdv(i,:)+dcdv(j,:), outern(@plus,dzdv(:,:,i),dzdv(:,:,j))) + ...
      outerds(@times,iLYyi,iLYyi,LD) + outerds(@times,iLYyj,iLYyj,LD) - outern(@plus,xiL2s(:,:,i),xiL2s(:,:,j))); % dyYydv term
  else
    Yy = Y*y'; % D x nn/2
    iLYyi = (iLs(:,:,i)*(Yy - x(I(Ln),:,min(pE,i))'))'; % nn/2 x D
    iLYyj = (iLs(:,:,j)*(Yy - x(J(Ln),:,min(pE,j))'))'; % nn/2 x D
    dQdv = nan(nn,D*(D+1)/2); % alloc
    dQdv(Ln,:) = bsxfun(@times,Q(Ln), ...
      bsxfun(@plus, dlc2dv+dcdv(i,:)+dcdv(j,:), outerns(@plus,dzdv(:,:,i),dzdv(:,:,j),Ln)) + ...
      outerds(@times,iLYyi,iLYyi,LD) + outerds(@times,iLYyj,iLYyj,LD) - outerns(@plus,xiL2s(:,:,i),xiL2s(:,:,j),Ln)); % dyYydv term
    dQdv(un,:) = dQdv(ln,:);
  end
  dbQbdv_k(k,LD) = prodd(beta(:,i)',dQdv,beta(:,j))*t;
  if nsym; dtikQdv_k(k,LD) = (uiK(i,:)*dQdv)*t; end
end

% parfor can only deal with consecutive itegers above, so now we change back
% into our indicies.
for k=1:K
  ij = all_ij(k);
  [i,j] = ind2sub(E,ij);
  ji = sub2ind2(E,j,i);
  
  bQb(i,j) = bQb_k(k);
  bQb(j,i) = bQb(i,j);
  
  dbQbdm(ij,:) = dbQbdm_k(k,:);
  dbQbdm(ji,:) = dbQbdm(ij,:);
  
  dbQbds(ij,LD) = dbQbds_k(k,LD);
  dbQbds(ij,uD) = dbQbds(ij,lD);
  dbQbds(ji,:) = dbQbds(ij,:);
  
  dbQbdv(ij,LD) = dbQbdv_k(k,LD);
  dbQbdv(ij,uD) = dbQbdv(ij,lD);
  dbQbdv(ji,:) = dbQbdv(ij,:);
  
  if i == j
    tiKQ(i)       = tiKQ_k(k);
    dtikQdm(i,:)  = dtikQdm_k(k,:);
    dtikQds(i,LD) = dtikQds_k(k,LD); dtikQds(i,uD) = dtikQds(i,lD);
    dtikQdv(i,LD) = dtikQdv_k(k,LD); dtikQdv(i,uD) = dtikQdv(i,lD);
  end
end

function c = outern(f,a,b)
% `Outer' function of dimension 1 (N)umerator (e.g. outer-add, outer-product)
% f             bsxfun function, e.g. @plus or @times
% a    A x D
% b    B X D
% c  A*B X D
c = reshape(bsxfun(f,permute(a,[1,3,2]),permute(b,[3,1,2])),size(a,1)*size(b,1),[]);

function c = outerd(f,a,b)
% `Outer' function of dimension 2 (D)enominator (e.g. outer-add, outer-product)
% f             bsxfun function, e.g. @plus or @times
% a    D x A
% b    D X B
% c    D X A*B
c = reshape(bsxfun(f,a,permute(b,[1,3,2])),[],size(a,2)*size(b,2));

function c = outernd(f,a,b)
% `Outer' function of dimension 1 and 2 (e.g. outer-add, outer-product)
% f             bsxfun function, e.g. @plus or @times
% a    A x B
% b    A x B
% c  A*A x B*B
[A,B] = size(a);
c = reshape(bsxfun(f,permute(a,[1,3,2,4]),permute(b,[3,1,4,2])),A*A,B*B);

function c = outerns(f,a,b,i)
% `Outer' function of dimension 1 (N)umerator (e.g. outer-add, outer-product)
% f             bsxfun function, e.g. @plus or @times
% a    A x D
% b    B X D
% i    N x 1    indices
% c    N X D
c = outern(f,a,b); c = c(i,:);

function c = outerds(f,a,b,i)
% `Outer' function of dimension 2 (D)enominator (e.g. outer-add, outer-product)
% f             bsxfun function, e.g. @plus or @times
% a    D x A
% b    D X B
% i    N x 1    indices
% c    D X N
c = outerd(f,a,b); c = c(:,i);

function diagx = diagd(x)
% take a derrvative matrix and returns what diag would do to a vector.
% x        N x D
% diagx  N*N x D
N = size(x,1);
diagx = reshape(bsxfun(@times,eye(N),permute(x,[3,1,2])),N*N,[]);