function [M, S, C, R, dMdm, dSdm, dCdm, dRdm, dMds, dSds, dCds, dRds, ...
  dMdv, dSdv, dCdv, dRdv] = gphd(gpmodel, m, s, v, dod, Sonly)
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
%   iK      nxnxE   inverse noisy covariance matrices
%   beta    nxE   iK*(targets - mean function of inputs)
% m         Dx1   mean     of         test input if v non-existent, else
%           Dx1   mean     of mean of test input
% s         DxD   variance of         test input if v non-existent, else
%           DxD   variance of mean of test input
% v         DxD   variance of         test input
% dod       str   derivative output dimension (string '2D' or '4D' (default))
% Sonly     bool  need to output S only (inc. derivatives)?
% M         Ex1   mean     or mean of mean of prediction
% S         ExE   variance or variance of mean of prediction
% C         DxE   inv(s) times input output covariance (of mean)
% R         ExE   mean of variance of prediction
% dMdm      Ex1 x D    deriv of output mean w.r.t. input mean-of-mean
% dSdm      ExE x D    deriv of output var-of-mean w.r.t input mean-of-mean
% dCdm      DxE x D    deriv of inv(s) input-output cov w.r.t. input mean-of-mean
% dRdm      ExE x D    deriv of output mean-of-var w.r.t. input mean-of-mean
% dMds      Ex1 x DxD  deriv of output mean w.r.t. input var-of-mean
% dSds      ExE x DxD  deriv of output var-of-mean w.r.t input var-of-mean
% dCds      DxE x DxD  deriv of inv(s) input-output cov w.r.t. input var-of-mean
% dRds      ExE x DxD  deriv of output mean-of-var w.r.t. input var-of-mean
% dMdv      Ex1 x DxD  deriv of output mean w.r.t. input var
% dSdv      ExE x DxD  deriv of output var-of-mean w.r.t input var
% dCdv      DxE x DxD  deriv of inv(s) input-output cov w.r.t. input var
% dRdv      ExE x DxD  deriv of output mean-of-var w.r.t. input var
%
% See also <a href="gph.pdf">gph.pdf</a>, GPH.M, GPHT.M.
% Copyright (C) 2014 by Carl Edward Rasmussen, & Rowan McAllister 2014-11-25

if isfield(gpmodel,'induce') && numel(gpmodel.induce) > 0
  x = bsxfun(@minus, gpmodel.induce, m');            % x is either nxD or nxDxE
else
  x = bsxfun(@minus, gpmodel.inputs, m');                            % x is nxD
end
[~, D, pE] = size(x); E = size(gpmodel.beta,2); C = zeros(D,E);
dCdm =nan(D,E,D); dCds =nan(D,E,D,D);
if nargin < 3, s = zeros(D); end
if nargin < 4, v = zeros(D); end
if nargout <=4, [M, S, C, R] = gph(gpmodel, m, s, v); return; end
h = gpmodel.hyp; iK = gpmodel.iK; beta = gpmodel.beta;
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end; hm = [h.m];
if ~isfield(h,'b'); [h.b] = deal(0); end
ss2 = exp(2*bsxfun(@plus,[h.s]',[h.s]));

% M                                             % first, compute predicted mean
[z,c,dzdm,~,dzdsv,dcdsv] = q(x, [h.l], s+v);
qb = exp(z).*beta;
M = bsxfun(@times,exp(c+2*[h.s]),qb)';         % mean without mean contribution
dMdm = sum(bsxfun(@times,M,permute(dzdm,[2,1,3])),2);
ldqds = bsxfun(@plus,dcdsv,dzdsv);
dMds = permute(sum(bsxfun(@times,M',ldqds),1),[2,1,3,4]);
dMdv = dMds;
M = sum(M,2);

% C                                 % then inv(s) times input-output covariance
for i=1:E
  il=diag(exp(-2*h(i).l)); il=il/(eye(D)+(s+v)*il); ilx=il*x(:,:,min(i,pE))';
  C(:,i) = ilx*qb(:,i);
  for d=1:D
    dCdm(:,i,d) = ilx*(qb(:,i).*dzdm(:,i,d)) - il(d,:)'*sum(qb(:,i));
    dCds(d,i,:,:) = -il(:,d)*C(:,i)';
    dCds(d,i,:,:) = dCds(d,i,:,:) + ...
      sum(bsxfun(@times,ilx(d,:)',bsxfun(@times,qb(:,i),ldqds(:,i,:,:))),1);
  end
end
C = bsxfun(@times, C, exp(c+2*[h.s]));   % covariance without mean contribution
dCdm = bsxfun(@times, dCdm, exp(c+2*[h.s]));
dCds = (dCds + permute(dCds,[1,2,4,3]))/2;
dCds = bsxfun(@times, dCds, exp(c+2*[h.s]));
dCdv = dCds;

% S                                                        variance of the mean
[bQb,~,dbQbdm,~,dbQbds,~,dbQbdv] = Q(x, [h.l], v, s, iK, beta); shm = (s*hm)';
S = ss2.*bQb - M*M' + C'*s*[h.m] + [h.m]'*s*C + [h.m]'*s*[h.m];
MdMdm = bsxfun(@times,M',dMdm); cm = reshape(shm*dCdm(:,:),[E,E,D]);
MdMds = bsxfun(@times,M',dMds); cs = reshape(shm*dCds(:,:),[E,E,D,D]);
MdMdv = bsxfun(@times,M',dMdv); cv = reshape(shm*dCdv(:,:),[E,E,D,D]);
dSdm = bsxfun(@times,ss2,dbQbdm) -MdMdm -permute(MdMdm,[2,1,3]) +cm +permute(cm,[2,1,3]);
dSds = bsxfun(@times,ss2,dbQbds) -MdMds -permute(MdMds,[2,1,3,4]) +cs +permute(cs,[2,1,3,4]);
dSdv = bsxfun(@times,ss2,dbQbdv) -MdMdv -permute(MdMdv,[2,1,3,4]) +cv +permute(cv,[2,1,3,4]);
chm = bsxfun(@times,permute(C,[2,3,1,4]),permute(hm,[3,2,4,1]));
chm = chm + permute(chm,[2,1,4,3]) + ...
  bsxfun(@times,permute(hm,[2,3,1,4]),permute(hm,[3,2,4,1]));
chm = (chm + permute(chm,[1,2,4,3]))/2;
dSds = dSds + chm;

% R                                               finally, mean of the variance
if nargin >= 6 && Sonly;
  R=nan(E); dRdm=nan(E,E,D); dRds=nan(E,E,D,D); dRdv=nan(E,E,D,D);
else
  R = bQb; dRdm = dbQbdm; dRds = dbQbds; dRdv = dbQbdv;
  [bQb, tiKQ, dbQbdm, dtikQdm, dbQbdsv, dtikQdsv] = ...
    Q(x, [h.l],  zeros(D), s+v, iK, beta); vhm = (v*hm)';
  R = ss2.*(bQb - R + diag(exp(-2*[h.s])-tiKQ)) + ...
    C'*v*[h.m] + [h.m]'*v*C + [h.m]'*v*[h.m];
  dRdm = bsxfun(@times,ss2, dbQbdm  -dRdm -bsxfun(@times,eye(E),dtikQdm));
  dRds = bsxfun(@times,ss2, dbQbdsv -dRds -bsxfun(@times,eye(E),dtikQdsv));
  dRdv = bsxfun(@times,ss2, dbQbdsv -dRdv -bsxfun(@times,eye(E),dtikQdsv));
  cm = reshape(vhm*dCdm(:,:),[E,E,D]); dRdm = dRdm+cm+permute(cm,[2,1,3]);
  cs = reshape(vhm*dCds(:,:),[E,E,D,D]); dRds = dRds+cs+permute(cs,[2,1,3,4]);
  cv = reshape(vhm*dCdv(:,:),[E,E,D,D]); dRdv = dRdv+cv+permute(cv,[2,1,3,4])+chm;
end

M = M + [h.m]'*m + [h.b]';                  % add contribution of mean function
dMdm = dMdm + permute([h.m]',[1,3,2]);
C = C + [h.m];                        % add the mean contribution to covariance

if nargin >= 5 && strcmp(dod,'2D')
  % Reshape to a partially unwrapped derivative matrix, whose rows are the
  % unwrapped dependent variables, and columns are the unwrapped independent
  % variables.
  DD = D*D;
  dMdm = reshape(dMdm, [], D );
  dSdm = reshape(dSdm, [], D );
  dCdm = reshape(dCdm, [], D );
  dRdm = reshape(dRdm, [], D );
  dMds = reshape(dMds, [], DD);
  dSds = reshape(dSds, [], DD);
  dCds = reshape(dCds, [], DD);
  dRds = reshape(dRds, [], DD);
  dMdv = reshape(dMdv, [], DD);
  dSdv = reshape(dSdv, [], DD);
  dCdv = reshape(dCdv, [], DD);
  dRdv = reshape(dRdv, [], DD);
end

% z     nxE    exp negative quatratics
% c     1xE    -log(det)/2
% dzdm  nxE x D
% dcdm  1xE x D
% dzdv  nxE x DxD
% dcdv  1xE x DxD
function [z,c,dzdm,dcdm,dzdv,dcdv] = q(x, L, V)
[n, D, pE] = size(x); E = size(L,2); z = zeros(n,E); c = zeros(1,E);
dzdm = nan(n,E,D); dcdm = zeros(1,E,D); dzdv = nan(n,E,D,D); dcdv = nan(1,E,D,D);
d = 1:D; d = d(ones(D,1),:);
for i=1:E
  il = diag(exp(-L(:,i)));                                        % Lambda^-1/2
  in = x(:,:,min(i,pE))*il;                               % (X - m)*Lambda^-1/2
  B = il*V*il+eye(D);                       % Lambda^-1/2 * V * Lambda^-1/2 + I
  z(:,i) = -sum(in.*(in/B),2)/2;
  c(i) = -sum(log(diag(chol(B))));                  % -log(det(Lambda\V + I))/2
  if nargout<3, continue, end
  il = diag(exp(-2*L(:,i))); iL = il/(V*il+eye(D)); xiL = x(:,:,min(i,pE))*iL;
  dzdm(:,i,:) = xiL;
  dzdv(:,i,:,:) = permute(reshape(repmat(xiL',D,1).*xiL(:,d)',[D,D,n])/2,[3,4,1,2]);
  dcdv(1,i,:,:) = -iL/2;
end


% bQb      ExE    quadratics of beta with Q
% tikQ     1xE    traces of the products of iK and Q
% dbQbdm   ExE x D
% dtikQdm  1xE x D
% dbQbds   ExE x DxD
% dtikQds  1xE x DxD
% dbQbdv   ExE x DxD
% dtikQdv  1xE x DxD
function [bQb,tiKQ,dbQbdm,dtikQdm,dbQbds,dtikQds,dbQbdv,dtikQdv] = Q(...
  x, L, V, s, iK, beta)
[n, D, pE] = size(x); E = size(L,2);
bQb = zeros(E); tiKQ = zeros(1,E);
iL = zeros(D,D,E); xiL = zeros(n,D,E); xiL2 = nan(n,D,D,E);
dbQbdm = nan(E,E,D); dtikQdm = nan(1,E,D); dbQbdv = nan(E,E,D,D);
dtikQdv = nan(1,E,D,D); dbQbds = nan(E,E,D,D); dtikQds = nan(1,E,D,D);
[z, c, dzdm, ~, dzdv, dcdv] = q(x, L, V);
for i=1:E
  il = diag(exp(-2*L(:,i)));
  iL(:,:,i) = il/(V*il + eye(D));
  xiL(:,:,i) = x(:,:,min(pE,i))*iL(:,:,i);
  if nargout > 6
    xiL2(:,:,:,i) = bsxfun(@times,xiL(:,:,i),permute(xiL(:,:,i),[1,3,2]));
  end
end
for i=1:E
  for j=1:i
    iLij = iL(:,:,i)+iL(:,:,j);
    R = s*iLij+eye(D); t = exp(c(i)+c(j))/sqrt(det(R)); iR = inv(R); Y = iR*s;
    Q = exp(bsxfun(@plus,z(:,i),z(:,j)')+maha(xiL(:,:,i),-xiL(:,:,j),Y/2));
    bQb(i,j) = beta(:,i)'*Q*beta(:,j)*t; bQb(j,i) = bQb(i,j);
    if nargout<3, continue, end
    Ydydm = -Y*iLij;
    dyYydm = bsxfun(@plus,permute(xiL(:,:,i)*Ydydm,[1,3,2]), ...
      permute(xiL(:,:,j)*Ydydm,[3,1,2]));
    dQdm = bsxfun(@times,Q,bsxfun(@plus,dzdm(:,i,:),permute(dzdm(:,j,:),[2,1,3]))+dyYydm);
    dbQbdm(i,j,:) = beta(:,j)'*reshape(beta(:,i)'*dQdm(:,:),[n,D])*t;
    dbQbdm(j,i,:) = dbQbdm(i,j,:);
    if nargout<5, continue, end
    y = bsxfun(@plus,permute(xiL(:,:,i),[2,1,3]),permute(xiL(:,:,j),[2,3,1]));
    iRy = reshape(iR'*y(:,:),[D,n,n]);
    dyYyds = bsxfun(@times,permute(iRy,[2,3,1,4]),permute(iRy,[2,3,4,1]))/2;
    dlc2ds = -iLij*iR/2;
    dQds = bsxfun(@times,Q,bsxfun(@plus,permute(dlc2ds,[3,4,1,2]),dyYyds));
    dbQbds(i,j,:,:) = reshape(beta(:,j)'*reshape(beta(:,i)'*dQds(:,:),[n,D*D]),[D,D])*t;
    dbQbds(j,i,:,:) = dbQbds(i,j,:,:);
    if nargout<7, continue, end
    Yy = Y*y(:,:);
    iLYy_i = bsxfun(@minus,reshape(iL(:,:,i)*Yy,[D,n,n]),xiL(:,:,i)');
    iLYy_j = bsxfun(@minus,reshape(iL(:,:,j)*Yy,[D,n,n]),permute(xiL(:,:,j)',[1,3,2]));
    dyYydv = (bsxfun(@times,permute(iLYy_i,[2,3,1,4]),permute(iLYy_i,[2,3,4,1])) + ...
      bsxfun(@times,permute(iLYy_j,[2,3,1,4]),permute(iLYy_j,[2,3,4,1])) - ...
      bsxfun(@plus,permute(xiL2(:,:,:,i),[1,4,2,3]),permute(xiL2(:,:,:,j),[4,1,2,3])))/2;
    dlc2dv = (iL(:,:,i)*Y*iL(:,:,i) + iL(:,:,j)*Y*iL(:,:,j))/2;
    dQdv = bsxfun(@times,Q, ...
      bsxfun(@plus, permute(dlc2dv+squeeze(dcdv(1,i,:,:)+dcdv(1,j,:,:)),[3,4,1,2]), ...
      bsxfun(@plus,dzdv(:,i,:,:),permute(dzdv(:,j,:,:),[2,1,3,4]))) + dyYydv);
    dbQbdv(i,j,:,:) = reshape(beta(:,j)'*reshape(beta(:,i)'*dQdv(:,:),[n,D*D]),[D,D])*t;
    dbQbdv(j,i,:,:) = dbQbdv(i,j,:,:);
  end
  if nargout>1, tiKQ(i) = sum(sum(iK(:,:,i).*Q))*t; end
  if nargout>3, dtikQdm(1,i,:)=sum(sum(bsxfun(@times,iK(:,:,i),dQdm),1),2)*t; end
  if nargout>5, dtikQds(1,i,:,:)=sum(sum(bsxfun(@times,iK(:,:,i),dQds),1),2)*t; end
  if nargout>7, dtikQdv(1,i,:,:)=sum(sum(bsxfun(@times,iK(:,:,i),dQdv),1),2)*t; end
end