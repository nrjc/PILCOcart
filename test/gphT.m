function gphT(deriv, dyn, m, s, v, delta)

% Test function for gph.m which:
%  1. Draws sample gph input data from a SEard GP,
%  2. computes analytic gph.m outputs {Ma, Sa, Ca, Ra},
%  3. computes numeric {Mn, Sn, Cn, Rn} gph outputs using MC samples,
%  4. compares {Ma, Sa, Ca, Ra} against {Mn, Sn, Cn, Rn}.
%
% See also <a href="gph.pdf">gph.pdf</a>, GPH.M, GPHD.M.
% Copyright (C) 2014 by Rowan McAllister 2016-01-20

NSAMPLES = 1e4;
SEED = 18;
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
rng(SEED);
addpath('../gp'); addpath('../util');
dbstop if error
USE_APPROX_SV = true;

% 1. SET GHP INPUTS -----------------------------------------------------------

if nargin < 2
  % Dimensions, seed and scale factors chosen s.t. additive subcomponents of
  % each {M,S,C,R} happen to be comparable, to increase probability that any
  % gph.m errors are easily discerned in gphT.m's output.
  D = 4; E = 3; n = 20; pE = E;
  x = 1+2*randn(n,D,pE); hm = 0.5*randn(1,D,E); hb = 3*randn(1,E);
  sf2 = 0.8+1*rand(1,1,E); ell = 3*rand(1,1,E,D); sn = 0.05+0.3*rand(E,1);
  my = bsxfun(@plus,sum(bsxfun(@times,x,hm),2),hb);   % mean function of inputs
  %           sum_D [ (n x D x pE) @times (1 x D x E) ] @plus (1 x E) = (n x E)
  K = bsxfun(@minus, permute(x,[1,4,3,2]), permute(x,[4,1,3,2]));
  %                 (n x 1 x pE x D) @minus (1 x n x pE x D) = (n x n x pE x D)
  K = sum(bsxfun(@rdivide, K.*K, ell), 4);                       % SEard kernel
  %           sum_D [ (n x n x pE x D) @rdivide (1 x 1 x E x D) ] = (n x n x E)
  K = bsxfun(@times, sf2, exp(-K/2));
  dyn.inputs = x; dyn.target = nan(n,E); dyn.W = nan*K; dyn.beta = nan(n,E);
  for e=1:E;
    K(:,:,e) = K(:,:,e) + sn(e)*eye(n);
    dyn.target(:,e) = chol(K(:,:,e))'*randn(n,1) + my(:,e) + sn(e)*randn(n,1);
    dyn.W(:,:,e) = inv(K(:,:,e));
    dyn.beta(:,e) = dyn.W(:,:,e) * (dyn.target(:,e) - my(:,e));
    hyp.l = log(unwrap(ell(1,1,e,:)));
    hyp.s = log(sqrt(sf2(1,1,e)));
    hyp.n = log(sn(e));
    hyp.m = unwrap(hm(1,:,e));
    hyp.b = hb(e);
    dyn.hyp(e) = hyp;
  end
else
  if isfield(dyn,'induce') && numel(dyn.induce) > 0
    x = bsxfun(@minus, dyn.induce, m');              % x is either nxD or nxDxE
  else
    x = bsxfun(@minus, dyn.inputs, m');                              % x is nxD
  end
  [n, D, pE] = size(x); E = size(dyn.beta,2);
end
if nargin < 3, m = 1+randn(D,1); end
if nargin < 4, r_s = randn(D); s = 0.0004*(r_s*r_s'); end
if nargin < 5, r_v = randn(D); v = 0.0004*(r_v*r_v'); end
if nargin < 6, delta = 1e-5; end

% 2. TEST GHP OUTPUT-DERIVATIVES ----------------------------------------------

if nargin < 1 || isempty(deriv)
  deriv = {'dMdm', 'dSdm', 'dCdm', 'dVdm', 'dMds', 'dSds', 'dCds', 'dVds', ...
    'dMdv', 'dSdv', 'dCdv', 'dVdv'};
elseif strcmp(deriv,'q'), deriv={'dzdm', 'dcdm', 'dzdv', 'dcdv'};
elseif strcmp(deriv,'Q'), deriv={'dbQbdm','dtikQdm','dbQbds','dtikQds','dbQbdv','dtikQdv'};
end
if iscell(deriv), ntests = numel(deriv); else ntests = 1; end
q_args = {x, m, [dyn.hyp.l], v, @q, 2, delta};       % 2 non-derivative outputs
Q_args = {x, m, [dyn.hyp.l], v, s, dyn.W, dyn.beta, @Q, 2, delta};
gph_args = {dyn, m, s, v, false, USE_APPROX_SV, @gphd, nargout(@gph), delta};
test_names = cell(ntests,1); cg = cell(ntests,1); c = 0;
for i=1:ntests
  if iscell(deriv), deriv_i = deriv{i}; else deriv_i = deriv; end
  c = c+1;
  test_names{c} = deriv_i;
  disp(['gphT: Derivative test: ',deriv_i]);
  switch deriv_i
    % [z c dzdm dcdm dzdv dcdv] = q(x_, m, L, V)
    case 'dzdm', cg{c} = cg_wrap(q_args{:}, [1,3], 2);
    case 'dcdm', cg{c} = cg_wrap(q_args{:}, [2,4], 2);
    case 'dzdv', cg{c} = cg_wrap(q_args{:}, [1,5], 4);
    case 'dcdv', cg{c} = cg_wrap(q_args{:}, [2,6], 4);
      % [bQb tiKQ t dbQbdm dtikQdm dbQbds dtikQds dbQbdv dtikQdv dt] = ...
      %   Q(x_, m, L, V, s, iK, beta)
    case 'dbQbdm',  cg{c} = cg_wrap(Q_args{:}, [1,3], 2);
    case 'dtikQdm', cg{c} = cg_wrap(Q_args{:}, [2,4], 2);
    case 'dbQbds',  cg{c} = cg_wrap(Q_args{:}, [1,5], 5);
    case 'dtikQds', cg{c} = cg_wrap(Q_args{:}, [2,6], 5);
    case 'dbQbdv',  cg{c} = cg_wrap(Q_args{:}, [1,7], 4);
    case 'dtikQdv', cg{c} = cg_wrap(Q_args{:}, [2,8], 4);
      % [M, S, C, R, dMdm, dSdm, dCdm, dRdm, dMds, dSds, dCds, dRds, ...
      %   dMdv, dSdv, dCdv, dRdv] = gphd(gpmodel, m, s, v)
    case 'dMdm', cg{c} = cg_wrap(gph_args{:}, [1, 5], 2);
    case 'dSdm', cg{c} = cg_wrap(gph_args{:}, [2, 6], 2);
    case 'dCdm', cg{c} = cg_wrap(gph_args{:}, [3, 7], 2);
    case 'dVdm', cg{c} = cg_wrap(gph_args{:}, [4, 8], 2);
    case 'dMds', cg{c} = cg_wrap(gph_args{:}, [1, 9], 3);
    case 'dSds', cg{c} = cg_wrap(gph_args{:}, [2,10], 3);
    case 'dCds', cg{c} = cg_wrap(gph_args{:}, [3,11], 3);
    case 'dVds', cg{c} = cg_wrap(gph_args{:}, [4,12], 3);
    case 'dMdv', cg{c} = cg_wrap(gph_args{:}, [1,13], 4);
    case 'dSdv', cg{c} = cg_wrap(gph_args{:}, [2,14], 4);
    case 'dCdv', cg{c} = cg_wrap(gph_args{:}, [3,15], 4);
    case 'dVdv', cg{c} = cg_wrap(gph_args{:}, [4,16], 4);
    otherwise, disp('WARNING: Unknown derivative test requested.');
  end
end
print_derivative_test_results(test_names, cg, EPSILON);

% 3. TEST GHP OUTPUTS ---------------------------------------------------------

% Check consistency with gpp when s or v zerod
dyn.induce = [];
maxdiff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
if ~USE_APPROX_SV
  [M1,S1] = gpp(dyn, m, s);
  [M2,S2] = gph(dyn, m, s, 0*v);
  [M3,~,~,S3] = gph(dyn, m, 0*s, s);
  assert(maxdiff({M1,S1},{M2,S2})< 1e-10);
  assert(maxdiff({M1,S1},{M3,S3})< 1e-10);
end

% 3.1. Compute Analytic Outputs            % 3 = 3-input-case, 4 = 4-input-case
[Ma{3,1},Sa{3,1},iCa{3,1}]         = gph(dyn,m,s);    Ca{3,1} = s*iCa{3,1};
[Ma{4,1},Sa{4,1},iCa{4,1},Ra{4,1}] = gph(dyn,m,s,v,false,USE_APPROX_SV);  Ca{4,1} = s*iCa{4,1};
na = 2;  % number gph m-function variants   % 1 = gph.m, 2 = gphd.m
[Ma{3,2},Sa{3,2},iCa{3,2}]         = gphd(dyn,m,s);   Ca{3,2} = s*iCa{3,2};
[Ma{4,2},Sa{4,2},iCa{4,2},Ra{4,2}, ...
  ~,~,~,~,~,~,~,~,~,~,~,~]         = gphd(dyn,m,s,v,false,USE_APPROX_SV); Ca{4,2} = s*iCa{4,2};
Sonly=true; [~,~,~,RNan,~,~]       = gphd(dyn,m,s,v,Sonly,USE_APPROX_SV);
assert(all(isnan(RNan(:))));

% 3.2. Compute Numeric Outputs
mu = mvnrnd(m,s,NSAMPLES);     % sample uncertain test-input or test-input-mean
Mn = cell(4,1); Sn = cell(4,1); Cn = cell(4,1); Rn = cell(4,1); iCn =cell(4,1);
for ni=3:4  % number gph inputs
  mf = nan(NSAMPLES,E,1); vf = nan(NSAMPLES,E,E); icf = nan(NSAMPLES,D,E);
  Cn{ni} = nan(D,E);
  if ni==3, var_test_input = zeros(D); else var_test_input = v; end
  for i=1:NSAMPLES
    print_loop_progress(i,NSAMPLES,['MC test outputs (',num2str(ni),'-input case)']);
    [mf(i,:), vf(i,:,:), icf(i,:,:)] = gph(dyn, mu(i,:)', var_test_input); % TODO put inputs: ,false,USE_APPROX_S
  end
  Mn{ni} = mean(mf,1)'; Sn{ni} = cov(mf); iCn{ni} = mean(icf,1);
  for d=1:D, for e=1:E, c=cov(mu(:,d),mf(:,e)); Cn{ni}(d,e)=c(1,2); end; end
  Rn{ni} = permute(mean(vf,1),[2,3,1]);
end
Sn{3} = Sn{3} + Rn{3};

% 3.3. Display Output-Comparisons
str = @(x) (num2str(unwrap(x)'));
stra = {' ', 'd'};
for ni=3:4  % number gph inputs
  fprintf('\nGPH.M %u-INPUT MONTE CARLO TEST RESULTS (nsamples %3.0e):\n', ...
    ni, NSAMPLES);
  fprintf('==================\n');
  fprintf('M numeric      : %s\n', str(Mn{ni}));
  for a=1:na; fprintf('M analytic gph%s: %s\n', stra{a}, str(Ma{ni,a})); end
  fprintf('------------------\n');
  fprintf('S numeric      : %s\n', str(Sn{ni}));
  for a=1:na; fprintf('S analytic gph%s: %s\n', stra{a}, str(Sa{ni,a})); end
  fprintf('------------------\n');
  fprintf('C numeric      : %s\n', str(Cn{ni}));
  for a=1:na; fprintf('C analytic gph%s: %s\n', stra{a}, str(Ca{ni,a})); end
  if ni==4
    fprintf('------------------\n');
    fprintf('R numeric      : %s\n', str(Rn{ni}));
    for a=1:na; fprintf('R analytic gph%s: %s\n', stra{a}, str(Ra{ni,a})); end
    fprintf('------------------\n');
    fprintf('iC numeric      : %s\n', str(iCn{ni}));
    for a=1:na; fprintf('iC analytic gph%s: %s\n', stra{a}, str(iCa{ni,a}));end
  end
  fprintf('==================\n');
  mdiff = maxdiff({Mn{ni},Sn{ni},Cn{ni}}, {Ma{ni,1},Sa{ni,1},Ca{ni,1}});
  if ni==4,mdiff=max(mdiff,maxdiff({Rn{ni},iCn{ni}},{Ra{ni,1},iCa{ni,1}}));end
  fprintf('Maximum difference = %4.2e\n', mdiff);
end

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
% Example call: dd = cg_wrap(x,mm,[dyn.hyp.l],v,delta,@q,[1,3],2);
function cg = cg_wrap(varargin)
[delta, ~, ini] = deal(varargin{end-2:end});
[d,dy,dh] = checkgrad(@cg_f,varargin{ini},delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f. Example call:
% dd = checkgrad(@cg_f,mm,delta,x,[dyn.hyp.l],v,@q,[1,3],2);
function [f, df] = cg_f(x,varargin)
[f,nf,~,outi,ini] = deal(varargin{end-4:end}); out=cell(nargout(f),1);
if size(x,1)==size(x,2), x=(x+x')/2; end       % perturb matrices symmetrically
in = varargin(1:end-5); in{ini} = x;
if nargout == 1, [out{1:nf}] = feval(f,in{:}); f = out{outi(1)};
else [out{:}] = feval(f,in{:}); [f, df] = deal(out{outi}); end

function [z,c,dzdm,dcdm,dzdv,dcdv] = q(x, m, L, V)
% q function used in GPH.PDF document.
%
% z     nxE    exp negative quatratics
% c       E    -log(det)/2
% dzdm  nxE x D
% dcdm    E x D
% dzdv  nxE x D*D
% dcdv    E x D*D
if ~isempty(m), x = bsxfun(@minus, x, m'); end
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
  x, m, L, V, s, iK, beta)
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
% Copyright (C) 2014 by Carl Edward Rasmussen and Rowan McAllister 2014-11-28
if ~isempty(m), x = bsxfun(@minus, x, m'); end
[n, D, pE] = size(x); E = size(L,2); DD = D*D; nn=n*n; EE = E*E;
bQb = zeros(E); tiKQ = zeros(1,E);
iL = zeros(D,D,E); iLs = zeros(D,D,E); xiL = zeros(n,D,E); xiLs = zeros(n,D,E);
xiL2s = nan(n,D*(D+1)/2,E);
dbQbdm = nan(EE,D); dtikQdm = nan(E,D); dbQbdv = nan(EE,DD);
dtikQdv = nan(E,DD); dbQbds = nan(EE,DD); dtikQds = nan(E,DD);
[z, c, dzdm, ~, dzdv, dcdv] = q(x, [], L, V);
dzdm = permute(dzdm,[1,3,2]); dzdv = permute(dzdv,[1,3,2]);
persistent uiK dQdm dQds J I dQdv              % terms of unchanging size > n*n
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
for i=1:E
  for j=1:i
    ij = sub2ind2(E,i,j); ji = sub2ind2(E,j,i);
    nsym = i==j;
    iLij = iL(:,:,i)+iL(:,:,j);
    R = s*iLij+eye(D); t = exp(c(i)+c(j))/sqrt(det(R)); Y = R\s;
    Q = exp(bsxfun(@plus,z(:,i),z(:,j)')+maha(xiL(:,:,i),-xiL(:,:,j),Y/2));
    bQb(i,j) = beta(:,i)'*Q*beta(:,j)*t; bQb(j,i) = bQb(i,j);
    
    if nargout<3, continue, end
    Q = Q(:);
    Ydydm = -Y*iLij;
    dQdm = bsxfun(@times,Q,outern(@plus,dzdm(:,:,i),dzdm(:,:,j))+...
      outern(@plus,xiL(:,:,i)*Ydydm, xiL(:,:,j)*Ydydm)); % dyYydm term
    dbQbdm(ij,:) = prodd(beta(:,i)',dQdm,beta(:,j))*t;
    dbQbdm(ji,:) = dbQbdm(ij,:);
    
    if nargout<5, continue, end
    dlc2ds = -iLij/R/2; dlc2ds = dlc2ds(LD)';
    if ~nsym
      y = outern(@plus,xiL(:,:,i),xiL(:,:,j)); % nn x D
      yiR2 = y/(R*sqrt(2));
      dQds = bsxfun(@times,Q,bsxfun(@plus,dlc2ds,...
        outerds(@times,yiR2,yiR2,LD))); % dyYyds term
    else
      y = outerns(@plus,xiL(:,:,i),xiL(:,:,j),Ln); % nn/2 x D
      yiR2 = y/(R*sqrt(2));
      dQds(Ln,:) = bsxfun(@times,Q(Ln),bsxfun(@plus,dlc2ds,...
        outerds(@times,yiR2,yiR2,LD))); % dyYyds term
      dQds(un,:) = dQds(ln,:);
    end
    dbQbds(ij,LD) = prodd(beta(:,i)',dQds,beta(:,j))*t;
    dbQbds(ij,uD) = dbQbds(ij,lD); dbQbds(ji,:) = dbQbds(ij,:);
    
    if nargout<7, continue, end
    dlc2dv = iLs(:,:,i)*Y*iLs(:,:,i) + iLs(:,:,j)*Y*iLs(:,:,j);
    dlc2dv = dlc2dv(LD)';
    if ~nsym
      Yy = Y*y'; % D x nn
      iLYyi = (iLs(:,:,i)*(Yy - x(I,:,min(pE,i))'))'; % nn x D
      iLYyj = (iLs(:,:,j)*(Yy - x(J,:,min(pE,j))'))'; % nn x D
      dQdv = bsxfun(@times,Q, ...
        bsxfun(@plus, dlc2dv+dcdv(i,:)+dcdv(j,:), outern(@plus,dzdv(:,:,i),dzdv(:,:,j))) + ...
        outerds(@times,iLYyi,iLYyi,LD) + outerds(@times,iLYyj,iLYyj,LD) - outern(@plus,xiL2s(:,:,i),xiL2s(:,:,j))); % dyYydv term
    else
      Yy = Y*y'; % D x nn/2
      iLYyi = (iLs(:,:,i)*(Yy - x(I(Ln),:,min(pE,i))'))'; % nn/2 x D
      iLYyj = (iLs(:,:,j)*(Yy - x(J(Ln),:,min(pE,j))'))'; % nn/2 x D
      dQdv(Ln,:) = bsxfun(@times,Q(Ln), ...
        bsxfun(@plus, dlc2dv+dcdv(i,:)+dcdv(j,:), outerns(@plus,dzdv(:,:,i),dzdv(:,:,j),Ln)) + ...
        outerds(@times,iLYyi,iLYyi,LD) + outerds(@times,iLYyj,iLYyj,LD) - outerns(@plus,xiL2s(:,:,i),xiL2s(:,:,j),Ln)); % dyYydv term
      dQdv(un,:) = dQdv(ln,:);
    end
    dbQbdv(ij,LD) = prodd(beta(:,i)',dQdv,beta(:,j))*t;
    dbQbdv(ij,uD) = dbQbdv(ij,lD); dbQbdv(ji,:) = dbQbdv(ij,:);
  end
  if nargout>1; uiK = reshape(iK,nn,E)'; end
  if nargout>1, tiKQ(i)       = (uiK(i,:)*Q(:))*t; end
  if nargout>3, dtikQdm(i,:)  = (uiK(i,:)*dQdm)*t; end
  if nargout>5, dtikQds(i,LD) = (uiK(i,:)*dQds)*t; dtikQds(i,uD) = dtikQds(i,lD); end
  if nargout>7, dtikQdv(i,LD) = (uiK(i,:)*dQdv)*t; dtikQdv(i,uD) = dtikQdv(i,lD); end
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