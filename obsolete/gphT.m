function gphT(deriv, dyn, m, s, v, delta)

% Test function for gph.m which:
%  1. Draws sample gph input data from a SEard GP,
%  2. computes analytic gph.m outputs {Ma, Sa, Ca, Ra},
%  3. computes numeric {Mn, Sn, Cn, Rn} gph outputs using MC samples,
%  4. compares {Ma, Sa, Ca, Ra} against {Mn, Sn, Cn, Rn}.
%
% See also <a href="gph.pdf">gph.pdf</a>, GPH.M, GPHD.M.
% Copyright (C) 2014 by Rowan McAllister 2014-08-19

NSAMPLES = 1e4;
SEED = 18;
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
rand('seed',SEED); randn('seed',SEED);
addpath('../gp'); addpath('../util');

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
  dyn.inputs = x; dyn.target = nan(n,E); dyn.iK = nan*K; dyn.beta = nan(n,E);
  for e=1:E;
    K(:,:,e) = K(:,:,e) + sn(e)*eye(n);
    dyn.target(:,e) = chol(K(:,:,e))'*randn(n,1) + my(:,e) + sn(e)*randn(n,1);
    dyn.iK(:,:,e) = inv(K(:,:,e));
    dyn.beta(:,e) = dyn.iK(:,:,e) * (dyn.target(:,e) - my(:,e));
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
  deriv = {'dMdm', 'dSdm', 'dCdm', 'dRdm', 'dMds', 'dSds', 'dCds', 'dRds', ...
    'dMdv', 'dSdv', 'dCdv', 'dRdv'};
elseif strcmp(deriv,'q'), deriv={'dzdm', 'dcdm', 'dzdv', 'dcdv'};
elseif strcmp(deriv,'Q'), deriv={'dbQbdm','dtikQdm','dbQbds','dtikQds','dbQbdv','dtikQdv'};
end
if iscell(deriv), ntests = numel(deriv); else ntests = 1; end
q_args = {x, m, [dyn.hyp.l], v, @q, 2, delta};       % 2 non-derivative outputs
Q_args = {x, m, [dyn.hyp.l], v, s, dyn.iK, dyn.beta, @Q, 2, delta};
gph_args = {dyn, m, s, v, @gphd, nargout(@gph), delta};
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
    case 'dRdm', cg{c} = cg_wrap(gph_args{:}, [4, 8], 2);
    case 'dMds', cg{c} = cg_wrap(gph_args{:}, [1, 9], 3);
    case 'dSds', cg{c} = cg_wrap(gph_args{:}, [2,10], 3);
    case 'dCds', cg{c} = cg_wrap(gph_args{:}, [3,11], 3);
    case 'dRds', cg{c} = cg_wrap(gph_args{:}, [4,12], 3);
    case 'dMdv', cg{c} = cg_wrap(gph_args{:}, [1,13], 4);
    case 'dSdv', cg{c} = cg_wrap(gph_args{:}, [2,14], 4);
    case 'dCdv', cg{c} = cg_wrap(gph_args{:}, [3,15], 4);
    case 'dRdv', cg{c} = cg_wrap(gph_args{:}, [4,16], 4);
    otherwise, disp('WARNING: Unknown derivative test requested.');
  end
end
print_derivative_test_results(test_names, cg, EPSILON);

% 3. TEST GHP OUTPUTS ---------------------------------------------------------

% 3.1. Compute Analytic Outputs            % 3 = 3-input-case, 4 = 4-input-case
[Ma{3,1}, Sa{3,1}, isCa31]          = gph(dyn, m, s);    Ca{3,1} = s*isCa31;
[Ma{4,1}, Sa{4,1}, isCa41, Ra{4,1}] = gph(dyn, m, s, v); Ca{4,1} = s*isCa41;
na = 2;  % number gph m-function variants   % 1 = gph.m, 2 = gphd.m
[Ma{3,2}, Sa{3,2}, isCa32]          = gphd(dyn, m, s);    Ca{3,2} = s*isCa32;
[Ma{4,2}, Sa{4,2}, isCa42, Ra{4,2}, ...
  ~,~,~,~,~,~,~,~,~,~,~,~] = gphd(dyn, m, s, v); Ca{4,2} = s*isCa42;

% 3.2. Compute Numeric Outputs
mu = mvnrnd(m,s,NSAMPLES);     % sample uncertain test-input or test-input-mean
Mn = cell(4,1); Sn = cell(4,1); Cn = cell(4,1); Rn = cell(4,1);
for ni=3:4  % number gph inputs
  mf = nan(NSAMPLES,E,1); vf = nan(NSAMPLES,E,E); Cn{ni} = nan(D,E);
  if ni==3, var_test_input = zeros(D); else var_test_input = v; end
  for i=1:NSAMPLES
    print_loop_progress(i,NSAMPLES,['MC test outputs (',ni,'-input case)']);
    [mf(i,:), vf(i,:,:), ~] = gph(dyn, mu(i,:)', var_test_input);
  end
  Mn{ni} = mean(mf,1)'; Sn{ni} = cov(mf);
  for d=1:D, for e=1:E, c=cov(mu(:,d),mf(:,e)); Cn{ni}(d,e)=c(1,2); end; end
  Rn{ni} = permute(mean(vf,1),[2,3,1]);
end
Sn{3} = Sn{3} + Rn{3};

% 3.3. Display Output-Comparisons
str = @(x) (num2str(unwrap(x)'));
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
for ni=3:4  % number gph inputs
  fprintf('\nGPH.M %u-INPUT MONTE CARLO TEST RESULTS (nsamples %3.0e):\n', ...
    ni, NSAMPLES);
  fprintf('==================\n');
  fprintf('M numeric      : %s\n', str(Mn{ni}));
  for a=1:na; fprintf('M analytic gph%u: %s\n', a, str(Ma{ni,a})); end
  fprintf('------------------\n');
  fprintf('S numeric      : %s\n', str(Sn{ni}));
  for a=1:na; fprintf('S analytic gph%u: %s\n', a, str(Sa{ni,a})); end
  fprintf('------------------\n');
  fprintf('C numeric      : %s\n', str(Cn{ni}));
  for a=1:na; fprintf('C analytic gph%u: %s\n', a, str(Ca{ni,a})); end
  if ni==4
    fprintf('------------------\n');
    fprintf('R numeric      : %s\n', str(Rn{ni}));
    for a=1:na; fprintf('R analytic gph%u: %s\n', a, str(Ra{ni,a})); end
  end
  fprintf('==================\n');
  mdiff = max_diff({Mn{ni},Sn{ni},Cn{ni}}, {Ma{ni,1},Sa{ni,1},Ca{ni,1}});
  if ni==4, mdiff = max(mdiff, max_diff(Rn{ni},Ra{ni,1})); end
  fprintf('Maximum difference = %4.2e\n', mdiff);
end

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
% Example call: dd = cg_wrap(x,mm,[dyn.hyp.l],v,delta,@q,[1,3],2);
function cg = cg_wrap(varargin)
[delta, ~, ini] = deal(varargin{end-2:end});
[d dy dh] = checkgrad(@cg_f,varargin{ini},delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f. Example call:
% dd = checkgrad(@cg_f,mm,delta,x,[dyn.hyp.l],v,@q,[1,3],2);
function [f, df] = cg_f(x,varargin)
[f,nf,~,outi,ini] = deal(varargin{end-4:end}); out=cell(nargout(f),1);
if size(x,1)==size(x,2), x=(x+x')/2; end       % perturb matrices symmetrically
in = varargin(1:end-5); in{ini} = x;
if nargout == 1, [out{1:nf}] = feval(f,in{:}); f = out{outi(1)};
else [out{:}] = feval(f,in{:}); [f, df] = deal(out{outi}); end

% z     nxE    exp negative quatratics
% c     1xE    -log(det)/2
% dzdm  nxE x D
% dcdm  1xE x D
% dzdv  nxE x DxD
% dcdv  1xE x DxD
function [z c dzdm dcdm dzdv dcdv] = q(x_, m, L, V)
x = bsxfun(@minus, x_, m');
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
function [bQb tiKQ dbQbdm dtikQdm dbQbds dtikQds dbQbdv dtikQdv] = Q(...
  x_, m, L, V, s, iK, beta)
x = bsxfun(@minus, x_, m');
[n, D, pE] = size(x); E = size(L,2);
bQb = zeros(E); tiKQ = zeros(1,E);
iL = zeros(D,D,E); xiL = zeros(n,D,E); xiL2 = nan(n,D,D,E);
dbQbdm = nan(E,E,D); dtikQdm = nan(1,E,D); dbQbdv = nan(E,E,D,D);
dtikQdv = nan(1,E,D,D); dbQbds = nan(E,E,D,D); dtikQds = nan(1,E,D,D);
[z, c, dzdm, ~, dzdv, dcdv] = q(x_, m, L, V);
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