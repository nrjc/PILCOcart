function gTrighNT(m, s, v, i, e)

% Test the GTRIGHN function. Check the three outputs using Monte Carlo, and the
% derivatives using finite differences. Carl Edward Rasmussen,
% Rowan McAllister 2014-12-09

if ~nargin
  D = 3;
  rng(1);
  m = randn(D,1);
  s = zeros(D); s = s*s'+eye(D);
  v = randn(D); v = v*v'+eye(D);
  i = [1;3];
  e = exp(randn(size(i)));
else
  D = length(m);
end
DA = D + 2*length(i);

n = 1e5;                                              % monte Carlo sample size
delta = 1e-4;                                    % for finite difference approx
epsilon = 1e-5;               % 'pass' threshold for low enough checkgrad error

% 1. Assert consistent output with gTrig.m if input-mean does not vary --------

[Mh,~,Ch,Vh] = gTrighN(m, 0*s, v, i, e);
[M, V, C] = gTrigN(m, v, i, e);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({Mh,Vh,Ch}, {M,V,C}) < 1e-10, ...
  'gTrighNT FAIL: gTrighN does not revert to gTrigN with non-random input mean.');

% 2. outputting gradients should not alter non-gradient outputs
[M1,S1,C1,V1] = gTrighN(m, s, v, i, e);
[M2,S2,C2,V2,~,~,~,~,~,~,~,~,~,~,~,~] = gTrighN(m, s, v, i, e);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({M1,S1,C1,V1}, {M2,S2,C2,V2})<1e-10, ...
  'gTrigNT: calling derivatives alters non-derivate outputs!')

% 3. TEST OUTPUT-DERIVATIVES --------------------------------------------------

args = {m, s, v, i, e, @gTrighN, 4, delta};          % 4 non-derivative outputs
%[M, S, C, V, ...
%  dMdm, dSdm, dCdm, dVdm, ...
%  dMds, dSds, dCds, dVds, ...
%  dMdv, dSdv, dCdv, dVdv] = gTrighN(m, s, v, i, e)
ntests = 12; test = cell(ntests,1); cg = cell(ntests,1); k = 0;
k=k+1; test{k} = 'dMdm'; cg{k} = cgwrap(args{:}, [1, 5], 1);
k=k+1; test{k} = 'dSdm'; cg{k} = cgwrap(args{:}, [2, 6], 1);
k=k+1; test{k} = 'dCdm'; cg{k} = cgwrap(args{:}, [3, 7], 1);
k=k+1; test{k} = 'dVdm'; cg{k} = cgwrap(args{:}, [4, 8], 1);
k=k+1; test{k} = 'dMds'; cg{k} = cgwrap(args{:}, [1, 9], 2);
k=k+1; test{k} = 'dSds'; cg{k} = cgwrap(args{:}, [2,10], 2);
k=k+1; test{k} = 'dCds'; cg{k} = cgwrap(args{:}, [3,11], 2);
k=k+1; test{k} = 'dVds'; cg{k} = cgwrap(args{:}, [4,12], 2);
k=k+1; test{k} = 'dMdv'; cg{k} = cgwrap(args{:}, [1,13], 3);
k=k+1; test{k} = 'dSdv'; cg{k} = cgwrap(args{:}, [2,14], 3);
k=k+1; test{k} = 'dCdv'; cg{k} = cgwrap(args{:}, [3,15], 3);
k=k+1; test{k} = 'dVdv'; cg{k} = cgwrap(args{:}, [4,16], 3);
print_derivative_test_results(test, cg, epsilon);

% 4. TEST OUTPUTS -------------------------------------------------------------

% Check consistency with gTrig when s or v zerod
[M1,S1] = gTrigN(m, s, i, e);
[M2,S2] = gTrighN(m, s, 0*v, i, e);
[M3,~,~,S3] = gTrighN(m, 0*s, s, i, e);
assert(max_diff({M1,S1},{M2,S2})< 1e-10);
assert(max_diff({M1,S1},{M3,S3})< 1e-10);

% Analytic
[Ma, Sa, Ca, Va] = gTrighN(m, s, v, i, e); Csa=s*Ca; Cva=v*Ca;

% Numeric
if all(~s(:))
  z = repmat(m,[1,n]);
else
  z = bsxfun(@plus, m, chol(s)'*randn(D,n));
end
mn = nan(DA,n);
Vn = zeros(DA);
Cvn = zeros(D,DA);
for k=1:n
  print_loop_progress(k,n,'outputs with MC');
  [mn(:,k), vn, cvn] = gTrigN(z(:,k),v,i,e);
  Vn = Vn + vn;
  Cvn = Cvn + cvn;
end
Mn = mean(mn,2); Sn = cov(mn'); Vn = Vn/n; Cvn=v*Cvn/n; Csn = nan(D,DA);
for d=1:D, for ii=1:DA, c=cov(z(d,:)',mn(ii,:)); Csn(d,ii)=c(1,2); end; end

disp('M: mean-of-mean: gTrig Monte Carlo')
disp('   (hier)    (numeric)')
disp([Ma Mn]); disp(' ');

disp('S: var-of-mean:  gTrig Monte Carlo')
disp('   (hier)    (numeric)')
disp([Sa(:) Sn(:)]); disp(' ');

disp('V: mean-of-var:  gTrig Monte Carlo')
disp('   (hier)    (numeric)')
disp([Va(:) Vn(:)]); disp(' ');

disp('Cs: cov:  gTrig Monte Carlo')
disp([Csa(:) Csn(:)]); disp(' ');

disp('Cv: cov:  gTrig Monte Carlo')
disp([Cva(:) Cvn(:)]); disp(' ');

% 5. FUCNTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
% Example call: dd = cg_wrap(x,mm,[dyn.hyp.l],v,delta,@q,[1,3],2);
function cg = cgwrap(varargin)
[delta, ~, ini] = deal(varargin{end-2:end});
[d,dy,dh] = checkgrad(@cgf,varargin{ini},delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f. Example call:
% dd = checkgrad(@cg_f,mm,delta,x,[dyn.hyp.l],v,@q,[1,3],2);
function [f, df] = cgf(x,varargin)
[f,nf,~,outi,ini] = deal(varargin{end-4:end}); out=cell(nargout(f),1);
if size(x,1)==size(x,2), x=(x+x')/2; end       % perturb matrices symmetrically
in = varargin(1:end-5); in{ini} = x;
if nargout == 1, [out{1:nf}] = feval(f,in{:}); f = out{outi(1)};
else [out{:}] = feval(f,in{:}); [f, df] = deal(out{outi}); end