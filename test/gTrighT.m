function gTrighT(m, s, v, i, e)

% Test the gTrig function. Check the three outputs using Monte Carlo, and the
% derivatives using finite differences. Carl Edward Rasmussen,
% Rowan McAllister 2014-12-09

if ~nargin
  D = 4;
  m = randn(D,1);
  s = zeros(D); s = s*s'+eye(D);
  v = randn(D); v = v*v'+eye(D);
  i = [1;3]; I = 2*length(i);
  e = exp(randn(size(i)));
else
  D = length(m);
end

n = 1e4;                                              % monte Carlo sample size
delta = 1e-4;                                    % for finite difference approx
epsilon = 1e-5;               % 'pass' threshold for low enough checkgrad error

% 1. Assert consistent output with gTrig.m if input-mean does not vary --------

[Mh,~,Ch,Vh] = gTrigh(m, 0*s, v, i, e);
[M, V, C] = gTrig(m, v, i, e);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({Mh,Vh,Ch}, {M,V,C}) < 1e-10, ...
  'gTrighT FAIL: gTrigh does not revert to gTrig with non-random input mean.');

% 2. TEST OUTPUT-DERIVATIVES --------------------------------------------------

args = {m, s, v, i, e, @gTrigh, 4, delta};           % 4 non-derivative outputs
%[M, S, C, V, ...
%  dMdm, dSdm, dCdm, dVdm, ...
%  dMds, dSds, dCds, dVds, ...
%  dMdv, dSdv, dCdv, dVdv] = gTrigh(m, s, v, i, e)
ntests = 12; test = cell(ntests,1); cg = cell(ntests,1); k = 0;
k=k+1; test{k} = 'dMdm'; cg{k} = cg_wrap(args{:}, [1, 5], 1);
k=k+1; test{k} = 'dSdm'; cg{k} = cg_wrap(args{:}, [2, 6], 1);
k=k+1; test{k} = 'dCdm'; cg{k} = cg_wrap(args{:}, [3, 7], 1);
k=k+1; test{k} = 'dVdm'; cg{k} = cg_wrap(args{:}, [4, 8], 1);
k=k+1; test{k} = 'dMds'; cg{k} = cg_wrap(args{:}, [1, 9], 2);
k=k+1; test{k} = 'dSds'; cg{k} = cg_wrap(args{:}, [2,10], 2);
k=k+1; test{k} = 'dCds'; cg{k} = cg_wrap(args{:}, [3,11], 2);
k=k+1; test{k} = 'dVds'; cg{k} = cg_wrap(args{:}, [4,12], 2);
k=k+1; test{k} = 'dMdv'; cg{k} = cg_wrap(args{:}, [1,13], 3);
k=k+1; test{k} = 'dSdv'; cg{k} = cg_wrap(args{:}, [2,14], 3);
k=k+1; test{k} = 'dCdv'; cg{k} = cg_wrap(args{:}, [3,15], 3);
k=k+1; test{k} = 'dVdv'; cg{k} = cg_wrap(args{:}, [4,16], 3);
print_derivative_test_results(test, cg, epsilon);

% 3. TEST OUTPUTS -------------------------------------------------------------

% Check consistency with gTrig when s or v zerod
[M1,S1] = gTrig(m, s, i, e);
[M2,S2] = gTrigh(m, s, 0*v, i, e);
[M3,~,~,S3] = gTrigh(m, 0*s, s, i, e);
assert(max_diff({M1,S1},{M2,S2})< 1e-10);
assert(max_diff({M1,S1},{M3,S3})< 1e-10);

% Analytic (joint)
[Mj_, Sj_] = gTrig([m;m], [s,s;s,s+v], [i;D+i], [e;e]);
i1 = 1:I; i2 = I+1:2*I;
Mj = Mj_(i1); Sj = Sj_(i1,i1); Vj = Sj_(i2,i2)-Sj;

% Analytic (hierarchical)
[Ma, Sa, Ca, Va] = gTrigh(m, s, v, i, e); Csa=s*Ca; Cva=v*Ca;

% Numeric
if all(~s(:))
  z = repmat(m,[1,n]);
else
  z = bsxfun(@plus, m, chol(s)'*randn(D,n));
end
mn = nan(I,n);
Vn = zeros(I);
Cvn = zeros(D,I);
for k=1:n
  print_loop_progress(k,n,'outputs with MC');
  [mn(:,k), vn, cvn] = gTrig(z(:,k),v,i,e);
  Vn = Vn + vn;
  Cvn = Cvn + cvn;
end
Mn = mean(mn,2); Sn = cov(mn'); Vn = Vn/n; Cvn=v*Cvn/n; Csn = nan(D,I);
for d=1:D, for ii=1:I, c=cov(z(d,:)',mn(ii,:)); Csn(d,ii)=c(1,2); end; end

disp('M: mean-of-mean: gTrig Monte Carlo')
disp('   (joint)   (hier)    (numeric)')
disp([Mj Ma Mn]); disp(' ');

disp('S: var-of-mean:  gTrig Monte Carlo')
disp('   (joint)   (hier)    (numeric)')
disp([Sj(:) Sa(:) Sn(:)]); disp(' ');

disp('V: mean-of-var:  gTrig Monte Carlo')
disp('   (joint)   (hier)    (numeric)')
disp([Vj(:) Va(:) Vn(:)]); disp(' ');

disp('Cs: cov:  gTrig Monte Carlo')
disp([Csa(:) Csn(:)]); disp(' ');

disp('Cv: cov:  gTrig Monte Carlo')
disp([Cva(:) Cvn(:)]); disp(' ');

disp('Note: I am currently unsure if (joint) test is correct - Rowan.')

% 4. FUCNTIONS ----------------------------------------------------------------

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