function gTrigNT(m, v, i, e)

% Test the gTrig function. Check the three outputs using Monte Carlo, and the
% derivatives using finite differences. Carl Edward Rasmussen, 2014-04-01

if ~nargin
  D = 4;
  m = randn(D,1);
  v = randn(D); v = v*v'+eye(D);
  i = [2; 4]; I = 2*length(i);
  e = exp(randn(size(i)));
else
  D = length(m);
end

n = 1e6;                                              % monte Carlo sample size
delta = 1e-4;                                    % for finite difference approx

x = bsxfun(@plus, m, chol(v)'*randn(D,n));
y = bsxfun(@times, [e; e], [sin(x(i,:)); cos(x(i,:))]);
y = y(reshape(1:I,I/2,2)',:);                                    % reorder rows

[M, V, C] = gTrig(m, v, i, e);
Q = cov([x' y']); Qv = Q(D+1:end,D+1:end); Qc  = v\Q(1:D,D+1:end);

disp(['mean: gTrig Monte Carlo'])
disp([M mean(y,2)]); disp([' ']);

disp(['var:  gTrig Monte Carlo'])
disp([V(:) Qv(:)]); disp([' ']);

disp(['cov:  gTrig Monte Carlo'])
disp([C(:) Qc(:)]); disp(' ');

% outputting gradients should not alter non-gradient outputs
[M1, V1, C1] = gTrigN(m, v, i, e);
[M2, V2, C2,~,~,~,~,~,~] = gTrigN(m, v, i, e);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
assert(max_diff({M1, V1, C1}, {M2, V2, C2})<1e-10, ...
  'gTrigNT: calling derivatives alters non-derivate outputs!')

E = D + I;
disp('dMdm')
for j = 1:E
  checkgrad(@gTrigT0, m, delta, v, i, e, j);
  disp(['this was element # ' num2str(j) '/' num2str(E)]);
end
disp(' ');

disp('dVdm')
for j = 1:E*E
  checkgrad(@gTrigT1, m, delta, v, i, e, j);
  disp(['this was element # ' num2str(j) '/' num2str(E*E)]);
end
disp(' ');

disp('dCdm')
for j = 1:E*D
  checkgrad(@gTrigT2, m, delta, v, i, e, j);
  disp(['this was element # ' num2str(j) '/' num2str(E*D)]);
end
disp(' ');

disp('dMdv')
for j = 1:E
  checkgrad(@gTrigT3, v(tril(ones(length(v)))==1), delta, m, i, e, j);
  disp(['this was element # ' num2str(j) '/' num2str(E)]);
end
disp(' ');

disp('dVdv')
for j = 1:E*E
  checkgrad(@gTrigT4, v(tril(ones(length(v)))==1), delta, m, i, e, j);
  disp(['this was element # ' num2str(j) '/' num2str(E*E)]);
end
disp(' ');

disp('dCdv')
for j = 1:E*D
  checkgrad(@gTrigT5, v(tril(ones(length(v)))==1), delta, m, i, e, j);
  disp(['this was element # ' num2str(j) '/' num2str(E*D)]);
end


function [f, df] = gTrigT0(m, v, i, e, j)
[M, V, C, dMdm] = gTrigN(m, v, i, e);
f = M(j); df = dMdm(j,:);

function [f, df] = gTrigT1(m, v, i, e, j)
[M, V, C, dMdm, dVdm] = gTrigN(m, v, i, e);
dVdm = reshape(dVdm,[size(V) length(m)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = V(p,q); df = squeeze(dVdm(p,q,:));

function [f, df] = gTrigT2(m, v, i, e, j)
[M, V, C, dMdm, dVdm, dCdm] = gTrigN(m, v, i, e);
dCdm = reshape(dCdm,[size(C) length(m)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = C(p,q); df = squeeze(dCdm(p,q,:));

function [f, df] = gTrigT3(v, m, i, e, j)
d = length(m);
vv(tril(ones(d))==1) = v; vv = reshape(vv,d,d);
vv = vv + vv' - diag(diag(vv));
[M, V, C, dMdm, dVdm, dCdm, dMdv] = gTrigN(m, vv, i, e);
dMdv = reshape(dMdv,[length(M) size(vv)]);
f = M(j); df = squeeze(dMdv(j,:,:));
df = df+df'-diag(diag(df)); df = df(tril(ones(d))==1);

function [f, df] = gTrigT4(v, m, i, e, j)
d = length(m);
vv(tril(ones(d))==1) = v; vv = reshape(vv,d,d);
vv = vv + vv' - diag(diag(vv));
[M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv] = gTrigN(m, vv, i, e);
dVdv = reshape(dVdv,[size(V) size(vv)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = V(p,q); df = squeeze(dVdv(p,q,:,:));
df = df+df'-diag(diag(df)); df = df(tril(ones(d))==1);

function [f, df] = gTrigT5(v, m, i, e, j)
d = length(m);
vv(tril(ones(d))==1) = v; vv = reshape(vv,d,d);
vv = vv + vv' - diag(diag(vv));
[M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv] = gTrigN(m, vv, i, e);
dCdv = reshape(dCdv,[size(C) size(vv)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = C(p,q); df = squeeze(dCdv(p,q,:,:));
df = df+df'-diag(diag(df)); df = df(tril(ones(d))==1);
