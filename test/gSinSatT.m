function gSinSatT(fcn, m, v, i, e)

% Test the gSin and gSat functions. Check the three outputs using Monte Carlo,
% and the derivatives by finite differences. Carl Edward Rasmussen, 2012-06-20

if nargin < 2
  D = 4;
  m = randn(D,1);
  v = randn(D); v = v*v'+eye(D);
  i = [1;2; 4]; I = length(i);
  e = exp(randn(size(i)));
else
  D = length(m);
end

n = 1e6;               % monte Carlo sample size
delta = 1e-4;          % for finite difference approx

x = bsxfun(@plus, m, chol(v)'*randn(D,n));
switch func2str(fcn)
  case 'gSin', y = bsxfun(@times, e, sin(x(i,:)));
  case 'gSat', y = bsxfun(@times, e, 9*sin(x(i,:))/8+sin(3*x(i,:))/8);
  otherwise, error('Can only handle gSin and gSat')
end

[M, V, C] = fcn(m, v, i, e);
Q = cov([x' y']); Qv = Q(D+1:end,D+1:end); Qc  = v\Q(1:D,D+1:end);

disp(['mean: ', func2str(fcn), '  Monte Carlo'])
disp([M mean(y,2)]); disp([' ']);

disp(['var:  ', func2str(fcn), '  Monte Carlo'])
disp([V(:) Qv(:)]); disp([' ']);

disp(['cov:  ', func2str(fcn), '  Monte Carlo'])
disp([C(:) Qc(:)]); disp(' ');

disp('dMdm')
for j = 1:I
  d = checkgrad(@gSinT0, m, delta, v, i, e, j, fcn);
  disp(['this was element # ' num2str(j) '/' num2str(I)]);
  if d > 1e-6; keyboard; end
end
disp(' ');

disp('dVdm')
for j = 1:I*I
  d = checkgrad(@gSinT1, m, delta, v, i, e, j, fcn);
  disp(['this was element # ' num2str(j) '/' num2str(I*I)]);
  if d > 1e-6; keyboard; end
end
disp(' ');

disp('dCdm')
for j = 1:I*D
  d= checkgrad(@gSinT2, m, delta, v, i, e, j, fcn);
  disp(['this was element # ' num2str(j) '/' num2str(I*D)]);
  if d > 1e-6; keyboard; end
end
disp(' ');

disp('dMdv')
for j = 1:I
  d = checkgrad(@gSinT3, v(tril(ones(length(v)))==1), delta, m, i, e, j, fcn);
  disp(['this was element # ' num2str(j) '/' num2str(I)]);
  if d > 1e-6; keyboard; end
end
disp(' ');

disp('dVdv')
for j = 1:I*I
  d = checkgrad(@gSinT4, v(tril(ones(length(v)))==1), delta, m, i, e, j, fcn);
  disp(['this was element # ' num2str(j) '/' num2str(I*I)]);
  if d > 1e-6; keyboard; end
end
disp(' ');

disp('dCdv')
for j = 1:I*D
  d = checkgrad(@gSinT5, v(tril(ones(length(v)))==1), delta, m, i, e, j, fcn);
  disp(['this was element # ' num2str(j) '/' num2str(I*D)]);
  if d > 1e-6; keyboard; end
end


function [f, df] = gSinT0(m, v, i, e, j, fcn)
[M, V, C, dMdm] = fcn(m, v, i, e);
f = M(j); df = dMdm(j,:);

function [f, df] = gSinT1(m, v, i, e, j, fcn)
[M, V, C, dMdm, dVdm] = fcn(m, v, i, e);
dVdm = reshape(dVdm,[size(V) length(m)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = V(p,q); df = squeeze(dVdm(p,q,:));

function [f, df] = gSinT2(m, v, i, e, j, fcn)
[M, V, C, dMdm, dVdm, dCdm] = fcn(m, v, i, e);
dCdm = reshape(dCdm,[size(C) length(m)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = C(p,q); df = squeeze(dCdm(p,q,:));

function [f, df] = gSinT3(v, m, i, e, j, fcn)
d = length(m);
vv(tril(ones(d))==1) = v; vv = reshape(vv,d,d);
vv = vv + vv' - diag(diag(vv));
[M, V, C, dMdm, dVdm, dCdm, dMdv] = fcn(m, vv, i, e);
dMdv = reshape(dMdv,[length(M) size(vv)]);
f = M(j); df = squeeze(dMdv(j,:,:));
df = df+df'-diag(diag(df)); df = df(tril(ones(d))==1);

function [f, df] = gSinT4(v, m, i, e, j, fcn)
d = length(m);
vv(tril(ones(d))==1) = v; vv = reshape(vv,d,d);
vv = vv + vv' - diag(diag(vv));
[M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv] = fcn(m, vv, i, e);
dVdv = reshape(dVdv,[size(V) size(vv)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = V(p,q); df = squeeze(dVdv(p,q,:,:));
df = df+df'-diag(diag(df)); df = df(tril(ones(d))==1);

function [f, df] = gSinT5(v, m, i, e, j, fcn)
d = length(m);
vv(tril(ones(d))==1) = v; vv = reshape(vv,d,d);
vv = vv + vv' - diag(diag(vv));
[M, V, C, dMdm, dVdm, dCdm, dMdv, dVdv, dCdv] = fcn(m, vv, i, e);
dCdv = reshape(dCdv,[size(C) size(vv)]);
dd = length(M); p = fix((j+dd-1)/dd); q = j-(p-1)*dd;
f = C(p,q); df = squeeze(dCdv(p,q,:,:));
df = df+df'-diag(diag(df)); df = df(tril(ones(d))==1);
