function [dd dy dh] = gpaT(deriv, gp, m, s)

if isempty(deriv)
D = gp.D; A = numel(gp.angi); n = size(gp.target,1);
nota = setdiff(1:size(m,1),gp.angi);
notA = [nota D+1:D+2*A];
hypA = [nota reshape(repmat(gp.angi,2,1),1,2*A)];

N = 1000000;
xs = bsxfun(@plus,chol(s)'*randn(6,N),m);
for j = 1:numel(gp.angi)
  xs(end+1,:) = sin(xs(gp.angi(j),:));
  xs(end+1,:) = cos(xs(gp.angi(j),:));
end

p = zeros(N,size(gp.target,2));
for j=1:size(gp.target,2)
  z = bsxfun(@times,gp.inputs(:,notA),exp(-gp.hyp(j).l(hypA)'));
  zs = bsxfun(@times,xs(notA,:)',exp(-gp.hyp(j).l(hypA)'));  
  K = exp(2*gp.hyp(j).s-maha(z,z)/2) + exp(2*gp.hyp(j).n)*eye(n);
  k = exp(2*gp.hyp(j).s-maha(zs,z)/2);
  mu = gp.inputs(:,1:D)*gp.hyp(j).m + gp.hyp(j).b;
  mus = xs(1:D,:)'*gp.hyp(j).m + gp.hyp(j).b;
  s2 = exp(2*gp.hyp(j).s) - sum(k.*(k/K),2);
  p(:,j) = k*(K\(gp.target(:,j)-mu)) + mus + sqrt(s2).*randn(N,1);
end 
    
disp(['mean']);
mean(p)

disp(['variance']);
cov(p)

disp(['input output covariance']);
xs(1:D,:)*p/N-mean(xs(1:6,:)')'*mean(p) 
end


delta = 1e-4; 
D = length(m);                                                      % input size

switch deriv
  
  case 'dMdm'
      [dd dy dh] = checkgrad(@gpT0, m, delta, gp, s);
 
  case 'dSdm'
      [dd dy dh] = checkgrad(@gpT1, m, delta, gp, s);
 
  case 'dVdm'
      [dd dy dh] = checkgrad(@gpT2, m, delta, gp, s);
 
  case 'dMds'
      [dd dy dh] = checkgrad(@gpT3, s(tril(ones(D))==1), delta, gp, m);
    
  case 'dSds'
      [dd dy dh] = checkgrad(@gpT4, s(tril(ones(D))==1), delta, gp, m);
 
  case 'dVds'
      [dd dy dh] = checkgrad(@gpT5, s(tril(ones(D))==1), delta, gp, m);
 
  case 'dMdp'
      p = unwrap(dynmodel);
      [dd dy dh] = checkgrad(@gpT6, p, delta, dynmodel, m, s) ;
 
  case 'dSdp'
      p = unwrap(dynmodel);
      [dd dy dh] = checkgrad(@gpT7, p, delta, dynmodel, m, s) ;
 
  case 'dVdp'
      p = unwrap(dynmodel);
      [dd dy dh] = checkgrad(@gpT8, p, delta, dynmodel, m, s) ;

end


function [f, df] = gpT0(m, gp, s)                             % dMdm
if nargout == 1
  M = gp.pred(m, s);
else
  [M, S, V, dMdm] = gp.pred(m, s);
  df = dMdm;
end
f = M;

function [f, df] = gpT1(m, gp, s)                             % dSdm
if nargout == 1
  [M, S] = gp.pred(m, s);
else
  [M, S, V, dMdm, dSdm] = gp.pred(m, s);
  df = dSdm;
end
f = S;

function [f, df] = gpT2(m, gp, s)                             % dVdm
if nargout == 1
  [M, S, V] = gp.pred(m, s);
else
  [M, S, V, dMdm, dSdm, dVdm] = gp.pred(m, s);
  df = dVdm;
end
f = V;

function [f, df] = gpT3(s, gp, m)                             % dMds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  M = gp.pred(m, s);
else
  [M, S, V, dMdm, dSdm, dVdm, dMds] = gp.pred(m, s);
  dd = length(M); dMds = reshape(dMds,dd,d,d); df = zeros(dd,d*(d+1)/2);
    for i=1:dd; 
        dMdsi(:,:) = dMds(i,:,:); dMdsi = dMdsi + dMdsi'-diag(diag(dMdsi)); 
        df(i,:) = dMdsi(tril(ones(d))==1);
    end
end
f = M;

function [f, df] = gpT4(s, gp, m)                             % dSds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  [M, S] = gp.pred(m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds] = gp.pred(m, s);
    dd = length(M); dSds = reshape(dSds,dd,dd,d,d); df = zeros(dd,dd,d*(d+1)/2);
    for i=1:dd; for j=1:dd                                      
        dSdsi(:,:) = dSds(i,j,:,:); dSdsi = dSdsi+dSdsi'-diag(diag(dSdsi)); 
        df(i,j,:) = dSdsi(tril(ones(d))==1);
    end; end
end
f = S;

function [f, df] = gpT5(s, gp, m)                             % dVds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  [M, S, V] = gp.pred(m, s);
else
  [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gp.pred(m, s);
    dd = length(M); dVds = reshape(dVds,d,dd,d,d); df = zeros(d,dd,d*(d+1)/2);
    for i=1:d; for j=1:dd
        dCdsi = squeeze(dVds(i,j,:,:)); dCdsi = dCdsi+dCdsi'-diag(diag(dCdsi)); 
        df(i,j,:) = dCdsi(tril(ones(d))==1);
    end; end
end
f = V;

function [f, df] = gpT6(p, gp, m, s)                          % dMdp
gp = rewrap(gp, p);
if nargout == 1
  M = gp.pred(m, s);
else
  [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp] = ...
                                                gp.pred(m, s);
  df = dMdp;
end
f = M;

function [f, df] = gpT7(p, gp, m, s)                          % dSdp
gp = rewrap(gp, p);
if nargout == 1
    [M, S] = gp.pred(m, s);
else
    [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp] = ...
                                                gp.pred(m, s);
    df = dSdp;
end
f = S;

function [f, df] = gpT8(p, gp, m, s)
gp = rewrap(gp, p);
if nargout == 1
    [M, S, V] = gp.pred(m, s);
else
    [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp, dVdp] = ...
                                                gp.pred(m, s);
    df = dVdp;

end
f = V;
