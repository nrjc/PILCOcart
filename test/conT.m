function [dd dy dh] = conT(deriv, policy, m, s, delta)

% test derivatives of controller functions. 2012-06-23

if nargin < 5; delta = 1e-4; end

if nargin == 0
  D = 2;
  d = 3;
  policy.w = randn(D,d);
  policy.b = randn(D,1);
  m = randn(d,1);
  s = randn(d); s = s*s';
  policy.maxU = '2 3';
  policy.fcn = @(policy, m, s)conCat(@conLin, @gSat, policy, m, s);
end

D = length(policy.maxU);
d = length(m);

switch deriv
  case 'dMdm'
      [dd dy dh] = checkgrad(@conT0, m, delta, policy, s);

  case 'dSdm'
      [dd dy dh] = checkgrad(@conT1, m, delta, policy, s);

  case 'dCdm'
      [dd dy dh] = checkgrad(@conT2, m, delta, policy, s);

  case 'dMds'
      [dd dy dh] = checkgrad(@conT3, s(tril(ones(d))==1), delta, policy, m);

  case 'dSds'
      [dd dy dh] = checkgrad(@conT4, s(tril(ones(d))==1), delta, policy, m);

  case 'dCds'
      [dd dy dh] = checkgrad(@conT5, s(tril(ones(d))==1), delta, policy, m);

  case 'dMdp'
      [dd dy dh] = checkgrad(@conT6, policy.p, delta, policy, m, s);
 
  case 'dSdp'
      [dd dy dh] = checkgrad(@conT7, policy.p, delta, policy, m, s);

  case 'dCdp'
      [dd dy dh] = checkgrad(@conT8, policy.p, delta, policy, m, s);

end

function [f, df] = conT0(m, policy, s)                                  % dMdm
if nargout < 2
    M = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm] = policy.fcn(policy, m, s);
    df = dMdm;
end
f = M; 

function [f, df] = conT1(m, policy, s)                                  % dSdm
if nargout < 2
    [M, S] = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm] = policy.fcn(policy, m, s);
     df = dSdm;
end
f = S;

function [f, df] = conT2(m, policy, s)                                  % dCdm
if nargout < 2
    [M, S, C] = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm] = policy.fcn(policy, m, s);
    df = dCdm;
end
f = C;

function [f, df] = conT3(s, policy, m)                                  % dMds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout < 2
    M = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds] = policy.fcn(policy, m, s);
    dd = length(M); dMds = reshape(dMds,dd,d,d); df = zeros(dd,d*(d+1)/2);
    for i=1:dd; 
        dMdsi(:,:) = dMds(i,:,:); dMdsi = dMdsi + dMdsi'-diag(diag(dMdsi)); 
        df(i,:) = dMdsi(tril(ones(d))==1);
    end
end
f = M;

function [f, df] = conT4(s, policy, m)                                  % dSds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout < 2
    [M, S] = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds] = policy.fcn(policy, m, s);
    dd = length(M); dSds = reshape(dSds,dd,dd,d,d); df = zeros(dd,dd,d*(d+1)/2);
    for i=1:dd; for j=1:dd
        dSdsi(:,:) = dSds(i,j,:,:); dSdsi = dSdsi+dSdsi'-diag(diag(dSdsi)); 
        df(i,j,:) = dSdsi(tril(ones(d))==1);
    end; end
end
f = S;

function [f, df] = conT5(s, policy, m)                                  % dCds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout < 2
    [M, S, C] = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds] = policy.fcn(policy, m, s);
    dd = length(M); dCds = reshape(dCds,d,dd,d,d); df = zeros(d,dd,d*(d+1)/2);
    for i=1:d; for j=1:dd
        dCdsi = squeeze(dCds(i,j,:,:)); dCdsi = dCdsi+dCdsi'-diag(diag(dCdsi)); 
        df(i,j,:) = dCdsi(tril(ones(d))==1);
    end; end
end
f = C;

function [f, df] = conT6(p, policy, m, s)                               % dMdp
policy.p = p;
if nargout < 2
    M = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp] = policy.fcn(policy, m, s);
    df = dMdp;
end
f = M;

function [f, df] = conT7(p, policy, m, s)
policy.p = p;
if nargout < 2
    [M, S] = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp] = policy.fcn(policy, m, s);
    df = dSdp;
end
f = S;

function [f, df] = conT8(p, policy, m, s)
policy.p = p;
if nargout < 2
    [M, S, C] = policy.fcn(policy, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] = ...
                                                      policy.fcn(policy, m, s);
    df = dCdp;
end
f = C;
