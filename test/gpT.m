function [dd dy dh] = gpT(deriv, dynmodel, m, s, delta)

% test derivatives of the gp family of functions, 2012-06-26
% Edited by Rown McAllister 2014-11-17
%#ok<*ALIGN>

if nargin < 2
  nn = 1000; np = 100;
  D = 5; E = 4;
  m = randn(D,1);
  s = randn(D); s = s*s';
  s(end,:) = 0;
  s(:,end) = 0;
  
  dynmodel.fcn = @gpd;
  dynmodel.hyp = [randn(D+1,E); zeros(1,E)];
  dynmodel.inputs = randn(nn,D);
  dynmodel.target = randn(nn,E);
  dynmodel.induce = randn(np,D,E);
end

if nargin < 5; delta = 1e-4; end
D = length(m);                                                      % input size

switch deriv
  
  case 'dMdm'
      [dd dy dh] = checkgrad(@gpT0, m, delta, dynmodel, s);
 
  case 'dSdm'
      [dd dy dh] = checkgrad(@gpT1, m, delta, dynmodel, s);
 
  case 'dVdm'
      [dd dy dh] = checkgrad(@gpT2, m, delta, dynmodel, s);
 
  case 'dMds'
      [dd dy dh] = checkgrad(@gpT3, s(tril(ones(D))==1), delta, dynmodel, m);
    
  case 'dSds'
      [dd dy dh] = checkgrad(@gpT4, s(tril(ones(D))==1), delta, dynmodel, m);
 
  case 'dVds'
      [dd dy dh] = checkgrad(@gpT5, s(tril(ones(D))==1), delta, dynmodel, m);
 
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


function [f, df] = gpT0(m, dynmodel, s)                             % dMdm
if nargout == 1
  M = dynmodel.fcn(dynmodel, m, s);
else
  [M, S, V, dMdm] = dynmodel.fcn(dynmodel, m, s);
  df = dMdm;
end
f = M;

function [f, df] = gpT1(m, dynmodel, s)                             % dSdm
if nargout == 1
  [M, S] = gpd(dynmodel, m, s);
else
  [M, S, V, dMdm, dSdm] = gpd(dynmodel, m, s);
  df = dSdm;
end
f = S;

function [f, df] = gpT2(m, dynmodel, s)                             % dVdm
if nargout == 1
  [M, S, V] = dynmodel.fcn(dynmodel, m, s);
else
  [M, S, V, dMdm, dSdm, dVdm] = dynmodel.fcn(dynmodel, m, s);
  df = dVdm;
end
f = V;

function [f, df] = gpT3(s, dynmodel, m)                             % dMds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  M = dynmodel.fcn(dynmodel, m, s);
else
  [M, S, V, dMdm, dSdm, dVdm, dMds] = dynmodel.fcn(dynmodel, m, s);
  dd = length(M); dMds = reshape(dMds,dd,d,d); df = zeros(dd,d*(d+1)/2);
    for i=1:dd; 
        dMdsi(:,:) = dMds(i,:,:); dMdsi = dMdsi+dMdsi'-diag(diag(dMdsi));  
        df(i,:) = dMdsi(tril(ones(d))==1);
    end
end
f = M;

function [f, df] = gpT4(s, dynmodel, m)                             % dSds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  [M, S] = gpd(dynmodel, m, s);
else
    [M, S, C, dMdm, dSdm, dCdm, dMds, dSds] = gpd(dynmodel, m, s);
    dd = length(M); dSds = reshape(dSds,dd,dd,d,d); df = zeros(dd,dd,d*(d+1)/2);
    for i=1:dd; for j=1:dd                                      
        dSdsi(:,:) = dSds(i,j,:,:); dSdsi = dSdsi+dSdsi'-diag(diag(dSdsi)); 
        df(i,j,:) = dSdsi(tril(ones(d))==1);
    end; end
end
f = S;

function [f, df] = gpT5(s, dynmodel, m)                             % dVds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  [M, S, V] = dynmodel.fcn(dynmodel, m, s);
else
  [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = dynmodel.fcn(dynmodel, m, s);
    dd = length(M); dVds = reshape(dVds,d,dd,d,d); df = zeros(d,dd,d*(d+1)/2);
    for i=1:d; for j=1:dd
        dCdsi = squeeze(dVds(i,j,:,:)); dCdsi = dCdsi+dCdsi'-diag(diag(dCdsi)); 
        df(i,j,:) = dCdsi(tril(ones(d))==1);
    end; end
end
f = V;

function [f, df] = gpT6(p, dynmodel, m, s)                          % dMdp
dynmodel = rewrap(dynmodel, p);
if nargout == 1
  M = dynmodel.fcn(dynmodel, m, s);
else
  [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp] = ...
                                                dynmodel.fcn(dynmodel, m, s);
  df = dMdp;
end
f = M;

function [f, df] = gpT7(p, dynmodel, m, s)                          % dSdp
dynmodel = rewrap(dynmodel, p);
if nargout == 1
    [M, S] = dynmodel.fcn(dynmodel, m, s);
else
    [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp] = ...
                                                dynmodel.fcn(dynmodel, m, s);
    df = dSdp;
end
f = S;

function [f, df] = gpT8(p, dynmodel, m, s)
dynmodel = rewrap(dynmodel, p);
if nargout == 1
    [M, S, V] = dynmodel.fcn(dynmodel, m, s);
else
    [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdp, dSdp, dVdp] = ...
                                                dynmodel.fcn(dynmodel, m, s);
    df = dVdp;

end
f = V;
