function [d dy dh] = lossT(deriv, cost, m, S, delta)
% checks the derivatives of the loss function
%
% 2012-06-26


% addpath ../base/
% addpath ../util;
if nargin == 0;
  D = 4;
  m = randn(4,1);
  S = randn(4); S = S*S';
  z = randn(4,1);
  W = randn(4); W = W*W';
  func = @lossQuad;
  deriv = 'dLdm';
end
D = length(m); if nargout < 5; delta = 1e-4; end

switch deriv

    case {'dLdm', 'dMdm'}
        [d dy dh] = checkgrad(@losstest01, m, delta, cost, S);

    case {'dLds', 'dMds'}
        [d dy dh] = checkgrad(@losstest02, S(tril(ones(D))==1), delta, cost, m);
      
    case 'dSdm'
        [d dy dh] = checkgrad(@losstest03, m, delta, cost, S);
        
    case 'dSds'
        [d dy dh] = checkgrad(@losstest04, S(tril(ones(D))==1), delta, cost, m);
        
    case {'dCdm', 'dVdm'}
        [d dy dh] = checkgrad(@losstest05, m, delta, cost, S);
        
    case {'dCds', 'dVds'}
        [d dy dh] = checkgrad(@losstest06, S(tril(ones(D))==1), delta, cost, m);

end

%%
function [f, df] = losstest01(m, cost, S)                   % dLdm

[L dLdm] = cost.fcn(cost, m, S);

f = L;
df = dLdm;


function [f, df] = losstest02(s, cost, m)                   % dLds

d = length(m);
ss(tril(ones(d))==1) = s; ss = reshape(ss,d,d); ss = ss + ss' - diag(diag(ss));

[L dLdm dLds] = cost.fcn(cost, m, ss);

f = L; df = dLds; df = 2*df-diag(diag(df)); df = df(tril(ones(d))==1);


function [f, df] = losstest03(m, cost, S)                   % dSdm

[L dLdm dLds S dSdm] = cost.fcn(cost, m, S);

f = S;
df = dSdm;


function [f, df] = losstest04(s, cost, m)                   % dSds

d = length(m);
ss(tril(ones(d))==1) = s; ss = reshape(ss,d,d); ss = ss + ss' - diag(diag(ss));

[L dLdm dLds S dSdm dSds] = cost.fcn(cost, m, ss);

f = S; df = dSds; df = 2*df-diag(diag(df)); df = df(tril(ones(d))==1);


function [f, df] = losstest05(m, cost, S)                   % dCdm

[L dLdm dLds S dSdm dSds C dCdm] = cost.fcn(cost, m, S);

f = C;
df = dCdm;


function [f, df] = losstest06(s, cost, m)                   % dCds

d = length(m);
ss(tril(ones(d))==1) = s; ss = reshape(ss,d,d); ss = ss + ss' - diag(diag(ss));

[L dLdm dLds S dSdm dSds C dCdm dCds] = cost.fcn(cost, m, ss);

f = C;
dCds = reshape(dCds,d,d,d); df = zeros(d,d*(d+1)/2);
    for i=1:d;
        dCdsi = squeeze(dCds(i,:,:)); dCdsi = dCdsi+dCdsi'-diag(diag(dCdsi)); 
        df(i,:) = dCdsi(tril(ones(d))==1);
    end;
