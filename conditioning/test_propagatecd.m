function d = test_propagatecd(m, S, plant, dynmodel, policy, deriv)
% state is [Dold (=Di+Du); Do]

delta = 1e-6;

D = length(m); Du = length(plant.maxU); Do = size(dynmodel.target,2);
Dold = D - Do; k = Dold+1:D;
dynmodel = preCalcDyn(dynmodel); % prep dynmodel

switch deriv

  case 'dMdm'
    for i = k
      d(i) = checkgrad(@proptest01, m, delta, S, plant, dynmodel, policy, i);
      disp(['i = ' num2str(i) ': d = ' num2str(d(i))]);
    end

  case 'dSdm'
    for i = k
       for j = k
        d(i,j) = checkgrad(@proptest02, m, delta, S, plant, dynmodel, policy, i, j);
        disp(['i = ' num2str(i) ', j = ' num2str(j) ': d = ' num2str(d(i,j))]);
       end
    end

  case 'dMds'
    for i = k
      d(i) = checkgrad(@proptest03, S(tril(ones(length(S)))==1), delta, m, S, plant, dynmodel, policy, i);
      disp(['i = ' num2str(i) ': d = ' num2str(d(i))]);
    end
    
  case 'dSds'
    for i = 1:D
      for j = 1:D
        d(i,j) = checkgrad(@proptest04, S(tril(ones(length(S)))==1), delta, m, plant, dynmodel, policy, i, j);
        disp(['i = ' num2str(i) ', j = ' num2str(j) ': d = ' num2str(d(i,j))]);
      end
    end

  case 'dMdp'
    for i = k
      d(i) = checkgrad(@proptest05, policy, delta, m, S, plant, dynmodel, i);
      disp(['i = ' num2str(i) ': d = ' num2str(d(i))]);
    end

  case 'dSdp'
    for i = k
      for j = k
        d(i,j) = checkgrad(@proptest06, policy, delta, m, S, plant, dynmodel, i, j);
        disp(['i = ' num2str(i) ', j = ' num2str(j) ': d = ' num2str(d(i,j))]);
      end
    end
end

if size(d,1) == 1; d = d(k); else d = d(k,k); end


function [f df] = proptest01(m, S, plant, dynmodel, policy, i)
% dMdm
[m, S, dynmodel, dmdm] = propagatecd(m, S, plant, dynmodel, policy);
f = m(i);
df = dmdm(i,:)';

function [f df] = proptest02(m, S, plant, dynmodel, policy, i, j)
% dSdm
[m, S, dynmodel, dmdm, dsdm] = propagatecd(m, S, plant, dynmodel, policy);
f = S(i,j);
df = squeeze(dsdm(i,j,:));

function [f df] = proptest03(SS,m, plant, dynmodel, policy, i)
% dMds
d = length(m);
S(tril(ones(d))==1) = SS; S = reshape(S,d,d);
S = S + S' - diag(diag(S));
[m, S, dynmodel, dmdm, dsdm, dmds] = propagatecd(S, m, plant, dynmodel, policy);
f = M(i);
df = squeeze(dmds(i,:,:));
df = df + df' - diag(diag(df)); df = df(tril(ones(d))==1);

function [f df] = proptest04(SS,m,plant,dynmodel,policy, i, j)
% dSds
d = length(m);
S(tril(ones(d))==1) = SS; S = reshape(S,d,d);
S = S + S' - diag(diag(S));
[m, S, dynmodel, dmdm, dsdm, dmds, dsds] = propagatecd(m, S, plant, dynmodel, policy);
f = S(i,j);
df = squeeze(dsds(i,j,:,:));
df = df + df' - diag(diag(df)); df = df(tril(ones(d))==1);

function [f df] = proptest05(policy, m, S, plant, dynmodel, i)
% dMdp
[m, S, dynmodel, dmdm, dsdm, dmds, dsds, dmdp] = ...
                                      propagatecd(m, S, plant, dynmodel, policy);
f = m(i);
df = dmdp(i,:)';

function [f df] = proptest06(policy, m, S, plant, dynmodel, i, j)
% dMdp
[m, S, dynmodel, dmdm, dsdm, dmds, dsds, dmdp, dsdp] = ...
                                      propagatecd(m, S, plant, dynmodel, policy);
f = S(i,j);
df = squeeze(dsdp(i,j,:));

