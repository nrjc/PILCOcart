function d = covFluct2T(dynmodel, m, s, deriv)

% test derivatives of the covFluct function, 2012-05-31

delta = 1e-6;
E = size(dynmodel.target,2);

switch deriv

case 'dCdm'
  for i = 1:E
    d(i) = checkgrad(@covFluctT0, m, delta, dynmodel, s, i);
    disp(['i = ' num2str(i) '/' num2str(E) ': d = ' num2str(d(i))]);
  end

  case 'dCds'
    for i = 1:E
      d(i) = checkgrad(@covFluctT1, s(tril(ones(length(s)))==1), ...
                                                        delta, dynmodel, m, i);
      disp(['i = ' num2str(i) '/' num2str(E) ': d = ' num2str(d(i))]);
    end
        
  otherwise
    fprintf('Unrecognised derivative, options are ''dCdm'' and ''dCds''\n');
end


function [f, df] = covFluctT0(m, dynmodel, s, i)
[a, b, c] = covFluct2(dynmodel, m, s);
f = a(i,i); df = squeeze(c(i,i,:));

function [f df] = covFluctT1(s, dynmodel, m, i)
d = length(m);
ss(tril(ones(d))==1) = s; ss = reshape(ss,d,d);
ss = ss + ss' - diag(diag(ss));
[C dynmodel, dCdm, dCds] = covFluct2(dynmodel, m, ss);
f = C(i,i); df = squeeze(dCds(i,i,:,:));
df = df + df' - diag(diag(df)); df = df(tril(ones(d))==1);
