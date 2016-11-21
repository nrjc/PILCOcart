function [nlp, nlpdp] = multiTrial(p, dyn, data, dyni, dyno)

NT = length(data); Np = length(unwrap(p)); 
nlp = zeros(1,NT); nlpdp = zeros(Np,NT);

dyn.beta = [p.beta]; dyn.on = [p.on]; dyn.pn = [p.pn];
dyn.hyp = rmfield(p,{'beta','on','pn'});
dyn.idx = rewrap(p,1:Np);
[dyn, ddyn] = preComp(dyn); dyn.iKdl = ddyn.iKdll;

if nargout < 2
  parfor i = 1:NT; nlp(i) = GPf(p, dyn, data(i), dyni, dyno); end
else
  parfor i = 1:NT;
    [nlp(i), nlpdpi] = GPf(p, dyn, data(i), dyni, dyno);
    nlpdp(:,i) = unwrap(nlpdpi);
  end
end

nlp = sum(nlp); 
if nargout == 2, nlpdp = rewrap(p, sum(nlpdp,2)); end

pwr = 30; maxsnr = 2000;                     % set maximum signal to noise ratio
snr1 = exp([p.s] - [p.n ]); r1 = (snr1/maxsnr).^pwr; nlp = nlp + sum(r1);
snr2 = exp([p.s] - [p.on]); r2 = (snr2/maxsnr).^pwr; nlp = nlp + sum(r2);
snr3 = exp([p.s] - [p.pn]); r3 = (snr3/maxsnr).^pwr; nlp = nlp + sum(r3);
if nargout > 1
  for i=1:length(nlpdp)
    nlpdp(i).n  = nlpdp(i).n  - pwr*r1(i); nlpdp(i).s = nlpdp(i).s + pwr*r1(i);
    nlpdp(i).on = nlpdp(i).on - pwr*r2(i); nlpdp(i).s = nlpdp(i).s + pwr*r2(i);
    nlpdp(i).pn = nlpdp(i).pn - pwr*r3(i); nlpdp(i).s = nlpdp(i).s + pwr*r3(i);
  end
end
