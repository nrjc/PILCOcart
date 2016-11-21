function [nll, dnll] = pCurbD(p,dyn,plant,data)

pwr = 30; maxsnr = 500;

[nll, dnll] = multiTrial(p,dyn,plant,data);

nX = size(data(1).state,2); nA = length(plant.angi); i = [plant.dyno plant.oldu nX-2*nA:nX];
xstd = std(cat(1,data.state),[],1);
xstd = [xstd(:,i(plant.dyni)) std(cat(1,data.action),[],1)];
llmls = bsxfun(@minus,[p.l],log(xstd')); e = exp(abs(llmls))/100;
nll = nll + sum(e(:).^pwr);

snr = exp([p.s] - [p.n]);
r = (snr/maxsnr).^pwr;
nll = nll + sum(r);

for i=1:length(dnll)
    dnll(i).l = dnll(i).l + pwr*e(:,i).^pwr.*sign(llmls(:,i));
    dnll(i).n = dnll(i).n - pwr*r(i);
    dnll(i).s = dnll(i).s + pwr*r(i);
end