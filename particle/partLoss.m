function [L dLdx sL] = partLoss(cost, x)

[D Nsamp] = size(x);
Lsamp = zeros(1,Nsamp); dLdx = zeros(D,Nsamp);

parfor i = 1:Nsamp
    [Lsamp(i) dLdx(:,i)] = cost.fcn(x(:,i),zeros(D),cost);
end

L = mean(Lsamp);
dLdx = dLdx/Nsamp;
sL = std(Lsamp);