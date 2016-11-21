function [L, sL] = calcCost(cost,S)
% Function to calculate the loss and the standard deviation of the loss
% given a rollout and the cost struct

H = size(S.M,2);
L = ones(1,H); SL = zeros(1,H);

for h = 1:H
  s.m = S.M(:,h); s.s = S.S(:,:,h);
  if any(any(isnan(s.s))) || max(eig(s.s)) > 1e10; break; end
  [L(h),SL(h)] = cost.fcn(cost,s);
end

sL = sqrt(SL);