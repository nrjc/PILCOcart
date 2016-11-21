function [L sL] = calcPartCost(cost,M)
% Function to calculate the loss and the standard deviation of the loss
% given a rollout and the cost struct

H = size(M,3);
L = zeros(1,H); sL = zeros(1,H); 

for h = 1:H
    [L(h),~,sL(h)]  = partLoss(cost,M(:,:,h));
end

    