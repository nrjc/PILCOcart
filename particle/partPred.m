function [M, U dynmodel] = partPred(policy, plant, dynmodel, m, s, H)

% predict rollout using particle approach
%
% Andrew McHutchon, 29/6/2011

if ~isfield(plant,'rStream'); 
    plant.rStream = RandStream.create('shr3cong','seed',plant.seed);
end
reset(plant.rStream);
oldx = []; oldy = []; Hc = plant.Hc;

E = size(dynmodel.target,2);
M = zeros(E,plant.Nsamp,H+1);
U = zeros(length(plant.maxU),plant.Nsamp,H+1);

for i = 1:H
  [X,u,minit,dynmodel,oldx,oldy] = partProp(m, s, plant, dynmodel, policy,oldx,oldy); % get next state
  if Hc > 0
      oldx = oldx(:,max(end-Hc+1,1):end,:);
      oldy = oldy(:,max(end-Hc+1,1):end,:);
  end

  if 1==i; M(:,:,i) = minit; end
  M(:,:,i+1) = X; U(:,:,i) = u;
 
  m = X; s = [];
end

