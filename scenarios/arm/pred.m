function [M S] = pred(ctrl, plant, dynmodel, m, s, H)

% predictive (marginal) distributions of a trajecory, 2014-04-02

E = length(m); S = zeros(E,E,H+1); M = zeros(E,H+1);
M(:,1) = m; S(:,:,1) = s;
for i = 1:H
  ctrl.policy.t = i;
  [ctrl.state, m, s] = plant.prop(m, s, plant, dynmodel, ctrl);
  M(:,i+1) = m(end-E+1:end); 
  S(:,:,i+1) = s(end-E+1:end,end-E+1:end);
end
