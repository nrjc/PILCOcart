function [M S] = predc(policy, plant, dynmodel, m, s, H)

% predictive (marginal) distributions of a trajecory
%
% 2011-12-06

E = length(m); S = cell(1,H+1); M = zeros(E,H+1);
M(:,1) = m; S{1} = s;
for i = 1:H
  [m s dynmodel] = plant.prop(m, s, plant, dynmodel, policy);
  M(:,i+1) = m(end-E+1:end); 
  S{i+1} = s(end-E+1:end,end-E+1:end);
end
