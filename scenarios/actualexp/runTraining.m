try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
catch
end
load swingupDMK3Expf8_H30.mat
cost=Cost(D);
ctrl.policy.p.w=ctrl.policy.p.w*0;
ctrl.policy.p.b=ctrl.policy.p.b*0;

for i=1:4
    H=5*i;
    learnPolicy
end
drawRealExpD;