try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
catch
end
load swingupDMK3Expf30_H30.mat
newanglenoise1 = 0.10; 
S0 = diag([1e-9 0.0125 newanglenoise1 0.08 1e-9 0.0125 ...
    newanglenoise1 0.08 1e-9 0.0125 newanglenoise1 0.08].^2);
S0(2,6)=(0.012499)^2; S0(6,2)=S0(2,6); %0.0671 on dx
S0(6,10)=(0.012499)^2; S0(10,6)=S0(6,10);
S0(2,10) = S0(2,6)/S0(2,2) * S0(2,6);
S0(10,2) = S0(2,10);

%angle of inner pendulum
S0(3,7)=(newanglenoise1)^2; S0(7,3)=S0(3,7); % 0.16 on dtheta
S0(7,11)=(newanglenoise1)^2; S0(11,7)=S0(7,11);
S0(3,11) = S0(3,7)/S0(3,3) * S0(3,7);
S0(11,3) = S0(3,11);

%angle of outer pendulum
S0(4,8)=(0.08)^2; S0(8,4)=S0(4,8); % 0.16 on dtheta
S0(8,12)=(0.08)^2; S0(12,8)=S0(8,12);
S0(4,12) = S0(4,8)/S0(4,4) * S0(4,8);
S0(12,4) = S0(4,12);
cost=Cost(D);
ctrl.policy.p.w=ctrl.policy.p.w*0;
ctrl.policy.p.b=ctrl.policy.p.b*0;
s.s=S0;
for i=1:6
    H=5*i;
    learnPolicy
end
setRunTrialDMK3(ctrl);
varUnits={'pixels','rad','rad'};
drawRealExpD;