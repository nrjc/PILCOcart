clear all; rng(3);
load('swingupDMK3Expe53_H20.mat','dyn', 'ctrl', 'H','pred');
nH=H;
ndyn=dyn;
nctrl=ctrl;
mypred=pred(53);
load('swingupDMK3Expe65_H40old_dyn.mat');

%merge dyns
tshift=19;
dyns(1)=dyn;
dyns(1).startT=1;
dyns(1).endT=tshift-1;
dyns(2)=ndyn;
dyns(2).startT=tshift-1;
dyns(2).endT=tshift+H; %+nH


%merge policies
newp=ctrl.policy.p;
for i=tshift:tshift+nH
	newp(i).w=nctrl.policy.p(i-tshift+1).w;
	newp(i).b=nctrl.policy.p(i-tshift+1).b;
end
newp(39).w = nctrl.policy.p(end).w; newp(39).b = nctrl.policy.p(end).b;
newp(40).w = nctrl.policy.p(end).w; newp(40).b = nctrl.policy.p(end).b;
ctrl.set_policy_p(newp);
j=66;
%keyboard;
pred(j)= valueS([], s, dyns, ctrl, cost, H);
pred(53)=mypred;

save('swingupDMK3Expe66_H40dyns2.mat');