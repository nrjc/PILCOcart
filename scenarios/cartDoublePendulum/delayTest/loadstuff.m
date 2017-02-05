% cart-doube-pole experiment
%
%    dyno
%  1   1  oldu        old value of u
%  2   2  dx          Verlocity of cat
%  3   3  dtheta1     angular velocity of inner pendulum
%  4   4  dtheta2     angular velocity of outer pendulum
%  5   5  x           Position of cart
%  6   6  theta1      angle of inner pendulum
%  7   7  theta2      angle of outer pendulum
%  8      u           Force on Cart
%  9   8  sin(theta1)
% 10   9  cos(theta1)
% 11  10  sin(theta2)
% 12  11  cos(theta2)
clear all;
close all;
check = true;
setdir;
L = 1;
check = iterateDir(1);
aveMatrix = zeros(5,6);
unstablePolicyParamStore = [];
stablePolicyParamStore = [];
while(check)
	L = L+1;
	checkstable;
	aveMatrix = aveMatrix + result;
	[row,col]=find(result==2);
	ctrl.poli
	
	for errornum = row'
		for delaynum = col'
			name = ['CartDoubleStabilize' int2str(errornum) ...
			 'delay' int2str(delaynum) 'l15_H30']; 
			 try
				 load(name);
			catch
				continue;
			end
			unstablePolicyParamStore = [unstablePolicyParamStore ctrl.policy.p];
		end
	end
	[row,col]=find(result==0);

	for errornum = row'
		for delaynum = col'
			name = ['CartDoubleStabilize' int2str(errornum) ...
			 'delay' int2str(delaynum) 'l15_H30']; 
			 try
				 load(name);
			catch
				continue;
			end
			stablePolicyParamStore = [stablePolicyParamStore ctrl.policy.p];
		end
	end
	check = iterateDir(L);
end
%Compute policy params
wStore = zeros(size(ctrl.policy.p.w));
for n=1:length(unstablePolicyParamStore)
	wStore = wStore + unstablePolicyParamStore(n).w;
end
unstableParams = wStore/length(unstablePolicyParamStore)
wStore = zeros(size(ctrl.policy.p.w));
for n=1:length(stablePolicyParamStore)
	wStore = wStore + stablePolicyParamStore(n).w;
end
stableParams = wStore/length(stablePolicyParamStore)
aveMatrix = aveMatrix / (L-1)

%plotall;