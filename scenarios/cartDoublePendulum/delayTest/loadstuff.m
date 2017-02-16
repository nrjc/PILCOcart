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