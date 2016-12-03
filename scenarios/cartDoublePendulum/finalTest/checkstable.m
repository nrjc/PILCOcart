%Initialize Workspace variables
clear all;
close all;
try
  rd = '../../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
[rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
  rd = '..';
  addpath([rd])
catch
end

%Initialize Results Matrix
result = NaN(5,6)
for errornum = 1:5
	for delaynum = 1:6
		name = ['CartDoubleStabilize' int2str(errornum) ...
		 'delay' int2str(delaynum) 'l15_H30']; 
		 try
			 load(name);
		catch
			continue;
		end
		result(errornum,delaynum) = 0;
		if (~checkBoundStable(stateS,stateM, 7)) %Check outer angle state bounded.
			result(errornum,delaynum) = result(errornum,delaynum) + 1;
		end
		result(errornum,delaynum) = result(errornum,delaynum) + checkrollout(stateS, stateM, lat, 7); 
	end
end
result