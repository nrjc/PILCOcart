% Cart-pole experiment: Gaussian-RBF policy
%
% Copyright (C) by Carl Edward Rasmussen, Marc Deisenroth, Andrew McHutchon,
% and Joe Hall. 2013-11-07.
%clear all; close all;

% newsettings;
% basename = 'cartPole_';
% 
% % mu0, S0:       for interaction
% % mu0Sim, S0Sim: for internal simulation (partial state only)
% %mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
% %mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);
% mu0Sim = mu0; S0Sim = S0;
% j = 1;

folder = '../../../multi-link-inverted-pendulum-with-machine-learning/demoprogram/benchdata/hffast5/0';
while j<40
rmpath(folder);
folder = strcat('../../../multi-link-inverted-pendulum-with-machine-learning/demoprogram/benchdata/hffast5/', num2str(j));  
addpath(folder)
loadData2pc;
x = [x; xx];
y = [y; yy];
trainDynModel;
if j > 16
    opt.length = -400;
else
    opt.length = -200;
end
learnPolicy;
j = j+1;
saving = strcat('hffast5_', num2str(j), '.mat');
save(saving)
outputCtrl; 
runTrial;
pause(120);

end


