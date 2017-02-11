% Cart-pole experiment: Gaussian-RBF policy
%
% Copyright (C) by Carl Edward Rasmussen, Marc Deisenroth, Andrew McHutchon,
% and Joe Hall. 2013-11-07.
%clear all; close all;



newsettings;
basename = 'cartPole_';

% mu0, S0:       for interaction
% mu0Sim, S0Sim: for internal simulation (partial state only)
%mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
%mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);
mu0Sim = mu0; S0Sim = S0;

x = [x; xx];
y = [y; yy];

j=1;

%for j = 1:N
  trainDynModel;
  learnPolicy;
  saving = strcat('hffast4ag_', num2str(j), '.mat');
  save(saving)
  %applyController;
  %if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
  %draw_rollout;
  %disp(['controlled trial # ' num2str(j)]);
%end
 