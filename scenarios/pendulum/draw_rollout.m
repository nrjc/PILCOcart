% Plot the most recent trajectory of the pendulum
%
% (C) Copyright 2009-2011 by Marc Deisenroth and Carl Rasmussen,
% 2011-05-04. 

% Edited by Joe Hall 2012-10-04.
% Edited by Jonas Umlauft 2014-04-16



% Framerate and sampling rate
fps = 30;
sr = 1/fps;

set(gcf,'DoubleBuffer','on');
title('Pendulum Simulation')

% Loop over states in trajectory
for r = 1:size(data(j).state,1)-1
  if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
  cost.t = r;
  if isempty(M{j})
     draw(plant,data(j).state(r,2), data(j).action(r,:), cost,  ...
      ['(random) trial # ' num2str(j) ', T=' num2str(H*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*H*j) ...
      ' sec'], ctrl.policy.maxU)
  else
      draw(plant,data(j+J).state(r,2), data(j+J).action(r,:), cost,  ...
      ['trial(rand + ctrl) # ' num2str(J) '+' num2str(j) ', T=' num2str(H*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*H*(j+J)) ...
      ' sec'], ctrl.policy.maxU, M{j}(:,r), Sigma{j}(:,:,r));
  end
  pause(dt);
end