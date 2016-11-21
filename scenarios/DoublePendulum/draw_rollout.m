% Plot the most recent trajectory of the pendubot.
%
% (C) Copyright 2009-2011 by Marc Deisenroth and Carl Rasmussen,
% 2011-05-04. Edited by Joe Hall 2012-10-04, Edited by Jonas Umlauft 2014-06-30

% Horizons and no. of rollouts with horizon
H0 = H;
R0 = 1;

% Framerate and sampling rate
fps = 30;
sr = 1/fps;

set(gcf,'DoubleBuffer','on');
title('Double Pendulum Simulation')
indexList = dt.*(0:size(data(j).state,1)-1);

% Loop over states in trajectory
for r = 1:size(data(j).state,1)-1
  cost.t = r;
  if exist('j','var') && ~isempty(M{j})
    draw(data(j).state(r,3),data(j).state(r,4), data(j).action(r,1), data(j).action(r,2), cost,  ...
      ['trial # ' num2str(j+J) ', T=' num2str(H0*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*size(x,1)) ...
      ' sec'], M{j}(:,r), Sigma{j}(:,:,r));
  else
    draw(data(j).state(r,3),data(j).state(r,4), data(j).action(r,1), data(j).action(r,2), cost,  ...
      ['(random) trial # ' num2str(1) ', T=' num2str(H0*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*size(x,1)) ...
      ' sec'])
  end
  pause(dt);
end
  