% Plot the most recent trajectory of the pendulum
%
% (C) Copyright 2009-2011 by Marc Deisenroth and Carl Rasmussen,
% 2011-05-04. Edited by Joe Hall 2012-10-04.

% Horizons and no. of rollouts with horizon
H0 = H;
R0 = 1;

% Framerate and sampling rate
fps = 30;
sr = 1/fps;

x0 = xx;

set(gcf,'DoubleBuffer','on');
title('Pendulum Simulation')
indexList = dt.*(0:size(x0,1)-1);

% Loop over states in trajectory
for r = 1:size(x0,1)
  cost.t = r;
  if exist('j','var') && ~isempty(M{j})
    draw(x0(r,2), x0(r,end), cost,  ...
      ['trial # ' num2str(j+J) ', T=' num2str(H0*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*size(x,1)) ...
      ' sec'], M{j}(:,r), Sigma{j}(:,:,r));
  else
     draw(x0(r,2), x0(r,end), cost,  ...
      ['(random) trial # ' num2str(1) ', T=' num2str(H0*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*size(x,1)) ...
      ' sec'])
  end
  pause(dt);
end