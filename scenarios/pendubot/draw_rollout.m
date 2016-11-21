function draw_rollout(plant,j,J,data,H,dt,cost,M,Sigma)

% Plot the most recent trajectory of the pendubot.
%
% (C) Copyright 2009-2011 by Marc Deisenroth and Carl Rasmussen,
% 2011-05-04. Edited by Joe Hall 2012-10-04.
% Edited by Rowan 2014-07-12

x0 = [data(J+j).state(1:end-1,:) data(J+j).action];
N = sum(arrayfun(@(a)size(a.state,1)-1,data(1:J+j)));

% Horizons and no. of rollouts with horizon
H0 = H;
R0 = 1;

% Framerate and sampling rate
fps = 30;
sr = 1/fps;

x0 = [data(J+j).state(1:end-1,:) data(J+j).action];

set(gcf,'DoubleBuffer','on');
title('Pendubot Simulation')
indexList = dt.*(0:size(x0,1)-1);

% Loop over states in trajectory
for r = 1:size(x0,1)
  cost.t = r;
  if exist('j','var') && ~isempty(M{j})
    draw(plant,x0(r,3), x0(r,4), x0(r,end), cost,  ...
      ['trial # ' num2str(j+J) ', T=' num2str(H0*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*N) ...
      ' sec'], M{j}(:,r), Sigma{j}(:,:,r));
  else
    draw(plant,x0(r,3), x0(r,4), x0(r,end), cost,  ...
      ['(random) trial # ' num2str(1) ', T=' num2str(H0*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*N) ...
      ' sec'])
  end
  pause(dt);
end