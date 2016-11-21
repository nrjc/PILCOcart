function draw_rollout(j, J, data, H, dt, cost, S)

% Plot a trajectory of the cartPole
%
% (C) Copyright 2009-2014 by Marc Deisenroth, Andrew McHutchon and
%                                             Carl Edward Rasmussen, 2014-12-02

x0 = [data(J+j).state(1:end-1,:) data(J+j).action];
N = sum(arrayfun(@(a)size(a.state,1)-1,data(1:J+j)));

set(gcf,'DoubleBuffer','on');
title('Pendulum Simulation')

% Loop over states in trajectory
for r = 1:size(data(j).state,1)-1
  if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
  cost.t = r;
  if nargin > 6 && ~isempty(S)
    draw(x0(r,1), x0(r,2), x0(r,6), cost,  ...
      ['controlled trial # ' num2str(j) ', T=' num2str(H*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*N) ...
      ' sec'], S.M(:,r), S.S(:,:,r));
  else
    draw(x0(r,1), x0(r,2), x0(r,6), cost,  ...
      ['(random) trial # ' num2str(1) ', T=' num2str(H*dt) ' sec'], ...
      ['total experience (after this trial): ' num2str(dt*N) ...
      ' sec']);
  end
  pause(dt);
end
