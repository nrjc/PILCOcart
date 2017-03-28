% cart-doube-pole experiment
%
%    dyno
%  1   1  oldu        old value of u
%  2   2  dx          Verlocity of cart
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
%
% Copyright (C) 2008-2015 by Marc Deisenroth and Carl Edward Rasmussen,
% Jonas Umlauft, Rowan McAllister 2015-07-20
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
catch
end
load CartDoubleSwingupRestart23_H40.mat
for j = 1:N
% trainDirect(dyn, data, dyni, plant.dyno, j<20);
  mu0 = [0 0 0 0 0 0 0]';   % initial state mean
  applyController;
dyn.train(data,dyni,plant.dyno);
  dyn.on = dyn.on';
  dyn.pn = dyn.pn';
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
	  'observation noise|process noise std|inducing targets', '%0.5f');

  learnPolicy;
  
  if pred(j).cost(end).m < 0.3
    H = H + 2;
  end
  animate(latent(j+J), data(j+J), dt, cost);
  disp(['controlled trial # ' num2str(j)]);
end