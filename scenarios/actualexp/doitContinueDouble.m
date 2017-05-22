% Cart and pole swingup task.
%

%  1  1  ooou        the oldest value of u
%  2  2  oox         even older cart position
%  3  3  ootheta1    even older angle of the inner pendulum
%  4  4  ootheta2    even older angle of the outer pendulum
%  5  5  oou         even older value of u
%  6  6  ox          old cart position
%  7  7  otheta1     old angle of the inner pendulum
%  8  8  otheta2     old angle of the outer pendulum
%  9  9  ou          old value of u
%COMMENTED OUT:  10    dx          verlocity of cart
%COMMENTED OUT:  11    dtheta1     angular velocity of inner pendulum
%COMMENTED OUT:  12    dtheta2     angular velocity of outer pendulum
%  13 10 x           cart position
%  14 11 theta1      angle of the inner pendulum
%  14 12 theta2      angle of the outer pendulum
%  15    u           force applied to cart
%     13 sin(theta1)  ALL BELOW USED FOR POLICY TRAINING
%     14 cos(theta1)
%     15 sin(theta2)
%     16 cos(theta2)
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-02-17

close all
format short; format compact;

basename = 'swingupDMK3Expf';
fullname = [basename int2str(num) '_H30.mat'];

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
catch
end
load(fullname);
cost = Cost(D);
setRunTrialDMK3(ctrl)
getRunTrialDMK3
for j = num+1:169
  dyn.train(data,[1:12],[10:12]);
  %trainDirect(dyn, data, [1:12], [10:12], j<144);
  %disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
  %          ['observation noise|process noise std|inducing targets'], '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  learnPolicy;

  filename = [basename num2str(j) '_H' num2str(H)]; save(filename);
  setRunTrialDMK3(ctrl)
  getRunTrialDMK3
  disp(['controlled trial # ' num2str(j)]);
end


%trainDirectS(dyns, data, [1:12], [10:12], j<44);