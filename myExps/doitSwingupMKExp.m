% Cart and pole swingup task.
%
%  1  1  oou        even older value of u
%  2  2  ox         old cart position
%  3  3  otheta     old angle of the pendulum
%  4  4  ou         old value of u
%  5  5  x          cart position
%  6  6  theta      angle of the pendulum
%  7     v          cart velocity
%  8     dtheta     angular velocity
%  9     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-02-17

clear
close all
format short; format compact;

basename = 'swingupMKExps';
varNames = {'x','theta'};

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'], [rd 'direct'], [rd 'test'], '');
catch
end
rng(3);

D = 6;
E = 2;

mu0 = [0 0 pi 0 0 pi]';                                 % initial state mean
S0 = diag([1e-9 0.05 0.0071 1e-9 0.05 0.0071].^2);      % initial state variance
%S0(2,5)=(0.0499)^2; S0(5,2)=S0(2,5); % gives 0.134 stddev on dx
S0(2,5)=(0.04999)^2; S0(5,2)=S0(2,5); % gives 0.04 stddev on dx
S0(3,6)=(0.0060)^2; S0(6,3)=S0(3,6); % gives 0.16 stddev on dtheta

s = struct('m',mu0,'s',S0);

plant.odei = [5 6 7 8];
plant.dyno = [5 6];
plant.augi = [];

poli = [1 2 4 5 7 8 9 10];
angi = [3 6];
H=60;

N = 10; K = 1; J = 1; U = 1; maxU = 10;
cost.fcn = @lossMK;
cost.ell = 0.5;
cost.gamma = 1.0;
cost.width = 0.25;

fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);

dyn = gpa(D+U, E, angi);                                % D+U inputs, E outputs
dyn.induce = zeros(50,0,1);                    % use 100 shared inducing inputs
dyn.opt = struct('length',-300,'verbosity',3,'method','BFGS','fh',6);

policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = maxU;                                 % max. amplitude of control
nc = 50;
mm = trigaug(s.m, zeros(length(s.m)), angi);
policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
policy.p.target = 0.1*randn(nc, U);
policy.p.hyp = log([1 0.7 1 1 0.7 1 1 1 1 0.01])';
policy.opt = struct('length',-300,'MFEPLS',20,'verbosity',3);

ctrl = CtrlNF(D,E,policy,angi,poli);

j=0;
getRunTrialMK

for j = 1:20
  trainDirect(dyn, data, [1:6], [5 6], j<20);
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            ['observation noise|process noise std|inducing targets'], '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  learnPolicy;

  filename = [basename num2str(j) '_H' num2str(H)]; save(filename);
  setRunTrialMK(ctrl)
  getRunTrialMK

  disp(['controlled trial # ' num2str(j)]);
end
