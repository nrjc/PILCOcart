% Cart and pole swingup task.
%

%  1  1  ooou        the oldest value of u
%  2  2  oox         even older cart position
%  3  3  ootheta     even old angle of the pendulum
%  4  4  oou        even older value of u
%  5  5  ox         old cart position
%  6  6  otheta     old angle of the pendulum
%  7  7  ou         old value of u
%  8  8  x          cart position
%  9  9  theta      angle of the pendulum
%  10     v          cart velocity
%  11     dtheta     angular velocity
%  12     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-02-17

clear
close all
format short; format compact;

basename = 'swingupMK3Expb';
varNames = {'x','theta'};

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'], [rd 'direct'], [rd 'test'], '');
catch
end
rng(3);

D = 9;
E = 2;

mu0 = [0 0 pi 0 0 pi 0 0 pi]';                                 % initial state mean

S0 = diag([1e-9 0.125 0.0071 1e-9 0.125 0.0071 1e-9 0.125 0.0071].^2);

S0(2,5)=(0.12499)^2; S0(5,2)=S0(2,5); %0.0671 on dx
S0(5,8)=(0.12499)^2; S0(8,5)=S0(5,8); 
S0(2,8) = S0(2,5)/S0(2,2) * S0(2,5);
S0(8,2) = S0(2,8);

S0(3,6)=(0.0060)^2; S0(6,3)=S0(3,6); % 0.16 on dtheta
S0(6,9)=(0.0060)^2; S0(9,6)=S0(6,9);
S0(3,9) = S0(3,6)/S0(3,3) * S0(3,6);
S0(9,3) = S0(3,9);

s = struct('m',mu0,'s',S0);

plant.odei = [8 9 10 11];
plant.dyno = [8 9];
plant.augi = [];

poli = [1 2 4 5 7 8 10 11 12 13 14 15];
angi = [3 6 9];
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
policy.p.hyp = log([1 1 1 1 1 1 0.7 0.7 0.7 0.7 0.7 0.7 1 0.01])';
policy.opt = struct('length',-300,'MFEPLS',20,'verbosity',3);

ctrl = CtrlNF(D,E,policy,angi,poli);

j=0;
getRunTrialMK3

for j = 1:20
  trainDirect(dyn, data, [1:9], [8 9], j<21);
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            ['observation noise|process noise std|inducing targets'], '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  learnPolicy;

  filename = [basename num2str(j) '_H' num2str(H)]; save(filename);
  setRunTrialMK3(ctrl)
  getRunTrialMK3

  disp(['controlled trial # ' num2str(j)]);
end
