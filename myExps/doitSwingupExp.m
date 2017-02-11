% Cart and pole swingup task.
%
%  1  1  oldu       old value of u
%  2  2  x          cart position
%  3  3  theta      angle of the pendulum
%  4  4  v          cart velocity
%  5  5  dtheta     angular velocity
%  6     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-01-18

clear
close all
format short; format compact;

basename = 'swingupExpzzzz';
varNames = {'x','theta','dx','dtheta'};

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'], [rd 'direct']);
catch
end
rng(2);

%mu0 = [0 0 pi 0 0]';                                       % initial state mean
%S0 = diag([1e-9 0.2 0.2 0.2 0.2].^2);                     % initial state variance
mu0 = [0 0 pi 0 0]';  
S0 = diag([1e-9 0.1 0.1 0.15 0.2].^2);
% mu0 = [0 0.0216 3.1424 0.0190 -0.036]';                       
% S0 = diag([1e-9 0.0325 0.0196 0.1015 0.2444].^2);
s = struct('m',mu0,'s',S0);

D = 5;
E = 4;
plant.odei = [2 3 4 5];
plant.dyno = [2 3 4 5];
% plant.augi = [];
% plant.oldu = [1];
poli = [1 2 4 5 6 7];
angi = [3];
% dt = 0.15; plant.dt = dt;                                 % time step is 150 ms
% plant.delay = 0.05;                  % with a delay in the contol loop of 50 ms
% H = ceil(5/plant.dt);
H = 60; %100;
% plant.noise = diag([0.02 pi/90 0.02 pi/90].^2);
% plant.noise = diag([0.05 pi/45 0.05 pi/45].^2);
% plant.noise = diag([0.05 pi/45 0.05/dt pi/45/dt].^2)
% plant.ctrltype = @(t,f,f0)zoh(t,f,f0);                % ctrl is zero order hold
% plant.ode = @dynamics;

N = 20; K = 1; J = 1; U = 1; maxU = 10;
cost.fcn = @loss;
cost.ell = 0.5; %1.0;
cost.gamma = 1.0;
cost.width = 0.25; %0.4;

fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);

dyn = gpa(6, 4, angi);      % 6 inputs, 4 outputs, and var number 2 is an angle
dyn.induce = zeros(50,0,1);                    % use 100 shared inducing inputs
dyn.opt = struct('length',-150,'verbosity',3,'method','BFGS','fh',6);

policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = maxU;                                 % max. amplitude of control
nc = 50;
mm = trigaug(mu0, zeros(5), angi);
policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
policy.p.target = 0.1*randn(nc, U);
policy.p.hyp = log([1 0.7 1 1 0.7 1 1 0.01])';
policy.opt = struct('length',-300,'MFEPLS',20,'verbosity',3);
ctrl = CtrlNF(D,E,policy,angi,poli);

j=0;
getRunTrial

%%j=2; %j-1
%%data(J+j).state=[];
%%data(J+j).action=[];

myCluster=parcluster('local'); myCluster.NumWorkers=10; parpool(myCluster,10);

%load('swingupExpj10_H100.mat');
%load('swingupExpp1_H100.mat');
%setRunTrial(ctrl)
%getRunTrial
for j = 1:N
  trainDirect(dyn, data, [1:5], [2:5], j<11);
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            'observation noise|process noise std|inducing targets', '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  learnPolicy;

  filename = [basename num2str(j) '_H' num2str(H)]; save(filename);
  setRunTrial(ctrl)
  getRunTrial
  

  disp(['controlled trial # ' num2str(j)]);
end


% plant.odei = [2 3 4 5];
% plant.dyno = [2 3 4 5];
% plant.augi = [];
% plant.oldu = [1];
% dt = 0.033; plant.dt = dt;
% plant.noise = diag([0.02 pi/90 0.02 pi/90].^2);
% plant.ctrltype = @(t,f,f0)zoh(t,f,f0); 
% plant.ode = @dynamics;
