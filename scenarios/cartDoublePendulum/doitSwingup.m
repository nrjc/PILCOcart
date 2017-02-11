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
clear; close all; clc
dbstop if error
basename = 'CartDoubleSwingup';

varNames = {'dx','dtheta1','dtheta2','x','theta1','theta2'};
varUnits = {'N','m/s','rad/s','rad/s','m','rad','rad'};
warning('on','all'); format short; format compact;
rng(2);

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
catch
end

D = 7;
E = 6;
U = 1;

% Indices
angi = [6 7];                 % indicies for vars treated as angles (using sin/cos rep)
augi = [];                    % indicies for vars augmented to the ode vars
dyni = [1 2 3 4 5 6 7];       % indicies for input into dynamics model
dyno = [2 3 4 5 6 7];         % indicies for output from the dynamics model and indicies to loss
odei = [2 3 4 5 6 7];         % indicies for ode solver
poli = [1 2 3 4 5 8 9 10 11]; % indicies for inputs to the policy

% Training parameters
dt = 1/20;                % [s] sampling time
plant.delay = 0.010;                 % with a delay in the contol loop of 10 ms
T = 3;                  % [s] horizon time
H = ceil(T/dt);           % prediction steps (optimization horizon)
S0 = diag([1e-4 0.1 0.01 0.01 0.1 0.1 0.1 ].^2); % initial state covariance
mu0 = [0 0 0 0 0 pi pi]';   % initial state mean
maxU = 20; 
N = 100;                   % number controller optimizations
J = 1;                    % J trajectories, each of length H for initial training
K = 1;                    % number of initial states for which we optimize

So = 0.25*[0.01/dt pi/180/dt pi/180/dt 0.01 pi/180 pi/180].^2; % noise levels, 1cm, 1 degree
s.m = mu0(1:D); s.s = S0(1:D,1:D);                              % initial state

% Plant structure
plant.angi = angi;
plant.constraint = @(x)abs(x(5))>4;           % absolute position less than 4 m
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);    % ctrl implemented as zero order hold
plant.dt = dt;
plant.ode = @dynamics;                                  % dynamics ode function
plant.poli = poli;
plant.noise = diag(So);                                     % measurement noise
plant.odei = odei;
plant.dyno = dyno;
plant.augi = augi;

% Policy
policy.fcn = @(policy,m,s)conCat(@conSeqLin,@gSat,policy,m,s);
policy.maxU = maxU;                      % max. amplitude of control
policy.p=struct([]);
for i=1:H
  policy.p(i).w = 0*randn(U, numel(poli));
  policy.p(i).b = 0*randn(U, 1);
end
policy.opt = ...
        struct('length',-30,'method','BFGS','MFEPLS',20,'verbosity',3,'fh',1);
global currT;
% policy.fcn = @(policy,m,s)conCat(@congp,@gSat7,policy,m,s);
% policy.maxU = maxU;                      % max. amplitude of control
% mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
% policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
% policy.p.target = 0.1*randn(nc, U);
% policy.p.hyp = zeros(length(poli)+2,1);
% policy.opt = ...
%       struct('length',-300,'method','BFGS','MFEPLS',20,'verbosity',3,'fh',1);

% Dynamics model object
dyn = gpa(D+U, E, angi, 'vfe'); % 8 inputs, 6 outputs, and var number 2, 3 are angles
dyn.induce = zeros(300, 0, E);                % use 100 shared inducing inputs
dyn.opt = ...
        struct('length',-300,'method','BFGS','MFEPLS',20,'verbosity',3,'fh',6);

% Cost object
cost = Cost(D);

% Set up figures
setupFigures

%% Initial Rollouts (apply random actions)
ctrlRand = Ctrl(D, E, struct('type','random','maxU',maxU));
for j = 1:J                                        % get the first observations
  [data(j), latent(j), realCost{j}] = ...
    rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost); %#ok<SAGROW>
  disp([data(j).state [data(j).action; zeros(1,U)]]);
  animate(latent, data, dt, cost);
end

% Set up controller
ctrl = CtrlNF(D, E, policy, angi, poli);
s = ctrl.reset_filter(s);

%% Start model-based policy search
for j = 1:N
% trainDirect(dyn, data, dyni, plant.dyno, j<20);
dyn.train(data,dyni,plant.dyno);
  dyn.on = dyn.on';
  dyn.pn = dyn.pn';
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
	  'observation noise|process noise std|inducing targets', '%0.5f');

  learnPolicy;
  
  if pred(j).cost(end).m < 0.3
    policy.p = ctrl.policy.p;
    for i = H+1:H+5
      policy.p(i).w = 0*randn(U, numel(poli));
      policy.p(i).b = 0*randn(U, 1);
    end
    ctrl.set_policy_p(policy.p);
    H = H + 5;
  end
  
  applyController;
  animate(latent(j+J), data(j+J), dt, cost);
  disp(['controlled trial # ' num2str(j)]);
end
