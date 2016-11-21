%% settings_cp.m
% *Summary:* Script set up the cart-pole scenario
%
% Copyright (C) 2008-2014 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2014-02-03
%
%% High-Level Steps
% # Define state and important indices
% # Set up scenario
% # Set up the plant structure
% # Set up the policy structure
% # Set up the cost structure
% # Set up the GP dynamics model structure
% # Parameters for policy optimization
% # Plotting verbosity
% # Some array initializations

%% Code
varNames = {'x','dx','dangle','angle'};
rng(1); format short; format compact; 
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end

% 1. Define state and important indices

% 1a. Full state representation (including all augmentations)
%
%  1  x          cart position
%  2  v          cart velocity
%  3  dtheta     angular velocity
%  4  theta      angle of the pendulum
%  5  sin(theta) complex representation ...
%  6  cos(theta) of theta
%  7  u          force applied to cart
%

% 1b. Important indices
% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for the inputs to the policy

odei = [1 2 3 4];            % varibles for the ode solver
augi = [];                   % variables to be augmented
dyno = [1 2 3 4];            % variables to be predicted (and known to loss)
angi = 4;                    % angle variables
dyni = [1 2 3 4 5 6];        % variables that serve as inputs to the dynamics GP
poli = [1 2 3 5 6];          % variables that serve as inputs to the policy

% 2. Set up the scenario
dt = 0.10;                                             % [s] sampling time
T = 5.0;                             % [s] initial prediction horizon time
H = ceil(T/dt);                  % prediction steps (optimization horizon)
mu0 = [0 0 0 0]';                                     % initial state mean
S0 = diag([0.1 0.1 0.1 0.1].^2);                  % initial state variance
N = 15;                                  % number controller optimizations
J = 1;                                % initial J trajectories of length H
K = 1;                    % number of initial states for which we optimize
nc = 100;                           % number of controller basis functions
So = [0.01 0.01 pi/180 pi/180].^2; % measurement noise levels, 1cm, 1 degree


% 3. Plant structure
plant.ode = @dynamics_cp;                             % dynamics ode function
plant.noise = diag(So);                               % measurement noise
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);    % ctrl implemented as zero order hold
plant.odei = odei;
plant.augi = augi;
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;

% 4. Policy structure
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s); % controller 
policy.maxU = 10;                                          % max. amplitude of 
                                                           % control
[mm, ss, cc] = gTrig(mu0, S0, plant.angi);                 % represent angles 
mm = [mu0; mm]; cc = S0*cc; ss = [S0 cc; cc' ss];          % in complex plane
policy.p.inputs = gaussian(mm(poli), ss(poli,poli), nc)';  % init. location of 
                                                           % basis functions
policy.p.targets = 0.1*randn(nc, length(policy.maxU));     % init. policy targets 
                                                           % (close to zero)
policy.p.hyp.l = log([1 1 1 0.7 0.7])';                    % initialize policy
policy.p.hyp.s = log(1);                                   % hyper-parameters
policy.p.hyp.n = log(0.01);


% 5. Set up the cost structure
cost.fcn = @loss_cp;                       % cost function
cost.gamma = 1;                            % discount factor
cost.p = 0.5;                              % length of pendulum
cost.width = 0.25;                         % cost function width
cost.expl =  0.0;                          % exploration parameter (UCB)
cost.angle = plant.angi;                   % index of angle (for cost function)
cost.target = [0 0 0 pi]';                 % target state


% 6. Dynamics model structure
dynmodel.fcn = @gpBase;                % function for GP predictions
dynmodel.train = @train;               % function to train dynamics model
dynmodel.induce = zeros(300,0,1);      % shared inducing inputs (sparse GP)
trainOpt = [300 200];                  % max. no. of line searches 
                                       % when training the GP
                                       % trainOpt(1): full GP
                                       % trainOpt(2): sparse GP (FITC)
dynmodel.approxS = 0;                  % approx. output covariance matrix ?
% 6.1: GP prior mean function
dynmodel.trainMean = 0;                % keep the GP prior mean fct fixed 
                                       % during training
% only relevant if mean fct is fixed:
[dynmodel.hyp(1:length(dyno)).m] ...
  = deal(zeros(length(dyni)+length(policy.maxU), 1)); 
[dynmodel.hyp.b] = deal(0);
for i=1:length(dyno); dynmodel.hyp(i).m(i) = 1; end            % identity mean

% 7. Parameters for policy optimization
opt.length = 150;                        % max. number of line searches
opt.MFEPLS = 30;                         % max. number of function evaluations
                                         % per line search
opt.verbosity = 1;                       % verbosity: specifies how much 
                                         % information is displayed during
                                         % policy learning. Options: 0-3
% 8. Plotting verbosity
plotting.verbosity = 1;            % 0: no plots
                                   % 1: some plots
                                   % 2: all plots
                                         
% 9. Some initializations
x = []; y = [];
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);