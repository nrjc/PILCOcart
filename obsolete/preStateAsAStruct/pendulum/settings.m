% Default parameters for the pendulum.
%
%  1   1  dtheta      angular velocity of pendulum
%  2      theta       angle of pendulum
%  3   2  sin(theta)
%  4   3  cos(theta)
%  5      u           torque applied to pendulum
%
% Copyright (C) by Carl Edward Rasmussen and Marc Deisenroth 2012-09-28
% Edited by Joe Hall 2014-02-26

varNames = {'dtheta','theta'};
rng(1); format short; format compact; 
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
                               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss']);
catch
end

% 1. Parameter initialization
odei = [1 2];                                % varibles for the ode solver
augi = [];                                     % variables to be augmented
dyno = [1 2];              % variables to be predicted (and known to loss)
angi = [2];                                              % angle variables
dyni = [1 2 3 4];        % variables that serve as inputs to the dynamics GP
poli = [1 3 4];             % variables that serve as inputs to the policy

dt = 0.1;                                              % [s] sampling time
T = 5;                                               % [s] prediction time
H = ceil(T/dt);                  % prediction steps (optimization horizon)
mu0 = [0 0]';                                         % initial state mean
S0 = 0.01*eye(2);                                 % initial state variance
N = 15;                                   % number of policy optimizations
J = 3;                     % no. of inital training rollouts (of length H)
nc = 20;                        % number of basis functions for controller


sigma_a = 0.01/sqrt(10*dt);
sigma_v = 0.1/sqrt(10*dt);

% 2. Plant structure
plant.ode = @dynamics;                             % dynamics ode function
%plant.noise = diag([0.1^2 0.01^2]);
plant.noise = diag([sigma_v^2, sigma_a^2]);
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0); % ctrl implemented as zero order hold
plant.odei = odei;
plant.augi = augi;
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;        

% 3. Policy structure

ctrl.fcn = @ctrlNF;
ctrl.state = [];
ctrl.init = [];
ctrl.policy.maxU = 2;
ctrl.U = length(ctrl.policy.maxU);
ctrl.policy.fcn = @(policy,m,s)conCat(@conGaussd,@gSat,policy,m,s);
% for conlin
%ctrl.policy.p.w = zeros(ctrl.U,numel(poli));
%ctrl.policy.p.b = zeros(ctrl.U,1);
% for conGaussd
mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
ctrl.policy.p.ll = log([1 0.7 0.7])';
ctrl.policy.p.cen = gaussian(mm(poli), eye(length(poli)), nc)';
ctrl.policy.p.w = 0.1*randn(nc, length(ctrl.policy.maxU));

% 4. Cost function
cost.fcn = @loss2;                                          % cost function
cost.gamma = 1;                                          % discount factor
cost.p = 0.5;                                         % length of pendulum
cost.width = 0.3;                                    % cost function width
cost.expl = 0;                                  % exploration parameter
cost.angle = plant.angi;                         % angle variables in cost
cost.target = [0 pi]';                                      % target state

% 5. Dynamics model structure
dynmodel.fcn = @gpBase;                      % function for GP predictions
dynmodel.train = @train;                % function to train dynamics model
dynmodel.induce = zeros(300,0,2);             % non-shared inducing inputs
dynmodel.approxS = 0;                  % use full output covariance matrix
dynmodel.trainMean = 0;           % keep the GP mean fixed during training
[dynmodel.hyp(1:2).m] = deal(zeros(5,1)); [dynmodel.hyp.b] = deal(0);
for i=1:2; dynmodel.hyp(i).m(i) = 1; end                   % identity mean
trainOpt = [300 500];     % max. no. of line searches when training the GP
                                                    % trainOpt(1): full GP
                                           % trainOpt(2): sparse GP (FITC)

dynmodel.opt = [-500 -500];                                          
% 6. Parameters for policy optimization
opt.length = 50;                               % max. no. of line searches
opt.MFEPLS = 30;        % max. no. of function evaluations per line search
opt.verbosity = 3;           % specifies how much information is displayed

% 7. Some initializations
x = []; y = [];
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);