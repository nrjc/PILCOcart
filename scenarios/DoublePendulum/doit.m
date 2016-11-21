% Pendubot experiment: Gaussian-RBF policy.
% Default parameters for the double pendulum.
%
%  1  1  dtheta1     angular velocity of inner pendulum
%  2  2  dtheta2     angular velocity of outer pendulum
%  3     theta1      angle inner pendulum
%  4     theta2      angle outer pendulum
%  5  3  sin(theta1)
%  6  4  cos(theta1)
%  7  5  sin(theta2)
%  8  6  cos(theta2)
%  9     u1          torque applied to the 1st joint
% 10     u2          torque applied to the 2nd joint
%
% Copyright (C) by Carl Edward Rasmussen, Marc Deisenroth, Joe Hall, 
% Jonas Umlauft, Rowan McAllister 2014-07-12

clear all; close all;
basename = 'doublepend_';
varNames = {'dtheta1','dtheta2','theta1','theta2'};
rng(1); format short; format compact; 
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
                              [rd 'gp/NIGP'],[rd 'control'],[rd 'loss']);
catch
end

% 1. Parameter initialization
odei = [1 2 3 4];                            % varibles for the ode solver
augi = [];                                     % variables to be augmented
dyno = [1 2 3 4];          % variables to be predicted (and known to loss)
angi = [3 4];                                            % angle variables
dyni = 1:8;            % variables that serve as inputs to the dynamics GP
poli = [1 2 5 6 7 8];       % variables that serve as inputs to the policy

dt = 0.1;                                              % [s] sampling time
T = 3.0;                                             % [s] prediction time
H = ceil(T/dt);                  % prediction steps (optimization horizon)
mu0 = [0 0 pi pi]';                                   % initial state mean
S0 = diag([0.1 0.1 0.01 0.01].^2);              % initial state covariance
N = 40;                                  % no. of controller optimizations
J = 1;                     % no. of inital training rollouts (of length H)
K = 1;                    % number of initial states for which we optimize
nc = 200;                                % size of controller training set

% 2. Plant structure
plant.ode = @dynamics;                             % dynamics ode function
plant.noise = diag(ones(1,4)*0.01.^2);                 % measurement noise
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0); % ctrl implemented as zero order hold
plant.odei = odei;            % indices to the varibles for the ode solver
plant.augi = augi;                        % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.noOldU = 0;
plant.noOldX = 0;
plant.prop = @propagated;

% 3. Policy structure
ctrl.fcn = @ctrlNF;
ctrl.policy.fcn =  @(policy,m,s)conCat(@conGaussd,@gSat,policy,m,s);
ctrl.policy.maxU = [2.0 2.0];                       % max. amplitude of control
ctrl.U = numel(ctrl.policy.maxU);
mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
ctrl.policy.p.cen = gaussian(mm(poli), eye(length(poli)), nc)';
ctrl.policy.p.w = 0.1*randn(nc, ctrl.U);
ctrl.policy.p.ll = log(repmat([1 1 0.7 0.7 0.7 0.7],[2 1]))';

% Construct state indexer
D = length(dyno); s.m = nan(D,1); s.s = nan(D);
[~,~,~,s] = ctrl.fcn(s, 'ResetFilter');
plant.ns = length(unwrap(s));
plant.np = length(unwrap(ctrl.policy.p));
plant.is = rewrap(s,1:plant.ns);         % state stuct of members' indexes

% 4. Cost function
cost.fcn = @loss;                                          % cost function
cost.gamma = 1;                                          % discount factor
cost.p = [0.5 0.5];                                   % lengths of pendula
cost.width = 0.5;                                    % cost function width
cost.expl = 0;                                            % no exploration
cost.angle = plant.angi;              % index of angle (for cost function)
cost.target = [0 0 0 0]';                                   % target state

% 5. Dynamics model structure
dynmodel.fcn = @gpBase;                      % function for GP predictions
dynmodel.train = @train;                % function to train dynamics model
dynmodel.induce = zeros(300,0,4);             % non-shared inducing inputs
dynmodel.approxS = 0;                  % use full output covariance matrix
dynmodel.trainMean = 0;           % keep the GP mean fixed during training
[dynmodel.hyp(1:length(dyno)).m] = deal(zeros(length(dyni)+ctrl.U, 1)); 
[dynmodel.hyp.b] = deal(0);
for i=1:length(dyno); dynmodel.hyp(i).m(i) = 1; end    
ctrl.dynmodel = dynmodel; ctrl.dynmodel.fcn = @gphd;
%trainOpt = [300 500];     % max. no. of line searches when training the GP
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


%% Initial Rollouts (apply random actions)
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);
for j = 1:J                                   % get the first observations
  [data(j), latent{j}, realCost{j}] = rollout(gaussian(mu0, S0), ...
    struct('policy',struct('maxU',ctrl.policy.maxU),'U',ctrl.U), H, plant, cost);
  disp([data(j).state [data(j).action;zeros(1,ctrl.U)]]);

  figure(5); clf; %draw_rollout;
end

% mu0, S0:       for interaction
% mu0Sim, S0Sim: for internal simulation (partial state only)
mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
s.m = mu0Sim(dyno); s.s = S0Sim(dyno,dyno);

%% Start model-based policy search
for j = 1:N
  trainDynModel;
  ctrl.on = [dynmodel.hyp.on];    % copy learnt observation noise to controller
  ctrl.dynmodel = dynmodel; ctrl.dynmodel.fcn = @gphd;
  learnPolicy;  
  applyController;
  if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5);
  draw_rollout;
  disp(['controlled trial # ' num2str(j)]);
end