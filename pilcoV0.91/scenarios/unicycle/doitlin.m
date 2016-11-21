%% settings_unicycle.m
% *Summary:* Script set up the unicycle scenario
%
% Copyright (C) 2008-2014 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modified: 2014-02-13
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
rng(1); format short; format compact; 
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'gp'],[rd 'control'],[rd 'loss']);
catch
end 

% 1. Define state and important indices

% 1a. Full state representation (including all augmentations)

%  1      dx      x velocity
%  2      dy      y velocity
%  3      dxc     x velocity of origin (self centered coordinates)
%  4      dyc     y velocity of origin (self centered coordinates)
%  5   1  dtheta  roll angular velocity
%  6   2  dphi    yaw angular velocity
%  7   3  dpsiw   wheel angular velocity
%  8   4  dpsif   pitch angular velocity
%  9   5  dpsit   turn table angular velocity
% 10      x       x position
% 11      y       y position
% 12   6  xc      x position of origin (self centered coordinates)
% 13   7  yc      y position of origin (self centered coordinates)
% 14   8  theta   roll angle
% 15   9  phi     yaw angle
% 16      psiw    wheel angle
% 17  10  psif    pitch angle
% 18      psit    turn table angle
% 19      ct      control torque for turn table
% 20      cw      control torque for wheel
varNames = {'dx','dy','dxc','dyc','droll','dyaw','dwheel','dpitch',...
    'dflywheel','x','y','xc','yc','roll','yaw','wheel','pitch','flywheel'};

% 1b. Important indices
% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for the inputs to the policy

odei = [5 6 7 8 9 10 11 14 15 16 17 18];
augi = [1 2 3 4 12 13];
dyno = [5 6 7 8 9 12 13 14 15 17];
angi = [];
dyni = [1 2 3 4 5 6 7 8 9 10];
poli = [1 2 3 4 5 6 7 8 9 10];
varNames = varNames(dyno);

% 2. Set up the scenario
dt = 0.15;                                                  % [s] sampling time
T = 10.0;                                 % [s] initial prediction horizon time
H = ceil(T/dt);                       % prediction steps (optimization horizon)
maxH = ceil(10.0/dt);                                        % max pred horizon
s = [0.02 0.02 0.02 0.02 0.02 0.1 0.1 0.02 0.02 0.02 0.02 0.02].^2;
S0 = diag(s);                 % initial state variance, 95% is +/- 11.4 degrees
mu0 = zeros(length(odei),1);                               % initial state mean
N = 40;                                       % number controller optimizations
J = 10;                                    % initial J trajectories of length H
K = 1;                         % number of initial states for which we optimize

% 3. Plant structure
plant.ode = @dynamics;                                  % dynamics ode function
plant.augment = @augment;         % function to augment the state ode variables
plant.constraint = @(x)(abs(x(14))>pi/2 | abs(x(17))>pi/2);    % ode constraint
%plant.noise = diag([pi/180/0.15*ones(1,5) 0.01 0.01 pi/180*ones(1,5)].^2); % Realisitic noise 
plant.noise = diag([0.01 0.03*ones(1,4) 0.003*ones(1,7)].^2); % Low noise
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);    % ctrl implemented as zero order hold
plant.odei = odei;                 % indices to the varibles for the ode solver
plant.augi = augi;                             % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;

% 4. Policy structure
policy.p.w = 1e-2*randn(2,length(poli));
policy.p.b = zeros(2,1);
policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);
policy.maxU = [10 50];

% 5. Set up the cost structure
cost.fcn = @loss;                                               % cost function
cost.gamma = 1;                                               % discount factor
cost.p = [0.22 0.81];                                                % rw and r
cost.width = 1;                                           % cost function width
cost.expl = 0;                                                 % no exploration

% 6. Dynamics model structure
dynmodel.fcn = @gpBase;                           % function for GP predictions
dynmodel.train = @train;                     % function to train dynamics model
dynmodel.induce = zeros(300,0,10);                 % non-shared inducing inputs
trainOpt = [-500 -500];                         % options for dynmodel training
dynmodel.approxS = 0;                       % approx. output covariance matrix?

% 6.1: GP prior mean function
dynmodel.trainMean = 0;                      % keep the GP prior mean fct fixed
[dynmodel.hyp(1:length(dyno)).m] ...
                        = deal(zeros(length(dyni)+length(policy.maxU),1)); 
[dynmodel.hyp.b] = deal(0); 
for i=1:length(dyno); dynmodel.hyp(i).m(i) = 1; end             % identity mean

opt.length = -100; opt.verbosity = 3;          % options for minimize for policy


x = []; y = [];                                  % initialize various variables
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);

for j = 1:J                                        % get the first observations
  [xx, yy, realCost{j}, latent{j}] = ...
      rollout(gaussian(mu0, S0), struct('maxU',policy.maxU/5), H, plant, cost);
  disp(xx)
  x = [x; xx]; y = [y; yy];
end

z(odei,:) = bsxfun(@plus, mu0, chol(S0)'*randn(length(odei),1000));   % compute
for i = 1:size(z,2), z(augi,i) = plant.augment(z(:,i)'); end % the distribution
mu0Sim = mean(z,2); S0Sim = cov(z');         % of augmented start state by MCMC
mu0Sim(odei) = mu0; S0Sim(odei,odei) = S0;        % Put in known correct values   
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno); clear z i;

for j = 1:N
  trainDynModel;
  learnPolicy;  
  applyController;
  disp(['controlled trial # ' num2str(j)])
end
