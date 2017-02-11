% Default parameters for the cartPole.
%
%  1  1  x          cart position
%  2  2  v          cart velocity
%  3  3  dtheta     angular velocity
%  4  4  theta      angle of the pendulum
%  5     sin(theta) 
%  6     cos(theta)
%  7     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Marc Deisenroth, Andrew McHutchon,
% and Joe Hall. 2013-11-07.
   varNames = {'x','px', 'pangle_1', 'p_angle_2', 'p u', 'pp u','angle_1', 'angle_2'};
%varNames = {'x', 'angle'};
rng(1); format short; format compact; 
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'],[rd 'GPNLDS']);
catch
end

% 1. Parameter initialization
odei = [1 2 3 4 5 6];                            % varibles for the ode solver
augi = [];                                     % variables to be augmented
dyno = [1 2 3 4 5 6 7 8];          % variables to be predicted (and known to loss)
angi = [7 8];                                                % angle variables
dyni = [1 2 3 4 5 6 7 8];  % variables that serve as inputs to the dynamics GP
poli = [1 2 3 4 7 8 9 10 11 12];         % variables that serve as inputs to the policy

dt = 0.0333;                                             % [s] sampling time
%T = 29.9545;                             % [s] initial prediction horizon time
H = 100; %ceil(T/dt);                  % prediction steps (optimization horizon)
mu0 = [0 0 pi pi 0 0 pi pi]';                                     % initial state mean
S0 = diag([0.05 0.05 0.01, 0.01 0 0 0.01 0.01].^2);                  % initial state variance
N = 15;                                  % number controller optimizations
J = 1;                                % initial J trajectories of length H
K = 1;                    % number of initial states for which we optimize
nc = 50;                           % number of controller basis functions

% 2. Plant structure
plant.dynamics = @dynamics;                        % dynamics ode function
plant.noise = diag(ones(1,8)*0.01.^2);                 % measurement noise
plant.dt = dt;
plant.ctrl = @zoh;                          % controler is zero order hold
plant.odei = odei;
plant.augi = augi;
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;

% 3. Policy structure
policyMappings = zeros(8, 10); % policyMappings*input = output
policyMappings(1, 1) = 1;
policyMappings(2, 1) = 1/dt;
policyMappings(2, 2) = -1/dt;
policyMappings(3, 5) = 1/dt;
policyMappings(3, 3) = -1/dt;
policyMappings(4, 6) = 1/dt;
policyMappings(4, 4) = -1/dt;
policyMappings(5, 7) = 1;
policyMappings(6, 8) = 1;
policyMappings(7, 9) = 1;
policyMappings(8, 10) = 1;


policy.copyMappings = policyMappings;
%policy.fcn = @(policy,m,s)conCat(@conGaussd,@gSat,policy,m,s);
policy.fcn = @(policy,m,s)conCat(@(policy, m, s)conMap(@conGaussd, @copyfcn, policy, m, s),@gSat,policy,m,s);
policy.maxU = 10;                              % max. amplitude of control
mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
mm = policyMappings*mm(poli);
policy.p.cen = gaussian(mm, eye(length(mm)), nc)';
policy.p.w = 0.08*randn(nc, length(policy.maxU));
policy.p.ll = log([1 1 1 1 0.7 0.7 0.7 0.7])';
% policy.regulate = @congpReg;

% set up the cost function
cost.fcn = @loss;                                           % cost function
cost.gamma = 1;                                             % discount factor
cost.p = [0.4 0.4];    %used to be [1 1]                                             % lenghts of the links
%cost.width = 0.3;                                             % cost function width
cost.width = [0.4 0.3 0.2 0.1];
cost.expl = 0;                                           % exploration parameter
cost.angle = plant.angi;                                    % angle variables in cost
cost.target = zeros(8,1);                                   % target state



% 5. Dynamics model structure
dynmodel.train = @trainMix;                % function to train dynamics model
[dynmodel.hyp(dyno).s] = deal(-1e1); 
[dynmodel.hyp(dyno).n] = deal(-1e1);
[dynmodel.hyp(dyno).on] = deal(-1e1);
[dynmodel.hyp(dyno).pn] = deal(-1e1);

% GP for predicting angle and position
dynmodel.sub{1}.train = @train;
dynmodel.sub{1}.fcn = @gpBase;                      % function for GP predictions
dynmodel.sub{1}.dyni = [1 2 3 4 5 6 7 8];
dynmodel.sub{1}.dyno = [1 7 8];
dynmodel.sub{1}.dynu = [1];
dynmodel.sub{1}.induce = zeros(600,0,3);             % non-shared inducing inputs
dynmodel.sub{1}.mean = 0;                % keep the GP mean fixed during training
[dynmodel.sub{1}.hyp(1:3).m] = deal(zeros(9,1)); [dynmodel.sub{1}.hyp.b] = deal(0); %v
%Set mean to linearly predict next position based on current and previous position
dynmodel.sub{1}.hyp(1).m(1) = 1;
dynmodel.sub{1}.hyp(2).m(7) = 1;
dynmodel.sub{1}.hyp(3).m(8) = 1;

% Part of dynmodel that appends previous states
copyMappings = zeros(5, 5); % copyMappings*input = ouput
copyMappings(1, 1) = 1;
copyMappings(2, 3) = 1;
copyMappings(3, 4) = 1;
copyMappings(4, 5) = 1;
copyMappings(5, 2) = 1;

dynmodel.sub{2}.copyMappings = copyMappings;
dynmodel.sub{2}.dyni = [1 5 7 8];
dynmodel.sub{2}.dyno = [2 3 4 5 6];
dynmodel.sub{2}.dynu = [1];
dynmodel.sub{2}.fcn = @copyfcn;
[dynmodel.sub{2}.hyp(1:5).pn] = deal(-1e1);

%dynmodel.mean = 0.01;                %Let GP mean vary during training
%[dynmodel.hyp(1:8).m] = deal(0.01*ones(11,1)); [dynmodel.hyp.b] = deal(0.01); %TODO: modify this thing to take into account multiple order markov
%for i=1:8; dynmodel.hyp(i).m(i) = 1; end                   % identity mean
%I think I could set the upper thing to 1:6 and it should work
%dynmodel.hyp(1).m(2) = dt; dynmodel.hyp(4).m(3) = dt;  % Euler integration
%dynmodel.hyp(1).m(2) = dt; dynmodel.hyp(4).m(3) = dt;  % Euler integration
trainOpt = [300 500];     % max. no. of line searches when training the GP
                                                    % trainOpt(1): full GP
                                           % trainOpt(2): sparse GP (FITC)

% 6. Parameters for policy optimization
opt.length = -150;                              % max. no. of line searches
opt.MFEPLS = 30;        % max. no. of function evaluations per line search
opt.verbosity = 3;           % specifies how much information is displayed
opt.method = 'BFGS';

% 7. Some initializations
x = []; y = [];
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);
