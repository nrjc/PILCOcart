% Example doit file.  Last edit, Marc on 2012-09-28, Rowan 2015-07-09
%
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

% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for the inputs to the policy
% difi  indicies for dynamics targets which are deltas (rather than values)

clear all; close all;
dbstop if error
basename = 'unicycle_';
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'gp/NIGP'],[rd 'control'],[rd 'loss']);
catch
end
rng(48);
format short; format compact;

D = 10;
E = 10;
U = 2;

% Indices
angi = [];
augi = [1 2 3 4 12 13];
difi = [1 2 3 4 5 6 7 8 9 10];
dyni = [1 2 3 4 5 6 7 8 10];
dyno = [5 6 7 8 9 12 13 14 15 17];
odei = [5 6 7 8 9 10 11 14 15 16 17 18];
poli = [1 2 3 4 5 6 7 8 9 10];

% Training parameters
dt = 0.1;                                                   % [s] sampling time
T = 2.0;                                  % [s] initial prediction horizon time
H = ceil(T/dt);                       % prediction steps (optimization horizon)
maxH = ceil(10.0/dt);                                        % max pred horizon
S0 = diag( ...                % initial state variance, 95% is +/- 11.4 degrees
  [0.02 0.02 0.02 0.02 0.02 0.1 0.1 0.02 0.02 0.02 0.02 0.02].^2);
mu0 = zeros(length(odei),1);                               % initial state mean
N = 40;                                       % number controller optimizations
J = 10;                                    % initial J trajectories of length H
K = 1;                         % number of initial states for which we optimize

% Plant structure
plant.dynamics = @dynamics;                             % dynamics ode function
plant.augment = @augment;         % function to augment the state ode variables
plant.constraint = inline('abs(x(8))>pi/2 | abs(x(11))>pi/2'); % ode constraint
%plant.noise = 0.01*diag(ones(1,12)*(pi/180).^2);           % measurement noise
%plant.noise = diag([0.01, 0.03*ones(1,4), 0.003*ones(1,7)].^2);
plant.noise = diag([0.01, 0.03*ones(1,4), 0.003*ones(1,5)].^2);
plant.dt = dt;
plant.ctrl = @lag;                      % control application (zoh, foh or lag)
plant.tau = 0.01;
plant.delay = 0.3*dt;
plant.odei = odei;                 % indices to the varibles for the ode solver
plant.augi = augi;                             % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.difi = difi;

% Policy
policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);
policy.p.w = 1e-2*randn(2,length(poli));
policy.p.b = zeros(2,1);
policy.maxU = [10 50];
policy.opt.length = -60;                      % options for minimize for policy
policy.opt.verbosity = 3;
global currT;

% initialize various variables
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);

% Dynamics model
dyn = gpa(D+U, E, angi);
dyn.induce = zeros(300,0,10);                      % non-shared inducing inputs
% dyn.opt = [50 25];                            % options for dynmodel training
% dyn.train = @trainMix;
% dyn.sub{1}.fcn = @gpi1d;
% dyn.sub{1}.train = @train;                 % function to train dynamics model
% dyn.sub{1}.induce = zeros(300,0,10);             % non-shared inducing inputs
% dyn.sub{1}.dyni = 1:length(dyni);
% dyn.sub{1}.dynu = 1:2;
% dyn.sub{1}.dyno = 1:length(dyno);

[mu0, S0, dyn, plant] = ctrlaugment(mu0, S0, dyn, plant, 2);

% Cost object
cost = Cost(D);

setupFigures

%% Initial rollouts (apply random actions)
ctrlRand = Ctrl(D, E, struct('type','random','maxU',policy.maxU));
for j = 1:J                                        % get the first observations
  [data(j), latent{j}, realCost{j}] = ...
    rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost);
  disp([data(j).state [data(j).action; zeros(1,ctrl.U)]]); disp(' ');
  animate(latent, data, dt, cost);
end

z(odei,:) = bsxfun(@plus, mu0, chol(S0)'*randn(length(odei),1000));   % compute
for i = 1:size(z,2), z(augi,i) = plant.augment(z(:,i)'); end % the distribution
mu0Sim = mean(z,2); S0Sim = cov(z');         % of augmented start state by MCMC
mu0Sim(odei) = mu0; S0Sim(odei,odei) = S0;        % Put in known correct values
s.m = mu0Sim(dyno); s.s = S0Sim(dyno,dyno);
clear z i;

% Controller structure
ctrl = CtrlNF(D, E, policy, angi, poli);
s = ctrl.reset_filter(s);

%% Start model-based policy search
for j = 1:N
  trainDynModel;
  learnPolicy;
  applyController;
  disp(['controlled trial # ' num2str(j)])
end
