% Edited by Rowan 2015-07-09
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

clear all; close all;
basename = 'uniRealNoise_';
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],...
    [rd 'gp'],[rd 'gp/NIGP'],[rd 'control'],[rd 'loss']);
catch
end
rng(3);
format short; format compact;

varNames = {'dx','dy','dxc','dyc','droll','dyaw','dwheel','dpitch',...
  'dflywheel','x','y','xc','yc','roll','yaw','wheel','pitch','flywheel'};

D = 10;
E = 10;
U = 2;

% Indices
angi = [];       % indicies for variables treated as angles (using sin/cos rep)
augi = [1 2 3 4 12 13]; % indicies for variables augmented to the ode variables
dyni = [1 2 3 4 5 6 7 8 9 10];      % indicies for inputs to the dynamics model
dyno = [5 6 7 8 9 12 13 14 15 17];  % indicies for dynmodel output + cost input
odei = [5 6 7 8 9 10 11 14 15 16 17 18];          % indicies for the ode solver
poli = [1 2 3 4 5 6 7 8 9 10];          % indicies for the inputs to the policy
varNames = varNames(dyno);

% Training parameters
dt = 0.15;                                                  % [s] sampling time
T = 10.0;                                 % [s] initial prediction horizon time
H = ceil(T/dt);                       % prediction steps (optimization horizon)
maxH = ceil(10.0/dt);                                        % max pred horizon
S0 = diag( ...                % initial state variance, 95% is +/- 11.4 degrees
  [0.02 0.02 0.02 0.02 0.02 0.1 0.1 0.02 0.02 0.02 0.02 0.02].^2);
mu0 = zeros(length(odei),1);                               % initial state mean
N = 40;                                       % number controller optimizations
J = 10;                                    % initial J trajectories of length H
K = 1;                         % number of initial states for which we optimize

% Plant structure
plant.ode = @dynamics;                                  % dynamics ode function
plant.augment = @augment;         % function to augment the state ode variables
plant.constraint = @(x)(abs(x(14))>pi/2 | abs(x(17))>pi/2);    % ode constraint
%plant.noise = diag([pi/180/0.15*ones(1,5) 0.01 0.01 pi/180*ones(1,5)].^2); % Realisitic noise
plant.noise = diag([pi/180/0.15*ones(1,5) 0.01 0.01 pi/180*ones(1,3)].^2); % Realisitic noise
% plant.noise = diag([0.01 0.03*ones(1,4) 0.003*ones(1,7)].^2); % Low noise
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);    % ctrl implemented as zero order hold
plant.odei = odei;                 % indices to the varibles for the ode solver
plant.augi = augi;                             % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;

% Policy structure
policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);
policy.p.w = 1e-2*randn(2,length(poli));
policy.p.b = zeros(2,1);
policy.maxU = [10 50];
policy.opt.length = -80;                      % options for minimize for policy
policy.opt.verbosity = 3;

% Dynamics model object
dyn = gpa(D+U, E, angi);
dyn.approxS = 0;
% dyn.train = @train;                        % function to train dynamics model
dyn.induce = zeros(300,0,10);                      % non-shared inducing inputs
[dyn.hyp(1:10).m] = deal(zeros(12,1)); [dyn.hyp.b] = deal(0);
for i=1:10; dyn.hyp(i).m(i) = 1; end                            % identity mean
dyn.hyp(8).m(1) = dt; dyn.hyp(9).m(2) = dt; dyn.hyp(10).m(4) = dt;
% dyn.trainMean = 0;                                      % keep the mean fixed
dyn.opt = [-500 -500];                          % options for dynmodel training

% Cost object
cost = Cost(D);

% initialize various variables
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);

setupFigures

%% Initial rollouts (apply random actions)
ctrlRand = Ctrl(D, E, struct('type','random','maxU',policy.maxU));
for j = 1:J                                   % get the first observations
  [data(j), latent{j}, realCost{j}] = ...
    rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost);
  disp([data(j).state [data(j).action;zeros(1,ctrl.U)]]);
  
  figure(5); clf; %draw;
end

z(odei,:) = bsxfun(@plus, mu0, chol(S0)'*randn(length(odei),1000));   % compute
for i = 1:size(z,2), z(augi,i) = plant.augment(z(:,i)'); end % the distribution
mu0Sim = mean(z,2); S0Sim = cov(z');         % of augmented start state by MCMC
mu0Sim(odei) = mu0; S0Sim(odei,odei) = S0;        % Put in known correct values
s.m = mu0Sim(dyno); s.s = S0Sim(dyno,dyno);
clear z i;

% Set up controller
ctrl = CtrlNF(D, E, policy, angi, poli);
s = ctrl.reset_filter(s);

%% Start model-based policy search
for j = 1:N
  trainDynModel;
  learnPolicy;
  applyController;
  disp(['controlled trial # ' num2str(j)])
end
