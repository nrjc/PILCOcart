% Example doit file. Last edit, Rowan on 2015-10-22
%
%  1   1  dtheta  roll angular velocity
%  2   2  dphi    yaw angular velocity
%  3   3  dpsiw   wheel angular velocity
%  4   4  dpsif   pitch angular velocity
%  5   5  dpsit   turn table angular velocity
%  6   6  xc      x position of origin (self centered coordinates)
%  7   7  yc      y position of origin (self centered coordinates)
%  8   8  theta   roll angle
%  9   9  phi     yaw angle
% 10  10  psif    pitch angle
% 11      dx      x velocity
% 12      dy      y velocity
% 13      dxc     x velocity of origin (self centered coordinates)
% 14      dyc     y velocity of origin (self centered coordinates)
% 15      x       x position
% 16      y       y position
% 17      psiw    wheel angle
% 18      psit    turn table angle
% 19      ct      control torque for turn table
% 20      cw      control torque for wheel

clear all; close all;
basename = 'uninlds_';
run ../../util/addpaths
rng(3);
format short; format compact;

varNames = {'droll','dyaw','dwheel','dpitch','dflywheel','xc','yc','roll', ...
  'yaw','pitch','dx','dy','dxc','dyc','x','y','wheel','flywheel'};

D = 10;
E = 10;
U = 2;

% Indices
angi = [];       % indicies for variables treated as angles (using sin/cos rep)
augi = [11 12 13 14 6 7];   % indicies for variables augmented to ode variables
dyno = D-E+1:D; % indicies for the output from dynamics model and input to loss
odei = [1 2 3 4 5 15 16 8 9 17 10 18];            % indicies for the ode solver
poli = 1:D;                             % indicies for the inputs to the policy
varNames = varNames(1:D);

% Training parameters
dt = 0.15;                                                  % [s] sampling time
T = 10.0;                                 % [s] initial prediction horizon time
H = ceil(T/dt);                       % prediction steps (optimization horizon)
maxH = ceil(T/dt);                                           % max pred horizon
S0 = (0.02*ones(1,18)).^2;    % initial state variance, 95% is +/- 11.4 degrees
S0([15,16]) = 0.1;
S0 = diag(S0);
mu0 = zeros(18,1);                                         % initial state mean
N = 40;                                       % number controller optimizations
J = 10;                                    % initial J trajectories of length H
K = 1;                         % number of initial states for which we optimize

% Plant structure
plant.ode = @dynamics;                                  % dynamics ode function
plant.augment = @augment;         % function to augment the state ode variables
plant.constraint = @(x)(abs(x(8))>pi/2 | abs(x(10))>pi/2);     % ode constraint
plant.noise = diag([pi/180/0.15*ones(1,5) 0.01 0.01 pi/180*ones(1,3)].^2);
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);           % controler is zero order hold
plant.odei = odei;                 % indices to the varibles for the ode solver
plant.augi = augi;                             % indices of augmented variables
plant.angi = angi;
plant.poli = poli;

% Policy structure
policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);
policy.maxU = [10 50];
policy.p.w = 1e-2*randn(U,length(poli));
policy.p.b = zeros(U,1);
policy.opt.length = -100;
policy.opt.verbosity = 3;
policy.opt.fh = 1;
policy.opt.MFEPLS = 10;
global currT;

% Dynamics model object
dyn = gpa(D+U, E, angi);
dyn.induce = zeros(50,0,1);                            % number inducing inputs
dyn.opt.length = 400;
dyn.opt.verbosity = 3;
dyn.opt.method = 'BFGS';

% Cost object
cost = Cost(D);
% cost.fcn = @lossSat;
% cost.z = zeros(10,1);
% cost.W = diag([1e-2 1e-4 1e-4 1e-3 1e-2 0 0 0 0 0]);

% initialize various variables
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1);

setupFigures

%% Initial rollouts (apply random actions)
ctrlRand = Ctrl(D, E, struct('type','random','maxU',policy.maxU));
for j = 1:J                                        % get the first observations
  [data(j), latent(j), realCost{j}] = ...
    rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost);
  disp([data(j).state [data(j).action; zeros(1,U)]]); disp(' ');
  animate(latent(j), data(j), dt, cost);
end

z(odei,:) = bsxfun(@plus, mu0(odei), chol(S0(odei,odei))'*randn(length(odei),1000));   % compute
for i = 1:size(z,2), z(augi,i) = plant.augment(z(:,i)'); end % the distribution
mu0Sim = mean(z,2); S0Sim = cov(z');         % of augmented start state by MCMC
mu0Sim(odei) = mu0(odei); S0Sim(odei,odei) = S0(odei,odei);        % Put in known correct values
s.m = mu0Sim(1:D); s.s = S0Sim(1:D,1:D);
s.s = s.s + (1e-6 + max([-eig(s.s); 0]))*eye(D); % s.s is sometimes not PSD 
clear z i;

% Set up controller
ctrl = CtrlNF(D, E, policy, angi, poli);
s = ctrl.reset_filter(s);

%% Start model-based policy search
for j = 1:N
  dyn.train(data,1:D,D-E+1:D); % trainDynModelnlds;
  learnPolicy;
  applyController;
  animate(latent(j+J), data(j+J), dt, cost);
  disp(['controlled trial # ' num2str(j)])
end
