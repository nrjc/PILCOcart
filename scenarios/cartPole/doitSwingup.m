% Cart and pole swingup task.
%
%  1  1  oldu       old value of u
%  2  2  x          cart position
%  3  3  theta      angle of the pendulum
%  4  4  v          cart velocity
%  5  5  dtheta     angular velocity
%  6     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-05-25

clear
close all
format short; format compact;

basename = 'swingup';
varNames = {'x','theta','dx','dtheta'};
rng(2);

run ../../util/addpaths

D = 5;
E = 4;
U = 1;

% Indices
angi = [3];
augi = [];
dyno = [2 3 4 5];
odei = [2 3 4 5];
poli = [1 2 4 5 6 7];

% Training parameters
dt = 0.15;                                 % time step is 150 ms
plant.delay = 0.05;                  % with a delay in the contol loop of 50 ms
T = 5.0;                                              % time horizon in seconds
H = ceil(T/dt);                      % horizon by number of discrete time steps
mu0 = [0 0 pi 0 0]';                                       % initial state mean
S0 = diag([1e-9 0.2 0.2 0.2 0.2].^2);                  % initial state variance
s = struct('m',mu0,'s',S0);
N = 20;                                       % number controller optimizations
K = 1;                         % number of initial states for which we optimize
J = 1;                  % J trajectories, each of length H for initial training
nc = 50;                                                % number of policy RBFs 
mm = trigaug(mu0, zeros(5), angi);

% Plant structure
plant.dt = dt;
plant.oldu = [1];
plant.noise = diag([0.02 pi/90 0.02 pi/90].^2);
% plant.noise = diag([0.05 pi/45 0.05 pi/45].^2);
% plant.noise = diag([0.05 pi/45 0.05/dt pi/45/dt].^2)
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);                % ctrl is zero order hold
plant.ode = @dynamics;
plant.odei = odei;
plant.dyno = dyno;
plant.augi = augi;

% Policy structure
maxU = 10; policy.maxU = maxU;  % max. amplitude of control
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
policy.p.target = 0.1*randn(nc, U);
policy.p.hyp = log([1 0.7 1 1 0.7 1 1 0.01])';
policy.opt = struct('length',-300,'MFEPLS',20,'verbosity',3);

% Dynamics model
dyn = gpa(D+U,E,angi,'vfe');       % 6 inputs, 4 outputs, and var 2 is an angle
dyn.induce = zeros(50,0,1);                    % use 100 shared inducing inputs

% Cost function
cost = Cost(D);

% Set up figures
setupFigures

%% Initial Rollouts (apply random actions)
realCost = cell(1,N+J);
ctrlRand = Ctrl(D,E,struct('type','random','maxU',maxU));
for j = 1:J                                        % get the first observations
  [data(j), latent(j), realCost{j}] = ...
            rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost);
  disp([data(j).state [data(j).action; zeros(1,U)]]);
end

% Set up controller
ctrl = CtrlNF(D,E,policy,angi,poli);
s = ctrl.reset_filter(s);

%% Start model-based policy search
for j = 1:N
  %trainDirect(dyn, data, [1:5], [2:5], j<8);
  dyn.train(data,1:D,dyno); % trainDynModelnlds;
  disptable(exp([dyn.on'; dyn.pn'; [dyn.hyp.n]]), varNames, ...
            'observation noise|process noise std|inducing targets', '%0.5f');
  learnPolicy;
  applyController;
  animate(latent(j+J), data(j+J), dt, cost);
  disp(['controlled trial # ' num2str(j)]);
end
