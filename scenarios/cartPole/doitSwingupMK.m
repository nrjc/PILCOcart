% Cart and pole swingup task.
%
%  1  1  oou        even older value of u
%  2  2  ox         old cart position
%  3  3  otheta     old angle of the pendulum
%  4  4  ou         old value of u
%  5  5  x          cart position
%  6  6  theta      angle of the pendulum
%  7     v          cart velocity
%  8     dtheta     angular velocity
%  9     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-05-26

clear
close all
format short; format compact;

basename = 'swingupMK6';
varNames = {'oou','ox','otheta','ou','x','theta'};
rng(3);

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'], [rd 'direct']);
catch
end

D = 6;
E = 2;
U = 1; 

% Indices
angi = [3 6];
augi = [];
dyno = [5 6];
odei = [5 6 7 8];
poli = [1 2 4 5 7 8 9 10];

% Training parameters
dt = 0.15; plant.dt = dt;                                 % time step is 150 ms
plant.delay = 0.05;                  % with a delay in the contol loop of 50 ms
H = ceil(2/dt);
mu0 = [0 0 pi 0 0 pi 0 0]';                                 % initial state mean
S0 = diag([1e-9 0.2 0.2 1e-9 0.2 0.2 0.2 0.2].^2);      % initial state variance
s.m = mu0(1:D); s.s = S0(1:D,1:D);

% Plant structure
plant.angi = angi;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);                % ctrl is zero order hold
plant.dt = dt;
plant.ode = @dynamics;
plant.poli = poli;
plant.prop = @propagated;
plant.noise = diag([0.02 pi/180].^2);
%plant.noise = diag([0.06 6*pi/180].^2);
plant.odei = odei;
plant.dyno = dyno;
plant.augi = augi;

% MCMC to approximate mu0 and S0
I = 500;
z = zeros(I,length(mu0));
ctrlRand = Ctrl(D,E,struct('type','random','maxU',0));
for i = 1:I
  print_loop_progress(i, I, 'sampling valid Markov states')
  [~, zz] = rollout(gaussian(mu0, S0), ctrlRand, 1, plant);
  z(i,:) = zz(2,:);
end
mu0 = mean(z)';
S0 = cov(z)+eps*eye(8);
s = struct('m',mu0(1:D),'s',S0(1:D,1:D));

N = 10; 
K = 1;                         % number of initial states for which we optimize
J = 1;                  % J trajectories, each of length H for initial training
maxU = 10;
nc = 50;
mm = trigaug(s.m, zeros(length(s.m)), angi);

% Policy structure
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = maxU;                                 % max. amplitude of control
policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
policy.p.target = 0.1*randn(nc, U);
policy.p.hyp = log([1 0.7 1 1 0.7 1 1 1 1 0.01])';
policy.opt = struct('length',-300,'MFEPLS',20,'verbosity',3);

% Dynamics model object
dyn = gpa(D+U, E, angi);                                % D+U inputs, E outputs
dyn.induce = zeros(50,0,1);                    % use 100 shared inducing inputs

% Cost object
cost = CostMK(D);

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
for j = 1:20
  H = H + 1;
  data(j).state = data(j).state(2:end,:);
  data(j).action = data(j).action(2:end,:);
  latent(j).state = latent(j).state(2:end,:);
  trainDirect(dyn, data, [1:6], plant.dyno, j<8);
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            ['observation noise|process noise std|inducing targets'], '%0.5f');
  learnPolicy;
  applyController;
  animate(latent(j+1), data(j+1), dt, cost);
  disp(['controlled trial # ' num2str(j)]);
end
