% Cart and double pole swingup task using a 2nd order Markov representation.
%
%  1   1  oou         even older value of u
%  2   2  ox          old position of cart
%  3   3  otheta1     old angle of inner pendulum
%  4   4  otheta2     old angle of outer pendulum
%  5   5  ou          old value of u
%  6   6  x           position of cart
%  7   7  theta1      angle of inner pendulum
%  8   8  theta2      angle of outer pendulum
%  9      dx          verlocity of cart
%  10     dtheta1     angular velocity of inner pendulum
%  11     dtheta2     angular velocity of outer pendulum    
%  12     u           force applied to cart
%
% Copyright (C) 2015 by Carl Edward Rasmussen, Rowan McAllister 2015-05-26

clear
close all
format short; format compact; 

basename = 'swingupMarkov';
varNames = {'x','theta1','theta2'};
rng(1);

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'control'],[rd 'loss'], [rd 'direct']);
catch
end

D = 8;
E = 3;
U = 1;

% Indices
angi = [3 4 7 8];
augi = [];
dyno = [6 7 8];
odei = [9 10 11 6 7 8];
poli = [1 2 5 6 9 10 11 12 13 14 15 16];

% Training parameters
T = 2.0;                                            % time horizon is 2 seconds
dt = 1/15; plant.dt = dt;                                  % time step is 67 ms
plant.delay = 0.01;                  % with a delay in the contol loop of 10 ms
H = ceil(T/dt);
N = 40;      % number controller optimizations
J = 1;       % J trajectories, each of length H for initial training
K = 1;       % number of initial states for which we optimize
maxU = 20;

mu0 = [0 0 pi pi 0 0 0 0 0 pi pi ]';                                  % initial state mean
S0 = diag([1e-9 0.2 0.2 0.2 1e-9 0.2 0.2 0.2 0.2 0.2 0.2].^2)/4;           % initial state variance

% Plant structure
plant.noise = 0.25*diag([0.01 pi/180 pi/180].^2);
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);                % ctrl is zero order hold
plant.ode = @dynamics;
plant.odei = odei;
plant.dyno = dyno;
plant.augi = augi;

% MCMC to approximate mu0 and S0
I = 500;
z=zeros(length(plant.odei)+E,I);
for i = 1:I
  print_loop_progress(i, I, 'sampling valid Markov states')
  clear simulate;  
  pre_x=gaussian(mu0(plant.odei), S0(plant.odei, plant.odei));
  z(1:E,i)=pre_x(length(plant.odei)-E+1:length(plant.odei));
  z(E+1:length(plant.odei)+E,i)=odestep(pre_x, 0, plant);
end
mu0nocontrol = mean(z,2); S0nocontrol = cov(z');
mu0 = [0 mu0nocontrol(1:E)' 0 mu0nocontrol(E+1:length(plant.odei)+E)']';  
S0([1,E+2],[1,E+2]) = diag([1e-9 1e-9].^2);
S0([2:E+1 E+3:length(mu0)],[2:E+1 E+3:length(mu0)]) = S0nocontrol;
S0 = S0 + eps*eye(length(mu0));
s = struct('m',mu0([1:5 plant.dyno]),'s',S0([1:5 plant.dyno],[1:5 plant.dyno]));

% Policy structure
policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = maxU;                                 % max. amplitude of control
nc = 50;
mm = trigaug(s.m, zeros(length(s.m)), angi);
policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)'; 
policy.p.target = 0.1*randn(nc, U);
policy.p.hyp = log([1 1 1 1 0.7 0.7 0.7 0.7 0.7 0.7 0.7 0.7 1 0.01])';
policy.opt = struct('length',-200,'method','BFGS','MFEPLS',20,'verbosity',3);

% Dynamics model object
dyn = gpa(D+U, E, angi, 'vfe'); % 8 inputs, 6 outputs, and var number 2, 3 are angles
dyn.induce = zeros(100, 0, 1);                 % use 100 shared inducing inputs
dyn.opt = struct('length',-200,'verbosity',3,'method','BFGS','fh',6);

% Cost object
cost = CostMK(D);

% Set up figures
setupFigures;

%% Initial Rollouts (apply random actions)
realCost = cell(1,N+J);
ctrlRand = Ctrl(D, E, struct('type','random','maxU',maxU));
for j = 1:J                                        % get the first observations
  [data(j), latent(j), realCost{j}] = ...
            rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost);
  disp([data(j).state [data(j).action; zeros(1,U)]]);
end

ctrl = CtrlNF(D,E,policy,angi,poli);
s = ctrl.reset_filter(s);

%% Start model-based policy search
for j = 1:N
  data(j).state = data(j).state(2:end,:);
  data(j).action = data(j).action(2:end,:);
  latent(j).state = latent(j).state(2:end,:);
  trainDirect(dyn, data, [1:5 plant.dyno], plant.dyno, 1);
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            ['observation noise|process noise std|inducing targets'], '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  
  if j<5            % early on, we reset the policy at every iteration
    x = cell2mat(arrayfun(@(Y)Y.state(1:end-1,:),data,'uniformoutput', 0)');
    x = x(:,[1:5 plant.dyno]);
    for i = 1:numel(angi)         % augment with trigonometric functions
      x(:,end+1) = sin(x(:,angi(i)));
      x(:,end+1) = cos(x(:,angi(i)));
    end
    x = x(:,poli);
    jj = randperm(size(x,1));
    policy.p.inputs = x(jj(1:min(nc,size(x,1))),:);
    policy.p.target = 0.1*randn(min(nc,size(x,1)),U);
    policy.p.hyp = log([std(x) 1 0.01]');
    ctrl.set_policy_p(policy.p);
  end

  learnPolicy;
  applyController;
  animate(latent(j+J), data(j+J), dt, cost);
  disp(['controlled trial # ' num2str(j)]);
end
