% Default parameters for the cartPole.
%
%  1  1  x          cart position
%  2  2  theta      angle of the pendulum
%  3  3  v          cart velocity
%  4  4  dtheta     angular velocity
%  5     sin(theta) 
%  6     cos(theta)
%  7     u          force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Marc Deisenroth, Andrew McHutchon,
% and Joe Hall 2014-04-14 , Edited by Jonas Umlauft 2014-06-30

clear; close all;
basename = 'cartPole_stabilize';
varNames = {'x','theta','dx','dtheta'};
format short; format compact; 
try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'], [rd 'GPNLDS']);
catch
end
SEED = 1; rand('seed',SEED); randn('seed',SEED);


% 1. Parameter initialization
odei = [1 2 3 4];                            % varibles for the ode solver
augi = [];                                     % variables to be augmented
dyno = [1 2 3 4];          % variables to be predicted (and known to loss)
angi = 2;                                                % angle variables
dyni = [1 2 3 4 5 6];  % variables that serve as inputs to the dynamics GP
poli = [1 3 4 5 6];         % variables that serve as inputs to the policy


dt = 0.10;                                             % [s] sampling time
T = 5.0;                             % [s] initial prediction horizon time
H = ceil(T/dt);                  % prediction steps (optimization horizon)
mu0 = [0 0 0 0]';                                     % initial state mean 
S0 = diag([0.1 0.1 0.1 0.1].^2);                  % initial state variance
N = 10;                                  % number controller optimizations
J = 2;                                % initial J trajectories of length H
K = 1;                    % number of initial states for which we optimize
nc = 20;                            % number of controller basis functions
So = [0.01 pi/180 0.01/dt pi/180/dt ].^2;     % noise levels, 1cm, 1 degree

% 2. Plant structure
plant.ode = @dynamics;                             % dynamics ode function 
plant.noise = diag(So);                                % measurement noise
plant.dt = dt;
plant.delay = 0.01;                                   % delay in ctrl loop
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);           % ctrl is zero order hold
plant.odei = odei;
plant.augi = augi;
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;

% 3. Controller structure
ctrl.fcn = @ctrlNF;
ctrl.state = [];
ctrl.init = [];
ctrl.policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);
ctrl.policy.maxU = 10;                          % max. amplitude of control
ctrl.U = length(ctrl.policy.maxU);
mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
ctrl.policy.p.w = randn(ctrl.U,numel(poli));
ctrl.policy.p.b = randn(ctrl.U,1);

% 4. Cost function
cost.fcn = @loss;                                          % cost function
cost.gamma = 1;                                          % discount factor
cost.p = 0.5;                                         % length of pendulum
cost.width = 0.25;                                   % cost function width
cost.expl = 0.0;                                          % no exploration
cost.angle = plant.angi;              % index of angle (for cost function)
cost.target = [0 0 0 0]';                                  % target state

% 5. Dynamics model structure
dynmodel.fcn = @gpBase;                      % function for GP predictions
dynmodel.approxS = 0;                  % use full output covariance matrix
dynmodel.trainMean = 0;           % keep the GP mean fixed during training
[dynmodel.hyp(1:4).m] = deal(zeros(7,1)); [dynmodel.hyp.b] = deal(0);
for i=1:4; dynmodel.hyp(i).m(i) = 1; end                   % identity mean
dynmodel.hyp(1).m(3) = dt; dynmodel.hyp(2).m(4) = dt;  % Euler integration
dynmodel.train = @train;
dynmodel.induce = zeros(400,0,4);
dynmodel.opt = -500;


% 6. Parameters for policy optimization
opt.length = -300;                           % max. no. of line searches
opt.MFEPLS = 30;        % max. no. of function evaluations per line search
opt.verbosity = 3;           % specifies how much information is displayed

% 7. Some initializations
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N); M = cell(N,1); Sigma = cell(N,1);

% 8. Set up figures
for i=1:5; figure(i); end
if usejava('awt'); 
    set(1,'windowstyle','docked','name','Policy Optimisation')
    set(2,'windowstyle','docked','name','Previous Optimisations')
    set(3,'windowstyle','docked','name','Loss')
    set(4,'windowstyle','docked','name','States')
    set(5,'windowstyle','docked','name','Rollout')
    desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
    desktop.setDocumentArrangement('Figures',2,java.awt.Dimension(2,3));
    clear desktop;
end






%% Initial Rollouts (apply random actions)
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);
for j = 1:J                                   % get the first observations
  [data(j), latent{j}, realCost{j}] = rollout(gaussian(mu0, S0), ...
      struct('policy',struct('maxU',ctrl.policy.maxU),'U',ctrl.U), H, plant, cost);
  disp([data(j).state [data(j).action;zeros(1,ctrl.U)]]);

  if ishandle(5); set(0,'currentfigure',5); clf(5); 
  else figure(5); set(5,'windowstyle','docked','name','Rollout'); end
  draw_rollout(j,0,data,H,dt,cost);
end

% mu0Sim, S0Sim: moments of (partial) state used for internal simulation
mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;      
mu0Sim = mu0Sim(dyno); S0Sim = S0Sim(dyno,dyno);

%% Start model-based policy search
for j = 1:N
  trainDynModel;
  learnPolicy;  
  applyController;
  if ishandle(5); set(0,'currentfigure',5); clf(5); 
  else figure(5); set(5,'windowstyle','docked','name','Rollout'); end
  draw_rollout(j,J,data,H,dt,cost,M,Sigma);
  disp(['controlled trial # ' num2str(j)]);
end