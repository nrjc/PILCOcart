% cart-doube-pole experiment
%
%    dyno
%  1   1  x           Position of cart
%  2   2  theta1      angle of inner pendulum
%  3   3  theta2      angle of outer pendulum
%  4   4  dx          Verlocity of cart
%  5   5  dtheta1     angular velocity of inner pendulum
%  6   6  dtheta2     angular velocity of outer pendulum    
%  7      sin(theta1)
%  8      cos(theta1)
%  9      sin(theta2)
% 10      cos(theta2)
% 11      u1          Force on Cart
%
% Copyright (C) 2008-2012 by Marc Deisenroth and Carl Edward Rasmussen,
% 2014-03-14
% Edited by Jonas Umlauft 2014-07-09
% Edited by Rowan 2014-07-12
clear all; close all;
basename = 'CartDouble_swingup';

varNames = {'x','theta1','theta2','dx','dtheta1','dtheta2'};
rng(1); warning('on','all'); format short; format compact; 

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],...
         [rd 'gp'],[rd 'gp/NIGP'],[rd 'control'],[rd 'loss']);
catch
end


% 1. Indices
odei = [1 2 3 4 5 6];      % indicies for the ode solver
augi = [] ;                %indicies for variables augmented to the ode variables
dyno = [1 2 3 4 5 6];      % indicies for the output from the dynamics model and indicies to loss
angi = [2 3];              % indicies for variables treated as angles (using sin/cos representation)
dyni = 1:10;               % indicies for inputs to the dynamics model
poli = [1 4 5 6 7 8 9 10]; %  indicies for the inputs to the policy
% difi  indicies for dynamics targets which are deltas (rather than values) 

% 2. Training parameters
dt = 0.1;                 % [s] sampling time
T = 5;                    % [s] prediction time
H = ceil(T/dt);           % prediction steps (optimization horizon)
maxH = H;                 % max pred horizon
nc = 50;                  % size of controller training set
S0 = diag([0.1 0.01 0.01 0.1 0.1 0.1 ].^2); % initial state covariance
mu0 = [0 pi pi 0 0 0]';   % initial state mean
N = 40;                   % number controller optimizations
J = 1;                    % J trajectories, each of length H for initial training
K = 1;                    % number of initial states for which we optimize
So = 0.01*[0.01 pi/180 pi/180 0.01/dt pi/180/dt pi/180/dt ].^2; % noise levels, 1cm, 1 degree

% 3. set up the plant structure
plant.ode = @dynamics;                              % dynamics ode function
plant.noise = diag(So);                                 % measurement noise
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0); % ctrl implemented as zero order hold
plant.odei = odei;              % indices to the varibles for the ode solver
plant.augi = augi;                          % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;
plant.constraint = @(x)abs(x(1))>4;

% 4. Controller structure
ctrl.fcn = @ctrlNF;
ctrl.policy.maxU = 20;                          % max. amplitude of control
ctrl.U = length(ctrl.policy.maxU);
mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
ctrl.policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
ctrl.policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
ctrl.policy.p.target = 0.1*randn(nc, ctrl.U);
ctrl.policy.p.hyp = zeros(length(poli)+2,1);

% Construct state indexer
D = length(dyno); s.m = nan(D,1); s.s = nan(D);
[~,~,~,s] = ctrl.fcn(s, 'ResetFilter');
plant.ns = length(unwrap(s));
plant.np = length(unwrap(ctrl.policy.p));
plant.is = rewrap(s,1:plant.ns);         % state stuct of members' indexes

% 4A Set up dynamics model structure
E = length(dyno); D = length(dyni) + ctrl.U;
dynmodel.fcn = @gpBase;                      % function for GP predictions
dynmodel.approxS = 0;                  % use full output covariance matrix
dynmodel.trainMean = 0;           % keep the GP mean fixed during training
[dynmodel.hyp(1:E).m] = deal(zeros(D,1)); [dynmodel.hyp.b] = deal(0);
for i=1:E; dynmodel.hyp(i).m(i) = 1; end                   % identity mean
dynmodel.hyp(1).m(4) = dt; dynmodel.hyp(2).m(5) = dt;  % Euler integration
dynmodel.hyp(3).m(6) = dt;
dynmodel.train = @train;
dynmodel.opt = -500;
ctrl.dynmodel = dynmodel; ctrl.dynmodel.fcn = @gphd;

% 5. set up the cost function
cost.fcn = @loss;                                 % cost function
cost.gamma = 1;                                   % discount factor
cost.p = [1 1];                                   % lenghts of the links
cost.width = 2*[1.2 0.5];                                 % cost function width
cost.expl = 0;                                    % exploration parameter
cost.angle = plant.angi;                          % angle variables in cost
cost.target = zeros(6,1);                         % target state

% 6. options for policy optimization
opt.length = 150;                        % max. number of line searches
opt.MFEPLS = 30;                         % max. number of function evaluations per line search
opt.verbosity = 3;                       % verbosity: specifies how much information is displayed
opt.method = 'BFGS';

% 7. some initializations
x = []; y = [];                                  % initialize various variables
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
  [data(j), latent{j}, realCost{j}] = rollout(mu0, ...
      struct('policy',struct('maxU',ctrl.policy.maxU),'U',ctrl.U), H, plant, cost);
  disp([data(j).state [data(j).action;zeros(1,ctrl.U)]]);

  if ishandle(5); set(0,'currentfigure',5); clf(5); 
  else figure(5); set(5,'windowstyle','docked','name','Rollout'); end
  draw_rollout(plant,j,0,data,H,dt,cost);
end

mu0Sim(odei,:) = mu0; S0Sim(odei,odei) = S0;
s.m = mu0Sim(dyno); s.s = S0Sim(dyno,dyno);


%% Start model-based policy search
for j = 1:N
  trainDynModel;
  ctrl.on = [dynmodel.hyp.on];    % copy learnt observation noise to controller
  ctrl.dynmodel = dynmodel; ctrl.dynmodel.fcn = @gphd;
  learnPolicy;  
  applyController;
  if ishandle(5); set(0,'currentfigure',5); clf(5); 
  else figure(5); set(5,'windowstyle','docked','name','Rollout'); end
  draw_rollout(plant,j,J,data,H,dt,cost,M,Sigma);
  disp(['controlled trial # ' num2str(j)]);
end
