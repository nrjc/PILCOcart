basename = 'arm_';
try
  rD = '../../';
  addpath([rD 'base'],[rD 'util'],[rD 'util/tprod'],...
    [rD 'gp'],[rD 'gp/NIGP'],[rD 'control'],[rD 'loss']);
catch
end
rng(2);
format short; format compact; 

%  1   1  theta   shoulder angle
%  2   2  phi     elbow angle relative to shoulder(>= 0, i.e. no hyperextension)
%  3   3  dtheta  shoulder angular velocity
%  4   4  dphi    elbow angular velocity relative to shoulder


varNames = {'theta', 'phi', 'dtheta', 'dphi'};

% odei  indicies for the ode solver
% augi  indicies for variables augmented to the ode variables
% dyno  indicies for the output from the dynamics model and indicies to loss
% angi  indicies for variables treated as angles (using sin/cos representation)
% dyni  indicies for inputs to the dynamics model
% poli  indicies for the inputs to the policy

odei = [1 2 3 4];
augi = [];
dyno = [1 2 3 4];
angi = [];
dyni = [1 2 3 4];
poli = [1 2 3 4];
varNames = varNames(dyno);

dt = 0.1;                                                  % [s] sampling time
T = 3.0;                                  % [s] initial prediction horizon time
H = ceil(T/dt);                       % prediction steps (optimization horizon)
maxH = ceil(10.0/dt);                                        % max pred horizon
S0 = (0.01 * diag([1 1 1/dt 1/dt])).^2;
mu0 = [0.785398;2.094395;0;0];                             % initial state mean
N = 20;                                       % number controller optimizations
J = 5;                                     % initial J trajectories of length H
K = 1;                         % number of initial states for which we optimize

plant.ode = @dynamics;                                  % dynamics ode function
% plant.constraint = @(x)(abs(x(14))>pi/2 | abs(x(17))>pi/2); % ode constraint
plant.noise = (0.03 * diag([1 1 1/dt 1/dt])).^2;
plant.dt = dt;
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);           % controler is zero order hold
plant.odei = odei;                 % indices to the varibles for the ode solver
plant.augi = augi;                             % indices of augmented variables
plant.angi = angi;
plant.poli = poli;
plant.dyno = dyno;
plant.dyni = dyni;
plant.prop = @propagated;
plant.delay = 0;

global odeA odeFIELD odeF2 odeF3 fieldCentre;
odeA = 1;                                       % Model stiffness parameter
odeFIELD = 1;                      % Field selector 1: Null, 2: Curl, 3: DF
odeF2 = 30;                       % Field strength constant for curl field
odeF3 = 30;                          % Field strength constant for DF field

ctrl.fcn = @ctrlNF;
ctrl.fcn2 = @ctrlNF2;
ctrl.state = [];
ctrl.init = [];
ctrl.policy.fcn = @(policy,m,s)conCat(@conAdd,@gSat,policy,m,s);
ctrl.policy.fcn2 = @(policy,m,s)conCat(@conAdd2,@gSat,policy,m,s);
ctrl.policy.sub{1}.fcn = @conlin; % second controller is conlin
ctrl.policy.sub{1}.poli = poli;
ctrl.policy.p{1}.w = 1e-2*randn(6,length(poli));
ctrl.policy.p{1}.b = zeros(6,1);
ctrl.policy.maxU = 20*ones(1,6);             % Control input will be offset by maxU
                                     % in dynamics.m to enforce positive signal
ctrl.U = length(ctrl.policy.maxU);
ctrl.policy.addCtrlNoise = 'off';
ctrl.policy.ctrlNoise = 0.025;     % Noise sd of muscle activation as a percentage

cost.fcn = @lossSat;                                           % cost function
cost.gamma = 1;
cost.width = 1;                                           % cost function width
cost.expl = 0;                                                 % no exploration
cost.z = [1.5708; 0.3713; 0; 0];                                 % Target state
cost.W = diag([2 2 0 0]);

[~,y1] = cartSpace([0.29 0.34], mu0(1:2)');
[~,y2] = cartSpace([0.29 0.34], cost.z(1:2)');
fieldCentre = [-0.1233538; (y1+y2)/2];             % Centre of reaching movement

dynmodel.fcn = @gpBase;
dynmodel.train = @train;                     % function to train dynamics model
dynmodel.induce = zeros(300,0,4);                 % non-shared inducing inputs

opt.length = -80; opt.verbosity = 3;          % options for minimize for policy
dynmodel.opt = [-500 -500];                         % options for dynmodel training

x = []; y = [];                                  % initialize various variables
fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);