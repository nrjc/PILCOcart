function propT

% test prop.m function, 2014-02-25
%
% Rowan McAllister

% Constants
N = 10;  % # input datum
E = 2;   % # output (state) dims = D0
pE = E;
F = 3;   % # action dims
D = E+F; % # input (state-action) dims
NANGI = 1;
SEED = 50;

% Setup
addpath('../base'); addpath('../util'); addpath('../control'); addpath('../gp');
rand('seed',SEED); randn('seed',SEED);

% Misc Inputs
m = randn(3*E,1);
s = randn(3*E); s = s*s';

% Plant
plant.poli = 1:E;
plant.dyno = 1:E;
plant.dyni = 1:E;
plant.difi = 1:E;
plant.angi = ones(NANGI,1);
plant.prop = @propagated;

% Policy
policy.p.w = randn(F,E);
policy.p.b = randn(F,1);
policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);
policy.maxU = 20*ones(1,F);

% Dynamics
dynmodel.inputs = randn(N,D,pE);
dynmodel.target = randn(N,E);
dynmodel.fcn = @gph;
dynmodel.iK = randn(N,N,E);
dynmodel.beta = randn(N,E);

for i=1:E
  h.l = log(rand(D,1));
  h.s = log(rand(1));
  h.n = log(rand(1));
  h.m = rand(D,1);
  h.b = rand(1);
  h.on = log(rand(1));
  dynmodel.hyp(i) = h;
end

% Test Compiles
[M, S] = prop(m, s, plant, dynmodel, policy)