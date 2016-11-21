% Cart and double pole swingup task.
%
%  1   1  x           Position of cart
%  2   2  theta1      angle of inner pendulum
%  3   3  theta2      angle of outer pendulum
%  4   4  dx          Verlocity of cart
%  5   5  dtheta1     angular velocity of inner pendulum
%  6   6  dtheta2     angular velocity of outer pendulum    
%  7   7  oldu        old value of u
%  8      u           force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2014-12-09

clear
close all
format short; format compact; 

basename = 'swingup2';
varNames = {'x','theta1','theta2','dx','dtheta1','dtheta2'};

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
               [rd 'gp/NIGP'],[rd 'control'],[rd 'loss'], [rd 'direct']);
catch
end
rng(1);

D = 8;
E = 6;

mu0 = [0 pi pi 0 0 0 0]';                                  % initial state mean
S0 = diag([0.2 0.2 0.2 0.2 0.2 0.2 0].^2)/4;           % initial state variance
s = struct('m',mu0,'s',S0);

plant.odei = [1 2 3 4 5 6];
plant.dyno = [1 2 3 4 5 6];
plant.augi = [];
plant.oldu = [7];
poli = [1 4 5 6 7 8 9 10 11];
angi = [2 3];
dt = 0.05; plant.dt = dt;                                 % time step is 150 ms
plant.delay = 0.01;                  % with a delay in the contol loop of 50 ms
H = ceil(5/plant.dt);
plant.noise = diag([0.01 pi/180 pi/180 0.01 pi/180 pi/180].^2);
plant.ctrltype = @(t,f,f0)zoh(t,f,f0);                % ctrl is zero order hold
plant.ode = @dynamics;

N = 25; K = 1; J = 1; U = 1; maxU = 20;
cost.fcn = @loss;
cost.ell = [0.6 0.6];
cost.gamma = 1.0;
cost.width = 1.0;

fantasy.mean = cell(1,N); fantasy.std = cell(1,N);
realCost = cell(1,N+J); M = cell(N,1); Sigma = cell(N,1); latent = cell(1,N+J);
ctrlRand = Ctrl(D, E, struct('type','random','maxU',maxU));
for j = 1:J                                        % get the first observations
  [data(j), latent{j}, realCost{j}] = ...
            rollout(gaussian(mu0(1:6), S0(1:6,1:6)), ctrlRand, H, plant, cost);
  disp([data(j).state [data(j).action; zeros(1,U)]]);
end

dyn = gpa(8, 6, angi);    % 8 inputs, 6 outputs, and var number 2, 3 are angles
dyn.induce = zeros(150, 0, 1);                 % use 100 shared inducing inputs

policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = maxU;                                 % max. amplitude of control
nc = 50;
mm = trigaug(mu0, zeros(7), angi);
policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
policy.p.target = 0.1*randn(nc, U);
policy.p.hyp = log([1 0.7 0.7 1 0.7 1 1 1 1 1 0.01])';
policy.opt = struct('length',-300,'MFEPLS',20,'verbosity',3);

ctrl = CtrlNF(policy,angi,poli,zeros(6,1));    % last element just to give size

% Set up figures
for i=1:6; figure(i); end
if usejava('awt');
  set(1,'windowstyle','docked','name','Policy training')
  set(2,'windowstyle','docked','name','Previous policy')
  set(3,'windowstyle','docked','name','Loss')
  set(4,'windowstyle','docked','name','States')
  set(5,'windowstyle','docked','name','Rollout')
  set(6,'windowstyle','docked','name','Dynamics training')
  desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
  desktop.setDocumentArrangement('Figures',2,java.awt.Dimension(2,3));
  clear desktop;
end

for j = 1:N
  trainDirect(dyn, data, [1:7], [1:6], j<7);
  disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            ['observation noise|process noise std|inducing targets'], '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  learnPolicy;
  applyController;
  if ishandle(5); set(0,'currentfigure',5); clf(5);
  else figure(5); set(5,'windowstyle','docked','name','Rollout'); end
  draw_rollout(j, J, data, H, dt, cost, S{j});
  disp(['controlled trial # ' num2str(j)]);
end
