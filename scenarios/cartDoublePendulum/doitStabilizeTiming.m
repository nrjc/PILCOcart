% cart-doube-pole experiment
%
%    dyno
%  1   1  oldu        old value of u
%  2   2  dx          Verlocity of cat
%  3   3  dtheta1     angular velocity of inner pendulum
%  4   4  dtheta2     angular velocity of outer pendulum
%  5   5  x           Position of cart
%  6   6  theta1      angle of inner pendulum
%  7   7  theta2      angle of outer pendulum
%  8      u           Force on Cart
%  9   8  sin(theta1)
% 10   9  cos(theta1)
% 11  10  sin(theta2)
% 12  11  cos(theta2)
%
% Copyright (C) 2008-2015 by Marc Deisenroth and Carl Edward Rasmussen,
% Jonas Umlauft, Rowan McAllister 2015-07-01
clear all;
errormag = [1]
delay = [0.06 0.10 0.14 0.18 0.22];
for errornum = 1:5
	for delaynum = 1
	    close all;
	    clc
	    dbstop if error
	    basename = ['CartDoubleStabilize' int2str(errornum) 'delay' int2str(delaynum) 'l'];

	    varNames = {'dx','dtheta1','dtheta2','x','theta1','theta2'};
	    varUnits = {'m/s','rad/s','rad/s','m','rad','rad'};
	    warning('on','all'); format short; format compact;
	    rng(1);

	    try
	      rd = '../../';
	      addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
		[rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
	    catch
	    end

	    D = 7;
	    E = 6;
	    U = 1;

	    % Indices
	    angi = [6 7];                 % indicies for vars treated as angles (using sin/cos rep)
	    augi = [];                    % indicies for vars augmented to the ode vars
	    dyni = [1 2 3 4 5 6 7];       % indicies for input into dynamics model
	    dyno = [2 3 4 5 6 7];         % indicies for output from the dynamics model and indicies to loss
	    odei = [2 3 4 5 6 7];         % indicies for ode solver
	    poli = [1 2 3 4 5 8 9 10 11]; % indicies for inputs to the policy

	    % Training parameters
	    dt = 1/20;                % [s] sampling time
	    plant.delay = delay(delaynum);                 % with a delay in the contol loop of 10 ms
	    T = 1.5;                  % [s] horizon time
	    H = ceil(T/dt);           % prediction steps (optimization horizon)
	    S0 = diag([1e-4 0.1 0.01 0.01 0.1 0.1 0.05 ].^2); % initial state covariance
	    S0(6,7) = 0.0049; S0(7,6) = 0.0049;
	    mu0 = [0 0 0 0 0 0 0]';   % initial state mean
	    maxU = 50; 
	    N = 15;                   % number controller optimizations
	    J = 1;                    % J trajectories, each of length H for initial training
	    K = 1;                    % number of initial states for which we optimize
	    %nc = 50;                                                % number of policy RBFs 

	    So = errormag(errornum)*0.25*[0.01/dt pi/180/dt pi/180/dt 0.01 pi/180 pi/180].^2; % noise levels, 1cm, 1 degree. This implements uncertainty in the readings!
	    %S0 = diag([1e-8,So]); % initial state covariance    % new change.
	    s.m = mu0(1:7); s.s = S0(1:7,1:7);                              % initial state

	    % Plant structure
	    plant.angi = angi;
	    plant.constraint = @(x)abs(x(5))>4;           % absolute position less than 4 m
	    plant.ctrltype = @(t,f,f0)zoh(t,f,f0);    % ctrl implemented as zero order hold
	    plant.dt = dt;
	    plant.ode = @dynamics;                                  % dynamics ode function
	    plant.poli = poli;
	    plant.noise = diag(So);                                     % measurement noise
	    plant.odei = odei;
	    plant.dyno = dyno;
	    plant.augi = augi;

	    policy.fcn = @(policy,m,s)conCat(@conlin,@gSat7,policy,m,s);
	    policy.maxU = maxU;                      % max. amplitude of control
	    policy.p.w = 0*randn(U, numel(poli));
	    policy.p.b = 0*randn(U, 1);
	    policy.opt = ...
		    struct('length',-50,'method','BFGS','MFEPLS',20,'verbosity',3,'fh',1);
	    global currT;

	    % Policy structure
	    % policy.fcn = @(policy,m,s)conCat(@congp,@gSat7,policy,m,s);
	    % policy.maxU = maxU;                      % max. amplitude of control
	    % mm = trigaug(mu0, zeros(length(mu0)), plant.angi);
	    % policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
	    % policy.p.target = 0.1*randn(nc, U);
	    % policy.p.hyp = zeros(length(poli)+2,1);
	    % policy.opt = ...
	    %       struct('length',-300,'method','BFGS','MFEPLS',20,'verbosity',3,'fh',1);

	    % Dynamics model object
	    dyn = gpa(D+U, E, angi, 'full');
	    %dyn.induce = zeros(50, 0, 1);                  % use 100 shared inducing inputs

	    % Cost object
	    cost = Cost(D);

	    % Set up figures
	    setupFigures

	    %% Initial Rollouts (apply random actions)
	    ctrlRand = Ctrl(D, E, struct('type','random','maxU',maxU));
	    for j = 1:J                                        % get the first observations
	      [data(j), latent(j), realCost{j}] = ...
		rollout(gaussian(mu0, S0), ctrlRand, H, plant, cost); %#ok<SAGROW>
	      disp([data(j).state [data(j).action; zeros(1,U)]]);
	      animate(latent(j), data(j), dt, cost);
	    end

	    % Set up controller
	    ctrl = CtrlNF(D, E, policy, angi, poli);
	    s = ctrl.reset_filter(s);

	    %% Start model-based policy search
	    for j = 1:N
	      %trainDirect(dyn, data, dyni, plant.dyno, j<20);
	      dyn.train(data,dyni,plant.dyno);
	      disptable(exp([dyn.on'; dyn.pn'; dyn.hyp.n]), varNames, ...
		  'observation noise|process noise std|inducing targets', '%0.5f');

	      learnPolicy;
	      applyController;
	      animate(latent(j+J), data(j+J), dt, cost);
	      disp(['controlled trial # ' num2str(j)]);
        end
	end
end
quit;
