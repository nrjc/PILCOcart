%% simulate.m
% *Summary:* Simulate dynamics using a given control scheme.
%
%    function next = simulate(x0, f, plant)
%
% Compute the next discrete time state from the current state and action by
% solving the continuous time plant dynamics ode. The type of implementation
% for the controller is specified in plant.ctrltype, one of zoh (Zero Order
% Hold), foh (First Order Hold) or expdecay (exponential decay), defined below,
% which turn the discrete time control action into a continuous time control
% signal. If no plant.delay exists, the ode-solver (ode45) is called directly
% with the plant.ctrltype, otherwise a time delay is introduced, and the
% control actions from the previous time interval (saved in persistent variable
% U) is mixed together with the new action, ensuring a smooth transition (the
% transition takes about one 100th of the time interval) to avoid creating
% difficulties for the ode solver at the transition.    
%
% *Input arguments:*
%
% x0           initial state
% f            vector of control actions
% plant        plant structure; fields used by this function are
%   .dt         time discretization
%   .ode        ordinary differential equation describing the system dynamics
%   .ctrltype   function defining control implementation, possible values
%      @zoh       zero-order-hold control (ZOH)
%      @foh       first-order-hold control (FOH) with rise time
%      @expdecay  lagged control with time constant
%   .delay      optional, continuous-time delay, in range [0 dt)
%
% *Output arguments:*
%
%  next    successor state (with additional control states if required)
%
%
% Copyright (C) 2008-2014 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, and Carl Edward Rasmussen.
%
% Last modification: 2013-12-16
%
%% High-Level Steps
% For each time step
% # Set up the control function
% # Simulate the dynamics (by calling ODE45)

function next = simulate(x0, f, plant)
%% Code

OPTIONS = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);         % accuracy of ode45
persistent U;                                % previous control action function 
dt = plant.dt; eval(['ctrltype = ', func2str(plant.ctrltype), ';']);   % scope!

% 1. Set up control function ------------------------------------------------
if isempty(U), U = @(t)ctrltype(t+dt, 0*f, 0*f); end; U0 = U(0);

% 1a (optional) handle delay
if isfield(plant, 'delay')
  d = plant.delay; s = -100/dt;   % set steepness of transition in mix function
  mix = @(t, a, b)(a(t-d)*exp(s*t)+b(t-d)*exp(s*d))/(exp(s*t)+exp(s*d));
  u = @(t)mix(t, U, @(t)ctrltype(t, f, U0));             % mix actions smoothly
else
  u = @(t)ctrltype(t, f, U0);
end

% 2. Simulate dynamics ------------------------------------------------------
[T, y] = ode45(plant.ode, [0 dt/2 dt], x0, OPTIONS, u);          % solve the ode
next = y(3,:)';                                       % extract the desired state 
U = @(t)ctrltype(t+dt, f, U0);               % save the control action function


function u = zoh(t, f, f0)                            % zero order hold control
u = f;

function u = foh(t, f, f0, risetime) % first order hold with specified risetime
if t < risetime, u = f0 + t*(f-f0)/risetime; else u = f; end

function u = expdecay(t, f, f0, tau) % exponential decay with time constant tau
u = f + exp(-t/tau)*(f0-f);
