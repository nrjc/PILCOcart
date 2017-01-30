function x1 = odestep(x0, f, plant)

% Compute the next discrete time state from the current state and action by
% solving the continuous time plant dynamics ode. The type of implementation
% for the controller is specified in plant.ctrltype, one of zoh (Zero Order
% Hold), foh (First Order Hold) or expdecay (exponential decay), defined below,
% which turn the discrete time control action into a continuous time control
% signal. If no plant.delay exists, the ode-solver (ode45) is called directly
% with the plant.ctrltype, otherwise the ode-solver is called twice, first for
% a time-interval of length delay with the previous controller (which is saved
% in the persistent variable "u"), and subsequently the remaining interval with
% the new controller. Note, that the local time passed through to the
% controller function is always in the range [0 dt].
%
% x0             initial state
% f              vector of control actions
% plant          plant structure; fields used by this function are
%   dt           time discretization
%   ode          ordinary differential equation describing the system dynamics
%   ctrltype     function defining control implementation, possible values
%     @zoh       zero-order-hold control (ZOH)
%     @foh       first-order-hold control (FOH) with rise time
%     @expdecay  lagged control with time constant
%   delay        optional, continuous-time delay, in range [0 dt)
% x1             consequtive state
%
% Copyright (C) 2008-2014 by Carl Edward Rasmussen, 2014-11-04

OPTIONS = odeset('RelTol', 1e-12, 'AbsTol', 1e-12);         % accuracy of ode45
persistent u;                                % previous control action function 
dt = plant.dt; eval(['ctrltype = ', func2str(plant.ctrltype), ';']);   % scope!
ulength = floor(plant.delay/dt) + 1;
if (isempty(u))
    for i=1:ulength
        u{i} = @(t)ctrltype(t, 0*f, 0*f); %TODO: Change. 
    end
end

if isfield(plant, 'delay')
  [T y] = ode45(plant.ode, linspace(dt-plant.delay,dt,3), x0, OPTIONS, u{1});
  udt = u{1}(dt);
  u = u(1:ulength-1);
  u{ulength} = @(t)ctrltype(t, f, udt);  
  [T y] = ode45(plant.ode, linspace(0,dt-plant.delay,3), y(3,:)', OPTIONS, u{1});     
else
  udt = u{1}(dt);
  u{1} = @(t)ctrltype(t, f, udt);  
  [T y] = ode45(plant.ode, [0 dt/2 dt], x0, OPTIONS, u{1});        % solve the ode
end    
x1 = y(3,:)';                                       % extract the desired state 


function u = zoh(t, f, f0)                            % zero order hold control
u = f;

function u = foh(t, f, f0, risetime) % first order hold with specified risetime
if t < risetime, u = f0 + t*(f-f0)/risetime; else u = f; end

function u = expdecay(t, f, f0, tau) % exponential decay with time constant tau
u = f + exp(-t/tau)*(f0-f);
