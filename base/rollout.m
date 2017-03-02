function [data, latent2, L] = rollout(start, ctrl, H, plant, cost, verb)
% Compute a state trajectory using an ode solver (and any additional dynamics)
% from a particular starting state with either a particular policy or random
% actions.
%
% [data, latent, L] = rollout(start, ctrl, H, plant, cost, verb)
%
% start        nX x 1  vector containing start state (without controls)
% ctrl                 controller structure
%   fcn        @       function implementing the controller
%   init       @       function initialising controller's filtered state
%   policy             policy structure
%     fcn      @       policy function
%     p                parameter structure (if empty use random actions)
%     maxU     nU x 1  vector of control input saturation values
% H                    rollout horizon in steps
% plant                the dynamical system structure
%   augi               (opt) indices for states passed to augment function
%   augment            (opt) augment state using a known mapping
%   constraint         (opt) stop rollout if violated
%   dyno               indices for states passed to cost
%   noise              observation noise
%   odei               indices for states passed to the ode solver
%   poli               indices for states passed to the policy
% cost                 cost object
% verb                 verbosity level
%
% data                data struct
%   state     H+1xnX  state matrix
%   action    H x nU  action matrix
% L          loss incurred at each timestep (1 by H+1)
% latent     matrix of latent states (H+1 by nX)
%
% Copyright (C) 2012-2016 Carl Edward Rasmussen and Rowan McAllister 2016-03-04

global currT; currT=1;
clear odestep;                           % clear persistent old action function
if isfield(plant,'augment'), augi = plant.augi;              % sort out indices
else plant.augment = @(x)[]; augi = []; end
calc_loss = nargout > 2;
if nargin < 6; verb = 0; end
D = ctrl.D; E = ctrl.E; F = ctrl.F; U = ctrl.U;
N = length(start); odei = plant.odei;
latent = NaN(H+1, N); y = NaN(H+1, N); L = zeros(1, H+1); u = zeros(H, U);
obs_noise = @()(randn(1,E)*chol(plant.noise));

latent(1,:) = start;                                               % initialise
y(1,1:D-E) = latent(1,1:D-E);
y(1,D-E+1:D) = latent(1,D-E+1:D) + obs_noise();     % add noise to observations
if (calc_loss)
    L(1) = cost.fcn(struct('m',latent(1,1:D)')).m; 
end

s.m = y(1,1:D)'; s = ctrl.reset_filter(s);             % reset filter if exists

for i = 1:H   % --------------------------------------------------- run ROLLOUT
  % Test constraints and stop rollout if violated -----------------------------
  if isfield(plant,'constraint') && plant.constraint(latent(i,:))
    H = i-1; if verb; disp('state constraints violated...'); end; break;
  end
  
  % Apply policy --------------------------------------------------------------
  s.m(1:D) = y(i,1:D)'; % receive an observation
  [uzm,~,~,s] = ctrl.fcn(s);
  u(i,:) = uzm(1:U); % action 'u' component of uzm
  s.m(D+1:F) = uzm(U+1:end); % predicted filter 'zm' component of uzm
  
  latent_tmp = [latent(i,1:D), u(i,:)];              % copy for N-Markov states
  latent(i+1,:) = augment(odestep(latent(i,odei), u(i,:), plant), plant);
  latent(i+1,1:D-E) = latent_tmp(end-D+E+1:end);
  
  y_tmp = [y(i,1:D), u(i,:), latent(i+1,D-E+1:D) + obs_noise()];
  y(i+1,1:D) = y_tmp(end-D+1:end);                   % TODO: add process noise?
  
  % Compute Cost --------------------------------------------------------------
  if calc_loss
      L(i+1) = cost.fcn(struct('m',latent(i+1,1:D)')).m; 
  end
end
if verb; disp(['Trial lasted ',num2str(floor(H)),' steps']); end

data.state = y(1:H+1,:); data.action = u(1:H,:);
latent2.state = latent(1:H+1,:); L = L(1,1:H+1);             % trim any trailing zeros

%data.state = latent2.state; % HERE!

function xa = augment(x, plant)
xa=nan(1,max([plant.odei,plant.augi]));
xa(plant.odei) = x;
xa(plant.augi) = plant.augment(xa);
