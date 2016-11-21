function [data, latent, L] = rollout(start, ctrl, H, plant, cost, verb)
% Compute a state trajectory using an ode solver (and any additional dynamics)
% from a particular starting state with either a particular policy or random
% actions.
%
% start        nX x 1  vector containing start state (without controls)
% ctrl                 controller structure
%   fcn        @       function implementing the controller
%   state              internal controller state structure
%   init               initial value for internal controller state structure
%   policy             policy structure
%     fcn      @       policy function
%     p                parameter structure (if empty use random actions)
%     maxU     nU x 1  vector of control input saturation values
% H                    rollout horizon in steps
% plant                the dynamical system structure
%   subplant           (opt) additional discrete-time dynamics
%   augment            (opt) augment state using a known mapping
%   constraint         (opt) stop rollout if violated
%   poli               indices for states passed to the policy
%   dyno               indices for states passed to cost
%   odei               indices for states passed to the ode solver
%   subi               (opt) indices for states passed to subplant function
%   augi               (opt) indices for states passed to augment function
%   angi               (opt) indices for states representing angles 
% cost                 cost structure
% verb                 verbosity level
%
% data                data struct
%   state     H+1xnX  data matrix
%   action    H x nU  action matrix
% L          loss incurred at each timestep (1 by H)
% latent     matrix of latent states (H+1 by nX)
%
% Copyright (C) 2012-2014 by Carl Edward Rasmussen, 2014-05-10

clear simulate;                                   % clear the simulate function
if isfield(ctrl, 'init'), ctrl.state = ctrl.init; end   % init controller state
if isfield(plant,'augment'), augi = plant.augi;             % sort out indices!
else plant.augment = @(x)[]; augi = []; end
if isfield(plant,'subplant'), subi = plant.subi;     % relevant for subdyanmics
else plant.subplant = @(x,y)[]; subi = []; end
if nargin < 6; verb = 0; end
odei = plant.odei; dyno = plant.dyno; angi = plant.angi;
simi = sort([odei subi]);
nX = length(simi)+length(augi); nU = ctrl.U; nA = length(angi);

u = zeros(H, nU); latent = zeros(H+1, nX+nU);
L = zeros(1, H); next = zeros(1,length(simi)); x = zeros(H+1, nX+2*nA);

state(simi) = start; state(augi) = plant.augment(state);      % initialisations
latent(1,1:nX) = state;
x(1,simi) = start' + randn(size(simi))*chol(plant.noise);       % add obs noise
x(1,augi) = plant.augment(x(1,:));             % augment noisy version of state

for i = 1:H % ----------------------------------------------------- run ROLLOUT
  x(i,end-2*nA+1:end) = gTrig(x(i,dyno)', zeros(length(dyno)), angi); % trigaug
  
  % Apply policy ... or random actions ----------------------------------------
  if isfield(ctrl.policy, 'fcn')
    [ctrl.state, u(i,:)] = ctrl.fcn(ctrl, plant, x(i,dyno)');
  else
    u(i,:) = ctrl.policy.maxU.*(2*rand(1,nU)-1);
  end
  latent(i,nX+1:end) = u(i,:);

  % Simulate dynamics ---------------------------------------------------------
  next(odei) = simulate(state(odei), u(i,:), plant);             % run dynamics
  next(subi) = plant.subplant(state, u(i,:));            % addittional dynamics
  next(augi) = plant.augment(next);                        % augment next state
  
  % Test constraints and stop rollout if violated -----------------------------
  if isfield(plant,'constraint') && plant.constraint(next)
    H = i-1; if verb; fprintf('state constraints violated...\n'); end; break;
  end
  
  % Constraints not violated so accept next state
  state = next;                           % update the noise-free, hidden state
  x(i+1,simi) = state(simi) + randn(size(simi))*chol(plant.noise);   % observed
  x(i+1,augi) = plant.augment(x(i+1,:));                  % augment noisy state
  latent(i+1,1:nX) = state;                                      % latent state
  
  % Compute Cost --------------------------------------------------------------
  if nargout > 2;  L(i) = cost.fcn(cost,state(dyno)',zeros(length(dyno))); end
end
if verb; fprintf('\nTrial lasted %i steps\n',floor(H));  end

x(end,end-2*nA+1:end) = gTrig(x(end,dyno)', zeros(length(dyno)), angi);
data.state = x(1:H+1,:); data.action = u(1:H,:);
latent = latent(1:H+1,:); L = L(1,1:H);              % trim any trailing zeros
