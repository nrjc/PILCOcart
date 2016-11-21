function [state, M, S] = propagate(m, s, plant, dynmodel, ctrl)

% Propagate the state distribution one time step forward.
%
% Copyright (C) 2008-2014 by Marc Deisenroth, Carl Edward Rasmussen, Henrik
% Ohlsson, Andrew McHuthon and Joe Hall, 2014-04-04

angi = plant.angi; poli = plant.poli; dyni = plant.dyni;

D0 = length(m);                                        % size of the input mean
D1 = D0 + 2*length(angi);          % length after mapping all angles to sin/cos
D2 = D1 + ctrl.U;                          % length after computing ctrl signal
D3 = D2 + D0;                                         % length after predicting
M = zeros(D3,1); M(1:D0) = m; S = zeros(D3); S(1:D0,1:D0) = s;   % init M and S

% 1) Augment state distribution with trigonometric functions ------------------
i = 1:D0; j = 1:D0; k = D0+1:D1;
[M(k), S(k,k), C] = gTrig(M(i), S(i,i), angi);               % the latent state
q = S(j,i)*C; S(j,k) = q; S(k,j) = q';

% 2) Compute distribution of the control signal -------------------------------
i = 1:D0; j = 1:D1; k = D1+1:D2;
[state, M(k), S(k,k), C] = ctrl.fcn(ctrl, plant, M(i), ...
                                        S(i,i)+diag(exp(2*[dynmodel.hyp.on])));
q = S(j,i)*C; S(j,k) = q; S(k,j) = q'; 

% 3) Compute distribution of the next state -----------------------------------
ii = [dyni D1+1:D2]; j = 1:D2;
if isfield(dynmodel,'sub'), Nf = length(dynmodel.sub); else Nf = 1; end
for n=1:Nf                               % potentially multiple dynamics models
  [dyn, i, k] = sliceModel(dynmodel,n,ii,D1,D2,D3); j = setdiff(j,k);
  [M(k), S(k,k), C] = dyn.fcn(dyn, M(i), S(i,i));
  S(k,k) = S(k,k) + diag(exp(2*[dyn.hyp.pn]));              % add process noise
  q = S(j,i)*C; S(j,k) = q; S(k,j) = q';
  
  j = [j k];                                   % update 'previous' state vector
end

% 4) Select distribution of the next state -----------------------------------
M = M(D2+1:D3); S = S(D2+1:D3,D2+1:D3); S = (S+S')/2;


% A1) Separate multiple dynamics models ---------------------------------------
function [dyn, i, k] = sliceModel(dynmodel,n,ii,D1,D2,D3) % separate sub-dynamics
if isfield(dynmodel,'sub')
  dyn = dynmodel.sub{n}; do = dyn.dyno; D = length(ii)+D1-D2;
  if isfield(dyn,'dyni'), di=dyn.dyni; else di=[]; end
  if isfield(dyn,'dynu'), du=dyn.dynu; else du=[]; end
  if isfield(dyn,'dynj'), dj=dyn.dynj; else dj=[]; end
  i = [ii(di) D1+du D2+dj]; k = D2+do;
  dyn.inputs = [dynmodel.inputs(:,[di D+du]) dynmodel.target(:,dj)];   % inputs
  dyn.target = dynmodel.target(:,do);                                 % targets
else
  dyn = dynmodel; k = D2+1:D3; i = ii;
end