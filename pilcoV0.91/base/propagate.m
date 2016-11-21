%% propagate.m
% *Summary:* Propagate the state distribution one time step forward.
%
%  [Mnext, Snext] = propagate(m, s, plant, dynmodel, policy)
%
% *Input arguments:*
%
%   m                 mean of the state distribution at time t           [D x 1]
%   s                 covariance of the state distribution at time t     [D x D]
%   plant             plant structure
%   dynmodel          dynamics model structure
%   policy            policy structure
%
% *Output arguments:*
%
%   Mnext             mean of the successor state at time t+1            [E x 1]
%   Snext             covariance of the successor state at time t+1      [E x E]
%
%
% Copyright (C) 2008-2013 by 
% Marc Deisenroth, Andrew McHutchon, Joe Hall, Henrik Ohlsson,
% and Carl Edward Rasmussen. 
%
% Last modified: 2013-01-23
%
%% High-Level Steps
% # Augment state distribution with trigonometric functions
% # Compute distribution of the control signal
% # Compute dynamics-GP prediction
% # Compute distribution of the next state
%

function [Mnext, Snext] = propagate(m, s, plant, dynmodel, policy)
%% Code

% extract important indices from structures
angi = plant.angi;  % angular indices
poli = plant.poli;  % policy indices
dyni = plant.dyni;  % dynamics-model indices

D0 = length(m);                                        % size of the input mean
D1 = D0 + 2*length(angi);          % length after mapping all angles to sin/cos
D2 = D1 + length(policy.maxU);             % length after computing ctrl signal
D3 = D2 + D0;                                         % length after predicting
M = zeros(D3,1); M(1:D0) = m; S = zeros(D3); S(1:D0,1:D0) = s;   % init M and S

% 1) Augment state distribution with trigonometric functions ------------------
i = 1:D0; j = 1:D0; k = D0+1:D1;
[M(k), S(k,k), C] = gTrig(M(i), S(i,i), angi);        % the latent state
q = S(j,i)*C; S(j,k) = q; S(k,j) = q';

mm = zeros(D1,1); ss = zeros(D1); mm(i) = M(i);       % state which policy uses
ss(i,i) = S(i,i) + diag(exp(2*[dynmodel.hyp.on]));    % has observ. noise added
[mm(k), ss(k,k), C] = gTrig(mm(i), ss(i,i), angi);    % gTrig the noisy state
q = ss(j,i)*C; ss(j,k) = q; ss(k,j) = q';

% 2) Compute distribution of the control signal -------------------------------
i = poli; j = 1:D1; k = D1+1:D2;
[M(k), S(k,k), C] = policy.fcn(policy, mm(i), ss(i,i));
q = S(j,i)*C; S(j,k) = q; S(k,j) = q'; 

% 3) Compute distribution of the change in state ------------------------------
i = [dyni D1+1:D2]; j = 1:D2; k = D2+1:D3; 
[M(k), S(k,k), C] = dynmodel.fcn(dynmodel, M(i), S(i,i));
S(k,k) = S(k,k) + diag(exp(2*[dynmodel.hyp.pn]));          % add process noise
q = S(j,i)*C; S(j,k) = q; S(k,j) = q';

% 4) Select distribution of the next state -----------------------------------
Mnext = M(D2+1:D3); Snext = S(D2+1:D3,D2+1:D3); Snext = (Snext+Snext')/2; 