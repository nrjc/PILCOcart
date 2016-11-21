% A script version of animate (which displays rollout trajectories), which 
% instead uses current scope variables to call the animate function.
%
% See also animate.m

assert(exist('latent','var') && exist('data','var'), ...
  [mfilename,': latent and data must exist.'])

% Animate inputs: animate(latent, data, dt, cost, b, movname, movtype)
if ~exist('dt','var')
  animate(latent, data);
elseif ~exist('cost','var')
  animate(latent, data, dt)
else
  animate(latent, data, dt, cost)
end

