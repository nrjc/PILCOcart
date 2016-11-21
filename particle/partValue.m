function [f, df, dynmodel] = partValue(policy, m, s, dynmodel, plant, cost, H)

% compute expected (discounted) cumulative cost for a given set of state distributions
%
% policy      (struct) for (lin. in param.) policy to be implemented
%   policy.fcn          policy function (linear/GP/...)
%   policy.w            policy weights
%   policy.p            policy parameters
%   policy.hyp          policy hyper-parameters (if exist)
%   policy.maxU         amplitude of allowable forces/torques
% m0          means of states
% S0          (shared) covariance matrix of state distributions
% dynmodel    (struct) for dynamics model (GP)
%   dynmodel.fcn        function used for GP predictions (e.g.: gpP1/gpP1d)
%   dynmodel.w          training targets (GP)
%   dynmodel.p          training inputs (GP)
%   dynmodel.hyp        GP log hyperparameters
% propFct     function name being used to propagate uncertainty
% cost        (struct) for cost function
%   cost.fcn            cost function (loss/lossq)
%   cost.p              cost parameters
%   cost.width          vector of standard deviations of immediate cost 
%                       -> scale mixture
%   cost.angle          array of angle indices
%   cost.gamma          discount factor
% H           length of prediction horizon
%
% f            expected cumulative (discounted) cost
% df           derivative of f wrt policy parameters
%
% oldx  dyni+du-by-t-by-Nsamp
%
% Copyright (C) 2008-2010 Marc Deisenroth & Carl Edward Rasmussen 2010-11-19
% Changed for particles: Andrew McHutchon, 30/6/2011

% Setup variables
if ~isfield(plant,'seed'); 
    plant.seed = 1;
end
if ~isfield(plant,'rStream'); 
    plant.rStream = RandStream.create('shr3cong','seed',plant.seed);
end
reset(plant.rStream); oldx = []; oldy = []; doldudm = []; doldudp = [];
Hc = plant.Hc;
L0 = 0; L = zeros(1,H); dp = 0;

if nargout <= 1                   % no derivatives required   
    for t = 1:H                     % for all time steps in horizon
        [X,~,~,dynmodel,oldx,oldy] = partProp(m, s, plant, dynmodel, policy,oldx,oldy); % get next state
        if Hc > 0
            oldx = oldx(:,max(end-Hc+1,1):end,:);
            oldy = oldy(:,max(end-Hc+1,1):end,:);
        end
        
        L(t) = cost.gamma^t*partLoss(cost,X);
        if isnan(L(t)); keyboard; end
        
        m = X; s = [];
    end
    
else % get derivatives
   
    for t = 1:H                                  % for all time steps in horizon
        % propagate samples
	
        [X, dXdm, dXdp, dynmodel, ~, oldx, oldy, doldudm, doldudp, dXdoldy] = ...
                     partPropd(m, s, plant, dynmodel, policy,oldx,oldy,doldudm,doldudp);
      
        if t > 1
	      if Hc > 0
            dXdm(:,:,2:end,:) = dXdm(:,:,2:end,:) + dXdoldy;
            dXdm(:,:,1:end-1,:) = dXdm(:,:,1:end-1,:) - dXdoldy;
            if t <= Hc+1; dXdm = dXdm(:,:,2:end,:); end % no derivative w.r.t. initial state
 	      end
            dXdp = etprod('123',dXdm,'1245',dmdp,'5243') + dXdp; % E-by-Nsamp-by-Np	
        end

              
        % Find loss
        [L(t) dLdX] = partLoss(cost, X);
        L(t) = cost.gamma^t*L(t);                         % discounting
        
        dp = dp + cost.gamma^t*etprod('1',dLdX,'23',dXdp,'231');
               
        % Update
        oldx = oldx(:,max(end-Hc+1,1):end,:);                 % D-by-Hc-by-Nsamp
        oldy = oldy(:,max(end-Hc+1,1):end,:);                 % E-by-Hc-by-Nsamp
        doldudm = doldudm(:,max(end-Hc+1,1):end,:,:);   % Du-by-Hc-by-Nsamp-by-D
        doldudp = doldudp(:,max(end-Hc+1,1):end,:,:);  % Du-by-Hc-by-Nsamp-by-Np

        m = X; s = [];                  % no longer a distribution - now samples
        if t > 1
            dmdp = cat(3,dmdp(:,:,max(end-Hc+1,1):end,:),permute(dXdp,[1,2,4,3])); % E-by-Nsamp-by-2-by-Np
        else
            dmdp = permute(dXdp,[1,2,4,3]);
        end

    end
end

f = L0 + sum(L);
if nargout == 2; df = rewrap(policy,dp); end
