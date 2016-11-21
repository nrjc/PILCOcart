function [f, df] = valuec(policy, m0, S0, dynmodel, plant, cost, H)

% compute expected (discounted) cumulative cost for a given (set of) initial
% state distributions
%
% policy        policy structure
%   policy.fcn  function which implements the policy
% m0            matrix (D by k) of initial state means
% S0            covariance matrix (D by D) for initial state
% dynmodel      dynamics model structure
% plant         plant structure
% cost          cost function structure
%   cost.fcn    function implementing cost
%   cost.gamma  discount factor
% H             length of prediction horizon
%
% f             expected cumulative (discounted) cost
% df            derivative struct of f wrt policy
%
% Copyright (C) 2008-2011 Marc Deisenroth & Carl Edward Rasmussen, 2011-12-06

p = unwrap(policy); dp = 0*p; L0 = 0; Lp = 0; dLp = 0;
E = size(dynmodel.target,2);

for k = 1:size(m0,2);                                    % for all start states  
  m = m0(:,k); S = S0; L = zeros(1,H);
  
  if nargout <= 1                                     % no derivatives required

    for t = 1:H                                 % for all time steps in horizon
      [m, S, dynmodel] = plant.prop(m, S, plant, dynmodel, policy); % get next state
      D = length(m); i = D-E+1:D;
      L(t) = cost.gamma^t.*cost.fcn(m(i), S(i,i), cost); % expected discounted cost
    end
    
  else                                             % otherwise, get derivatives

    dmOlddp = zeros([E, size(p)]);
    dSOlddp = zeros([E, E, size(p)]);
    
    for t = 1:H                                 % for all time steps in horizon      
      [m, S, dynmodel, dmdmOld, dSdmOld, dmdSOld, dSdSOld, mdp, Sdp] = ...
                    plant.prop(m, S, plant, dynmodel, policy); % get next state
      
      dmdp = etprod('134',dmdmOld,'12',dmOlddp,'234') ...    % some derivatives
                            + etprod('145',dmdSOld,'123',dSOlddp,'2345') + mdp;
      dSdp = etprod('1245',dSdmOld,'123',dmOlddp,'345') ...
             + etprod('1256',dSdSOld,'1234',dSOlddp,'3456') + Sdp;
      
      D = length(m); i = D-E+1:D;
      [L(t), dLdm, dLdS] = cost.fcn(m(i), S(i,i), cost);      % predictive cost
      L(t) = cost.gamma^t*L(t);                                      % discount
      dp = dp + cost.gamma^t * (etprod('23',dLdm(:),'1',dmdp(i,:),'123') + ...            
                    etprod('34',dLdS,'12',dSdp(i,i,:),'1234')); % discount derivatives
        
      dmOlddp = dmdp; dSOlddp = dSdp;                             % bookkeeping
    end
    
  end
  L0 = L0 + sum(L);                                                % accumulate

end

L0 = L0./size(m0,2); dp = dp/size(m0,2); % normalize

if isfield(policy,'regulate'); [Lp dLp] = policy.regulate(policy); end
    
f =  L0 + Lp; df = rewrap(p, dp + dLp);                  
