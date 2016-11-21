function [d,dy,dh] = lossT(s, dyn, ctrl, cost, H, expl, cc_prev, n, delta)
%% Brief Description and Interface
% Summary: Test derivatives of the loss function
%
%   [d, dy, dh] = lossT(s, dyn, ctrl, cost, H, expl, cc_prev, n, delta)
%
% INPUTS:
%
%   s          state structure
%   dyn        GP dynamics model object
%   ctrl       controller object
%     p        policy parameters (can be a structure)
%   cost       cost structure
%   H          prediction horizon. Default: 4
%   expl       exploration struct
%   cc_prev    cumulative (discounted) cost structure of previous rollout
%   n          number of trials left
%   delta      finite difference parameter. Default: 1e-4
%
% OUTPUTS:
%
%   dd         relative error of analytical vs. finite difference gradient
%   dy         analytical gradient
%   dh         finite difference gradient
%
% Copyright (C) 2008-2015 by Marc Deisenroth, Andrew McHutchon, Joe Hall,
% Carl Edward Rasmussen, Rowan McAllister 2016-05-10

plant = create_test_object('plant', 'cartDoublePendulum');
if ~exist('dyn'    ,'var'); dyn = create_test_object('dyn', plant);         end
if ~exist('ctrl'   ,'var'); ctrl = create_test_object('CtrlNF', plant, dyn);end
if ~exist('s'      ,'var'); s = create_test_object('state', plant, ctrl);   end
if ~exist('cost'   ,'var'); cost = create_test_object('cost', plant);       end
if ~exist('H'      ,'var'); H = 4;                                          end
%if~exist('expl'   ,'var'); expl = [];                                      end
if ~exist('expl'   ,'var'); expl = create_test_object('expl');              end
if ~exist('n'      ,'var'); n = 3;                                          end
if ~exist('cc_prev','var'); cc_prev = [];                                   end
if ~exist('delta'  ,'var'); delta = 1e-4;                                   end

% call checkgrad directly
[d,dy,dh] = checkgrad('loss',ctrl.policy.p,delta,s,dyn,ctrl,cost,H,expl,...
  cc_prev,n);
