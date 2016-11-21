function dynmodel = trainGPNLDS(data, dynmodel, plant, reinit)
% function to train a GP dynamical model for the time series data stored in the
% struct array data. This version uses the 'direct' method.
%       
% data            .     struct array of observed data trials
%   state      Ti x nX
%   action
% dynmodel        .
%   inputs              pseudo inputs
%   induce     Np x D   pseudo training inputs
%   beta                K^-1 * y, where y is the pseudo targets
%   hyp           .     struct of hyperparameters
%     l         D x 1   log lengthscales
%     s           s     log signal standard deviation
%     n           s     log noise standard dev on pseudo targets
%     pn          s     log process noise standard deviation
%     on          s     log observation noise standard deviation
%     m         D x 1   GP prior mean linear weights
%     b           s     GP prior mean bias, 1-by-1
%   trainMean           switch for GP prior mean, 0: fixed, 1: optimised
%   opt           s     options struct for minimize
% plant           s     struct containing variable indices, in particular:
%   dyno        1 x E   the variables to use for our model
%   angi                subset of dyno which have been augmented with sin & cos
% reinit                switch 0: no init, 1: just pseudo, 2: pseudo and hypers
%
% Copyright (C) 2014 by Andrew McHutchon and Carl Edward Rasmussen 2014-10-08

E = length(plant.dyno); Da = length(plant.angi);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 0) The following (long) section of code handles initialising the NLDS GP
if nargin < 3 || reinit > 0
  fprintf('Running initial training\n');                     % Initial training
  dyn.inputs = []; dyn.target = []; 
  dyn.induce = zeros(size(dynmodel.induce,1),0);
  x = cell2mat(arrayfun(@(Y)Y.state(1:end-1,:),data,'uniformoutput',0)');
  y = cell2mat(arrayfun(@(Y)Y.state(2:end,:),data,'uniformoutput',0)');
  u = cell2mat(arrayfun(@(U)U.action,data,'uniformoutput',0)');
  x = x(:,[plant.dyno plant.oldu end-2*Da+1:end]); % the dyno and trigaug vars
  x = x(:,plant.dyni);
  dyn.inputs = [x u]; dyn.target = y(:,plant.dyno);
  if isfield(dynmodel,'hyp')&&isfield(dynmodel.hyp,'m')
    [dyn.hyp(1:E).m] = deal(dynmodel.hyp.m);            % GP prior mean must be 
    [dyn.hyp.b] = deal(dynmodel.hyp.b);              % set before calling train
  end
  dyn = train(dyn,[-500,-500]);
end

if nargin < 3 || 2 == reinit     % ------------ (Re)initialise parameter struct
  p = dyn.hyp;                           % Initialise hypers from initial AR GP
  % Initial GP just has observation noise, we initialise the NLDS GP by
  % splitting the variance equally into observation and process noise
  on = num2cell([p.on] - log(2)/2); 
  [p.on] = deal(on{:}); [p.pn] = deal(on{:});
  n = num2cell([dyn.hyp.s] - log(300)); [p.n] = deal(n{:});
  dynmodel.hyp = p;
else
  p = dynmodel.hyp;
end

if nargin < 3 || 2 == reinit      % ------------ Transition model pseudo inputs
  if size(dyn.induce,2)==0;
    inputs = dyn.inputs;                                       % all dyn inputs
   else
     inputs = dyn.induce;
   end                                                              % from FITC
else
  inputs = dynmodel.inputs;
end

% --------------------------------------------- Transition model pseudo targets
if 1 == reinit;
  y = gpPred(dynmodel,inputs);                          % Use previous dynmodel
elseif 2 == reinit;
  y = gpPred(dyn,inputs);                              % Use AR GP just trained
end
if nargin < 3 || reinit > 0      % ---------------------------- Initialise beta
  dynmodel.inputs = inputs;                                     % Assign inputs
  dynmodel = preComp(dynmodel);                            % calculate K, R, iK
  for i=1:E
    p(i).beta = solve_chol(dynmodel.R(:,:,i),y(:,i)-inputs*p(i).m-p(i).b);
  end
else
  for i=1:E; p(i).beta = dynmodel.beta(:,i); end % copy beta from dynmodel
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------------------------ Now the real work begins ------------------------- %
p = minimize(p,@pCurbD,dynmodel.opt,dynmodel,plant,data);   % optimize dynmodel
dynmodel.beta = [p.beta]; [dynmodel.hyp] = deal(rmfield(p,'beta'));
dynmodel = preComp(dynmodel);                           % recompute beta and iK
dynmodel = rmfield(dynmodel,{'K','R','Kclean'});
