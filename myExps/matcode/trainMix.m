function dynmodel = trainMix(dynmodel, iter)
% Function to train a mixture of dynamics models
% 
%   dynmodel
%     .inputs     training inputs, these are stored once, all together
%     .target     training targets
%     .hyp        NOT hyperparameters...but a way of getting noise
%     .sub{n}     Nf by 1 cell array of structures
%       .fcn        prediction function
%       .train      training function
%       .dyni       submodel input indices, indexing dynmodel.inputs
%       .dynj       furhter submodel input indices, indexing dynmodel.target
%       .dyno       submodel output indices, elements of 1:length(plant.dyno)
%
% Andrew McHutchon and Joe Hall
% 5th July 2013

Nf=length(dynmodel.sub);

for n=1:Nf
  d = dynmodel.sub{n};                                                 % unwrap
  if isfield(d,'train')
    idx = []; tdx = []; udx = [];
    if isfield(d,'dyni'), idx = d.dyni; end        % input indices for submodel
    if isfield(d,'dynj'), tdx = d.dynj; end          % inputs from prev outputs
    if isfield(d,'dynu'), udx = d.dyni(end)+d.dynu; end          % inputs from control
    d.inputs = [dynmodel.inputs(:,idx) dynmodel.target(:,tdx) dynmodel.inputs(:, udx)];        % inputs
    d.target = dynmodel.target(:,d.dyno);                             % targets
    
    if nargin < 2, d = d.train(d); else d = d.train(d, iter); end       % TRAIN
    
    dynmodel.sub{n} = rmfield(d,{'inputs','target'});                % rewrap
%     if isfield(d,'hyp'), 
%         [dynmodel(d.dyno).hyp.s] = deal(d.hyp.s); %%It's a bug!?
%         [dynmodel(d.dyno).hyp.n] = deal(d.hyp.n);
%     end
    if isfield(d,'hyp'), 
        [dynmodel.hyp(d.dyno).s] = deal(d.hyp.s); 
        [dynmodel.hyp(d.dyno).n] = deal(d.hyp.n);
        [dynmodel.hyp(d.dyno).on] = deal(d.hyp.on);
        [dynmodel.hyp(d.dyno).pn] = deal(d.hyp.pn);
        [dynmodel.noise(d.dyno)] = deal(d.noise);
    end
  end
end
