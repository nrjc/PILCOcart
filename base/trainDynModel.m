% Script to learn the dynamics model
%
% (C) Copyright 2010-2014 by Carl Edward Rasmussen, Marc Deisenroth and
%                                                  Andrew McHutchon, 2014-04-23

x = cell2mat(arrayfun(@(Y)Y.state(1:end-1,:),data,'uniformoutput',0)');
y = cell2mat(arrayfun(@(Y)Y.state(2:end,:),data,'uniformoutput',0)');
u = cell2mat(arrayfun(@(U)U.action,data,'uniformoutput',0)');
x = x(:,[plant.dyno end-2*length(plant.angi)+1:end]);   % dyno and trigaug vars
dynmodel.inputs = [x(:,plant.dyni) u]; dynmodel.target = y(:,plant.dyno);

dynmodel = dynmodel.train(dynmodel, dynmodel.opt);          % train dynamics GP

h = dynmodel.hyp;                                     % display hyperparameters
disptable([exp([h.pn]); exp([h.on]); sqrt(dynmodel.noise); ...
                               exp([h.s])./sqrt(dynmodel.noise)], varNames, ...
                   ['process noises: |observ. noises: |total noises: |SNRs: '])
