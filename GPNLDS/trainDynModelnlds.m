% Script to train and display dynamics model using the GPNLDS model

if ~ishandle(6); figure(6); else set(0,'CurrentFigure',6); end; clf(6);
if ~isfield(dynmodel,'hyp') || j < 15; reinit = 2; else reinit = 1; end
dynmodel = trainGPNLDS(data, dynmodel, noplant, reinit);    % train dynamics GP

h = dynmodel.hyp;                                     % display hyperparameters
dynmodel.oldh = unwrap(h); dynmodel.oldn = size(dynmodel.beta,1);
disptable([exp([h.pn]); exp([h.on]); sqrt(exp(2*[h.pn])+exp(2*[h.on])); exp([h.s])./sqrt(exp(2*[h.pn])+exp(2*[h.on]))],...
            varNames,'process noises: |observ. noises: |total noises: |SNRs: ')
