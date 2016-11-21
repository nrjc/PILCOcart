function dynmodel = preCalcDyn(dynmodel)

% Dimensions
[Nt E] = size(dynmodel.target);
D = size(dynmodel.inputs,2);

% Hyperparameters
hyp = dynmodel.hyp; 
iell = exp(-hyp(1:D,:)); % D-by-E

% Calculate pre-computable matrices if necessary
if ~isfield(dynmodel,'R')
    dynmodel.R = zeros(Nt,Nt,E);
    for i=1:E
        inp = bsxfun(@times,dynmodel.inputs,iell(:,i)');
        K = exp(2*hyp(D+1,i)-maha(inp,inp)/2) + exp(2*hyp(end,i))*eye(Nt);
        if isfield(dynmodel,'noise')
            K = K + diag(dynmodel.noise(:,i));
        end
        dynmodel.R(:,:,i) = chol(K);
    end
end     
    
if ~isfield(dynmodel,'beta')
    dynmodel.beta = zeros(Nt,E);
    for i=1:E
        dynmodel.beta(:,i) = solve_chol(dynmodel.R(:,:,i),dynmodel.target(:,i));
    end
end