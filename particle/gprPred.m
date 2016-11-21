function [M S dynmodel dMdxs dSdxs] = gprPred(dynmodel,xs,yc)
% Function to make a gpr prediction for a test point. Specifically this
% function allows pre-computed matricies to be used.
% 
% Inputs
%   dynmodel        Struct with fields:
%       .inputs     Input training data
%       .target     Target training data
%       .hyp        GP hyperparameters
%       .beta       (K + sn2*I)^-1 * target
%       .iK         (K + sn2*I)^-1, Nt-by-Nt
%   xs              input test point, Ns-by-D
%
% Outputs
%   M               Ouput test prediction, Ns-by-E
%   S               Output test variance, Ns-by-E
%
% Andrew McHutchon, 21/02/2011

% Handle other configurations
if isfield(dynmodel,'inp'); dynmodel.inputs = dynmodel.inp; end
if isfield(dynmodel,'tar'); dynmodel.target = dynmodel.tar; end

% Dimensions
Ns = size(xs,1);
[Nt E] = size(dynmodel.target);
D = size(dynmodel.inputs,2);

% Hyperparameters
hyp = dynmodel.hyp; 
sf2 = exp(2*hyp(end-1,:)); 
iell = exp(-hyp(1:D,:)); % D-by-E

% Calculate pre-computable matrices if necessary
if ~isfield(dynmodel,'R')
    dynmodel.R = zeros(Nt,Nt,E);
    for i=1:E
        inp = bsxfun(@times,dynmodel.inputs,iell(:,i)');
        K = exp(2*hyp(D+1,i)-maha(inp,inp)/2) + exp(2*hyp(end,i))*eye(Nt);
        dynmodel.R(:,:,i) = chol(K);
    end
end     
    
if ~isfield(dynmodel,'beta')
    dynmodel.beta = zeros(Nt,E);
    for i=1:E
        dynmodel.beta(:,i) = solve_chol(dynmodel.R(:,:,i),dynmodel.target(:,i));
    end
end

% Find the test point covariance matrices
Ks = zeros(Nt,Ns,E);
for i=1:E
    Ks(:,:,i) = sf2(i)*exp(-sq_dist(diag(iell(:,i))*dynmodel.inputs',diag(iell(:,i))*xs')/2);
end
if nargout >3
    Xmx = permute(bsxfun(@minus,permute(dynmodel.inputs,[1,3,2]),permute(xs,[3,1,2])),[1,2,4,3]); % N-by-Ns-by-1-by-D
    XmxiLam = bsxfun(@times,Xmx,permute(iell.^2,[4,3,2,1]));     % N-by-Ns-by-E-by-D
    dKsdxs = bsxfun(@times,XmxiLam,Ks);                      % N-by-Ns-by-E-by-D
end

% Calculate posterior mean 
M = etprod('12',Ks,'312',dynmodel.beta,'32');     % Ns-by-E
if nargout > 3
    dMdxs = etprod('123',dKsdxs,'4123',dynmodel.beta,'42');       % Ns-by-E-by-D
end

% The posterior variance
if nargout > 1
    iKKs = zeros(Nt,Ns,E);
    for i=1:E; iKKs(:,:,i) = solve_chol(dynmodel.R(:,:,i),Ks(:,:,i)); end 
    Kss = repmat(sf2,Ns,1); % Ns-by-E
    S = Kss - etprod('12',Ks,'312',iKKs,'312');  % Ns-by-E
    
    if nargout > 3
        dSdxs = -2*etprod('123',dKsdxs,'4123',iKKs,'412'); % Ns-by-E-by-D
    end
end
        

