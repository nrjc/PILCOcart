function [M, S] = gpPred(dynmodel,xs,incnoise)
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

% Dimensions
Ns = size(xs,1);
if isfield(dynmodel,'beta'); [Nt, E] = size(dynmodel.beta);
else [Nt, E] = size(dynmodel.target); end
D = size(dynmodel.inputs,2);
if size(xs,2) ~= D; 
    error('Incorrect 2nd dimension of xs: should be %i, is %i\n',D,size(xs,2)); 
end


% Hyperparameters
h = dynmodel.hyp; 
sf2 = exp(2*[h.s]); 
iell = exp(-[h.l]); iL = exp(-2*[h.l]); % D-by-E
sn2 = exp(2*[h.n]);
if isfield(h,'m'); A = [h.m]; else A = zeros(D,E); end
if isfield(h,'b'); b = [h.b]; else b = zeros(1,E); end

% Calculate pre-computable matrices if necessary
if ~isfield(dynmodel,'iK') && ~isfield(dynmodel,'R')
    dynmodel.R = zeros(Nt,Nt,E);
    for i=1:E
        inp = bsxfun(@times,dynmodel.inputs,iell(:,i)');
        K = exp(2*h(i).s-maha(inp,inp)/2) + sn2(i)*eye(Nt);
        dynmodel.R(:,:,i) = chol(K);
    end
end     
    
if ~isfield(dynmodel,'beta')
    dynmodel.beta = zeros(Nt,E);
    for i=1:E
        y = dynmodel.target(:,i) - dynmodel.inputs*A(:,i) - b(i);
        dynmodel.beta(:,i) = solve_chol(dynmodel.R(:,:,i),y);
    end
end

% Find the test point covariance matrices
if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce);
    x = dynmodel.induce; else x = dynmodel.inputs; end
Xmx2 = reshape(bsxfun(@minus,permute(x,[1,3,2]),permute(xs,[3,1,2])).^2,Nt*Ns,D); % N*Ns-by-D
Ks = reshape(exp(bsxfun(@plus,-0.5*Xmx2*iL,2*[h.s])),Nt,Ns,E);  % N-by-Ns-by-E
                                 
% Calculate posterior mean 
M = etprod('12',Ks,'312',dynmodel.beta,'32');     % Ns-by-E
M = bsxfun(@plus,M + xs*A,b);                              

% The posterior variance
if nargout > 1
    iKKs = zeros(Nt,Ns,E);
    if isfield(dynmodel,'iK')
        for i=1:E; iKKs(:,:,i) = dynmodel.iK(:,:,i)*Ks(:,:,i); end
    else
        for i=1:E; iKKs(:,:,i) = solve_chol(dynmodel.R(:,:,i),Ks(:,:,i)); end
    end
    KsiKKs = etprod('12',Ks,'312',iKKs,'312');
    S = bsxfun(@minus,sf2,KsiKKs);    % Ns-by-E
    
    if 3==nargin && incnoise; S = bsxfun(@plus,S,sn2); end                          
end
        