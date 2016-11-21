function [M S dynmodel] = gpJoint(dynmodel,xs)
% Function to make a gpr prediction for a test point. Specifically this
% function allows pre-computed matricies to be used.
% 
% Inputs
%   dynmodel        Struct with fields:
%       .inputs     Input training data
%       .target     Target training data
%       .hyp        GP hyperparameters
%       .beta       (K + sn2*I)^-1 * target
%       .R          chol(K + sn2*I), Nt-by-Nt
%   xs              input test point, Ns-by-D
%   yc              targets to condition on Ns-1-by-E
%
% Outputs
%   M              Ouput test mean, Ns-by-E
%   S              Output test variance, Ns-by-E
%
% Andrew McHutchon, 21/02/2011

% Handle other configurations
if isfield(dynmodel,'inp'); dynmodel.inputs = dynmodel.inp; end
if isfield(dynmodel,'tar'); dynmodel.target = dynmodel.tar; end
if nargin < 3; yc = []; end

if any(isnan(xs(:))) || any(isnan(yc(:))); keyboard; end

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
    XmX = permute(bsxfun(@minus,permute(dynmodel.inputs,[1,3,2]),permute(xs,[3,1,2])),[1,2,4,3]); % N-by-Ns-by-1-by-D
    XmxiLam = bsxfun(@times,XmX,permute(iell.^2,[4,3,2,1]));     % N-by-Ns-by-E-by-D
    dKsdxs = bsxfun(@times,XmxiLam,Ks);                      % N-by-Ns-by-E-by-D
end

% The joint posterior mean
M = etprod('12',Ks,'312',dynmodel.beta,'32');                          % Ns-by-E
if nargout > 3
    dMdxs = etprod('123',dKsdxs,'4123',dynmodel.beta,'42');       % Ns-by-E-by-D
end

% The joint posterior covariance matrix
Kss = zeros(Ns,Ns,E); iKKs = zeros(Nt,Ns,E);
for i=1:E
    inp = bsxfun(@times,xs,iell(:,i)');
    Kss(:,:,i) = exp(2*hyp(D+1,i)-maha(inp,inp)/2);      % Ns-by-Ns-by-E
    iKKs(:,:,i) = solve_chol(dynmodel.R(:,:,i),Ks(:,:,i));
end
S = Kss - etprod('124',Ks,'314',iKKs,'324');             % Ns-by-Ns-by-E