function [M S dynmodel dMdxs dSdxs] = fitcPred(dynmodel,xs,yc)
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
[np pD pE] = size(dynmodel.induce);     % number of pseudo inputs per dimension

% Hyperparameters
hyp = dynmodel.hyp; 
sf2 = exp(2*hyp(end-1,:)); sn2 = exp(2*hyp(end,:));
iell = exp(-hyp(1:D,:)); % D-by-E
x = dynmodel.inputs; y = dynmodel.target; u = dynmodel.induce;
ridge = 1e-6;

% Calculate pre-computable matrices if necessary
if ~isfield(dynmodel,'beta')
 iK = zeros(np,Nt,E); dynmodel.iK2 = zeros(np,np,E); dynmodel.beta = zeros(np,E);
    
 for i=1:E
    pinp = bsxfun(@times,u(:,:,min(i,pE)),iell(:,i)');
    inp = bsxfun(@times,x,iell(:,i)');
    Kmm = exp(2*hyp(D+1,i)-maha(pinp,pinp)/2) + ridge*eye(np);  % add small ridge
    Kmn = exp(2*hyp(D+1,i)-maha(pinp,inp)/2);
    L = chol(Kmm)';
    V = L\Kmn;                                             % inv(sqrt(Kmm))*Kmn
    if isfield(dynmodel,'noise')
      G = sqrt(1+(sf2(i)-sum(V.^2)+dynmodel.noise(:,i)')/sn2(i));
    else
      G = sqrt(1+(sf2(i)-sum(V.^2))/sn2(i));
    end
    V = bsxfun(@rdivide,V,G);
    Am = chol(sn2(i)*eye(np) + V*V')';
    At = L*Am;                                    % chol(sig*B) [thesis, p. 40]
    iAt = At\eye(np);
% The following is not an inverse matrix, but we'll treat it as such: multiply
% the targets from right and the cross-covariances left to get predictive mean.
    iK(:,:,i) = ((Am\(bsxfun(@rdivide,V,G)))'*iAt)';
    dynmodel.beta(:,i) = iK(:,:,i)*y(:,i);      
    iB = iAt'*iAt*sn2(i);              % inv(B), [Ed's thesis, p. 40]
    dynmodel.iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
  end
end

% Find the test point covariance matrices
Ks = zeros(np,Ns,E); Xmx = zeros(np,Ns,E,D);
for i=1:E
    j = min(size(u,3),i);
    Ks(:,:,i) = sf2(i)*exp(-sq_dist(diag(iell(:,i))*u(:,:,j)',diag(iell(:,i))*xs')/2);
    if nargout >3
        Xmx(:,:,i,:) = bsxfun(@minus,permute(u(:,:,j),[1,4,3,2]),permute(xs,[3,1,4,2])); % np-by-Ns-by-E-by-D
    end
end
if nargout >3
    XmxiLam = bsxfun(@times,Xmx,permute(iell.^2,[4,3,2,1]));     % N-by-Ns-by-E-by-D
    dKsdxs = bsxfun(@times,XmxiLam,Ks);                      % N-by-Ns-by-E-by-D
end

% Calculate posterior mean 
M = etprod('12',Ks,'312',dynmodel.beta,'32');     % Ns-by-E
if nargout > 3
    dMdxs = etprod('123',dKsdxs,'4123',dynmodel.beta,'42');       % Ns-by-E-by-D
end

% Posterior variance
if nargout > 1
    iKKs = zeros(np,Ns,E);
    for i=1:E; iKKs(:,:,i) = dynmodel.iK2(:,:,i)*Ks(:,:,i); end
     Kss = repmat(sf2,Ns,1); % Ns-by-E
     S = Kss - etprod('12',Ks,'312',iKKs,'312');  % Ns-by-E
     
     if nargout > 3
        dSdxs = -2*etprod('123',dKsdxs,'4123',iKKs,'412'); % Ns-by-E-by-D
     end
end
