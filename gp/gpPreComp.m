function gpmodel = gpPreComp(gpmodel)
% Function to calculate the precomputable GP predictive matrices, iK and
% beta. iK is the inverse covariance matrix of the training inputs and 
% beta = iK*y, the training targets. Both of these matrices have their
% analogues in the FITC approximation. The fields oldh and oldn are used to
% test whether the matrices need to be recomputed.
%
% The function also tests to see if the GP prior mean function paramters m
% and b are present. If only one of these is present then the other field
% is created and set to zeros as later functions will expect them to either
% not exist at all or to both exist.
%
% Input
% gpmodel   dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%     .m    D-by-1 linear weights for the GP mean
%     .b    1-by-1 biases for the GP mean
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%
% Output adds the fields
%   iK      n-by-n-by-E, inverse covariance matrix
%   beta    n-by-E, iK*(targets - mean function of inputs)
%   oldh    unwrapped vector of the hyperparameters used to calculate iK & beta
%   oldn    the number of training cases
%
% Andrew McHutchon, 8th July 2013

h = gpmodel.hyp; [n, D] = size(gpmodel.inputs);
if ~isfield(h,'m'); [h.m] = deal(zeros(D,1)); end   % add zero mean if ...
if ~isfield(h,'b'); [h.b] = deal(0); end            % not specified by user
gpmodel.hyp = h;
if isfield(gpmodel,'induce') && numel(gpmodel.induce) > 0;
      gpmodel = initgp1(gpmodel);                  % FITC initialisation
else  gpmodel = initgp0(gpmodel);                  % Full GP initialisation
end
gpmodel.oldh = unwrap(h); gpmodel.oldn = n;

%%%%%%%%%% init functions %%%%%%%%%%%%%%%%

% --------------- full GP initialisation ----------------------------------
function gpmodel = initgp0(gpmodel)
h = gpmodel.hyp; x = gpmodel.inputs;
n = size(x,1); E = length(h);       % n = # training points, E = output dim
                                              
K = zeros(n,n,E); gpmodel.iK = zeros(n,n,E); gpmodel.beta = zeros(n,E);
for i=1:E                                            % compute K and inv(K)
    inp = bsxfun(@times,x,exp(-h(i).l'));
    K(:,:,i) = exp(2*h(i).s-maha(inp,inp)/2);
    if isfield(gpmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n) + diag(gpmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*h(i).n)*eye(n))';
    end
    gpmodel.iK(:,:,i) = L'\(L\eye(n));
    if isfield(gpmodel,'target'); 
        y = gpmodel.target(:,i) - x*h(i).m - h(i).b;
        gpmodel.beta(:,i) = L'\(L\y);
    end
end

% -------- FITC initialisation -------------------------------------------
function gpmodel = initgp1(gpmodel)
ridge = 1e-6;                    % jitter to make matrix better conditioned
input = gpmodel.inputs; pinput = gpmodel.induce; h = gpmodel.hyp;
[np, ~, pE] = size(pinput); n = size(input,1); E = length(h);
iK = zeros(np,n,E); iK2 = zeros(np,np,E); beta = zeros(np,E);
    
for i=1:E
    pinp = bsxfun(@times,pinput(:,:,min(i,pE)),exp(-h(i).l'));
    inp = bsxfun(@times,input,exp(-h(i).l'));
    Kmm = exp(2*h(i).s-maha(pinp,pinp)/2) + ridge*eye(np); % add small ridge
    Kmn = exp(2*h(i).s-maha(pinp,inp)/2);
    L = chol(Kmm)';
    V = L\Kmn;                                         % inv(sqrt(Kmm))*Kmn
    G = exp(2*h(i).s)-sum(V.^2);
    if isfield(gpmodel,'nigp'); G = G + gpmodel.nigp(:,i)'; end
    G = sqrt(1+G/exp(2*h(i).n));
    V = bsxfun(@rdivide,V,G);
    Am = chol(exp(2*h(i).n)*eye(np) + V*V')';
    At = L*Am;                                    % chol(sig*B) [thesis, p. 40]
    iAt = At\eye(np);
% The following is not an inverse matrix, but we'll treat it as such: multiply
% the targets from right and the cross-covariances left to get predictive mean.
    iK(:,:,i) = ((Am\(bsxfun(@rdivide,V,G)))'*iAt)';
    if isfield(gpmodel,'target')
        y = gpmodel.target(:,i) - input*h(i).m - h(i).b;
        beta(:,i) = iK(:,:,i)*y;     
    end
    iB = iAt'*iAt.*exp(2*h(i).n);              % inv(B), [Ed's thesis, p. 40]
    iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
end
gpmodel.iK = iK2;
if isfield(gpmodel,'target'); gpmodel.beta = beta; end
