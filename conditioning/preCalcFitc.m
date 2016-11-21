function dynmodel = preCalcFitc(dynmodel)

% Dimensions
[Nt E] = size(dynmodel.target);
D = size(dynmodel.inputs,2);
[np pD pE] = size(dynmodel.induce);

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