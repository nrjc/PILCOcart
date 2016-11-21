function [M S dynmodel dMdxs dSdxs dMdyc] = fitcCond(dynmodel,xs,yc)
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
%   ys              Ouput test prediction, Ns-by-E
%   s2              Output test variance, Ns-by-E
%
% Andrew McHutchon, 21/02/2011

% Handle other configurations
if isfield(dynmodel,'inp'); dynmodel.inputs = dynmodel.inp; end
if isfield(dynmodel,'tar'); dynmodel.target = dynmodel.tar; end
if nargin == 2; yc = []; end

% Dimensions
Ns = size(xs,1);
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

% Find the test point covariance matrices
Ks = calcKux(sf2,iell,u,xs);
if nargout >3
    XmX = bsxfun(@minus,permute(u,[1,4,3,2]),permute(xs,[3,1,4,2])); % np-by-Ns-by-E-by-D
    XmxiLam = bsxfun(@times,XmX,permute(iell.^2,[4,3,2,1]));     % N-by-Ns-by-E-by-D
    dKsdxs = bsxfun(@times,XmxiLam,Ks);                      % np-by-Ns-by-E-by-D
end

% Find the joint test posterior distribution

% Calculate posterior mean
M = etprod('12',Ks,'312',dynmodel.beta,'32');     % Ns-by-E
if nargout > 3
    dMdxs = etprod('123',dKsdxs,'4123',dynmodel.beta,'42');       % Ns-by-E-by-D
end

% Calculate posterior covariance
Kss = zeros(Ns,Ns,E);
iKKs = etprod('123',dynmodel.iK2,'143',Ks,'423');
for i=1:E
    inp = bsxfun(@times,xs,iell(:,i)');
    Kss(:,:,i) = exp(2*hyp(D+1,i)-maha(inp,inp)/2);
end
S = Kss - etprod('124',Ks,'314',iKKs,'324');                % Ns-by-Ns-by-E

% Deriviatives of the joint        
if nargout > 3
    XmX = permute(bsxfun(@minus,permute(xs,[3,1,2]),permute(xs,[1,3,2])),[1,2,4,3]); % Ns-by-Ns-by-1-by-D
    XmX = bsxfun(@times,XmX,permute(iell.^2,[4,3,2,1])); % Ns-by-Ns-by-E-by-D
    dKss = bsxfun(@times,Kss,XmX);                       % Ns-by-Ns-by-E-by-D
    ds2 = dKss - etprod('1234',dKsdxs,'5134',iKKs,'523'); % Ns-by-Ns-by-E-by-D
    for i=1:Ns; ds2(i,i,:) = 2*ds2(i,i,:); end
end

% Now we condition on Ns-1 points
dMdyc = zeros(E,Ns-1,E);
if ~isempty(yc)
  k = 1:(Ns-1); Mc = zeros(E,1); Sc = zeros(E,1);
  dM = zeros(E,Ns,D); dSdxs = zeros(E,Ns,D); 

  for i=1:E
    %S(k,k,i) = S(k,k,i) + sn2(i)*eye(Ns-1);       % Add noise
    S(k,k,i) = S(k,k,i) + 1e-8*eye(Ns-1);       % Add noise
    iSym = S(k,k,i)\(yc(:,i)-M(k,i));           % k-by-1
    SiS = S(end,k,i)/S(k,k,i);                  % 1-by-k
    iSS = S(k,k,i)\S(k,end,i);                  % k-by-1
            
    Mc(i) = M(end,i) + SiS*(yc(:,i)-M(k,i));
    Sc(i) = S(end,end,i) - SiS*S(k,end,i);
        
    % Derivatives
    if nargout > 3
        SiSdS = permute(bsxfun(@times,SiS',ds2(k,k,i,:)),[1,2,4,3]); % k-by-k-by-D
        SiSdSd = etprod('12',SiS,'43',ds2(k,k,i,:),'1342');  % k-by-D
        for j=1:length(k); SiSdS(j,j,:) = SiSdSd(j,:); end
            
        dM(i,k,:) = -etprod('12',SiSdS,'132',iSym,'3') + ...
                bsxfun(@times,permute(ds2(k,end,i,:),[1,4,2,3]),iSym) - ...
                bsxfun(@times,SiS',permute(dMdxs(k,i,:),[1,3,2]));
        dM(i,end,:) = squeeze(dMdxs(end,i,:)) + etprod('1',ds2(end,k,i,:),'4231',iSym,'2');    
            
        dSdxs(i,k,:) = etprod('12',SiSdS,'132',iSS,'3') ...
                 -2*bsxfun(@times,permute(ds2(k,end,i,:),[1,4,2,3]),iSS);
        dSdxs(i,end,:) = squeeze(ds2(end,end,i,:)) - 2*permute(ds2(end,k,i,:),[4,2,1,3])*iSS;
            
        dMdyc(i,:,i) = SiS;
    end
  end
  M = Mc; S = Sc; dMdxs = dM;
else
    M = M'; S = squeeze(S);
    if nargout > 3; dMdxs = permute(dMdxs,[2,1,3]); dSdxs = permute(ds2,[3,1,4,2]); end
end

function Kmn = calcKux(sf2,iell,u,x)
Nt = size(x,1); E = length(sf2); Ns = size(u,1);
Kmn = zeros(Ns,Nt,E);
for i=1:E
    j = min(size(u,3),i);
    Kmn(:,:,i) = sf2(i)*exp(-sq_dist(diag(iell(:,i))*u(:,:,j)',diag(iell(:,i))*x')/2);
end