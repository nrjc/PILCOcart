function [M S dynmodel dMdxs dSdxs dMdyc] = gpCond(dynmodel,xs,yc)
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

% First we need to find the joint test point distribution

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
     
% Deriviatives of the joint        
if nargout > 3
    XmX = permute(bsxfun(@minus,permute(xs,[3,1,2]),permute(xs,[1,3,2])),[1,2,4,3]); % Ns-by-Ns-by-1-by-D
    XmX = bsxfun(@times,XmX,permute(iell.^2,[4,3,2,1])); % Ns-by-Ns-by-E-by-D
    dKss = bsxfun(@times,Kss,XmX);                       % Ns-by-Ns-by-E-by-D
    ds2 = dKss - etprod('1234',dKsdxs,'5134',iKKs,'523'); % Ns-by-Ns-by-E-by-D
    for i=1:Ns; ds2(i,i,:) = 2*ds2(i,i,:); end
end
        
% Now we condition on Ns-1 points
if ~isempty(yc)
  k = 1:(Ns-1); Mc = zeros(E,1); Sc = zeros(E,1);
  dM = zeros(E,Ns,D); dSdxs = zeros(E,Ns,D); dMdyc = zeros(E,Ns-1,E);

  for i=1:E
    S(k,k,i) = S(k,k,i) + exp(2*hyp(end,i))*eye(Ns-1);
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

if any(S(:) < 0); keyboard; end
if any(isnan(S(:))) || any(isnan(M(:))); keyboard; end

  