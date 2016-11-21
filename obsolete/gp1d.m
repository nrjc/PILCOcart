function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gp1d(gpmodel, m, s)

% Compute joint predictions for the FITC sparse approximation to multiple GPs
% with uncertain inputs. If dynmodel.nigp exists, individual noise contribu-
% tions are added.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters                                    [D+2 x  E ]
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   induce  inducint inputs                                         [ np x  D ]
%   nigp    (optional) individual noise variance terms              [ n  x  E ]
% m         mean of the test distribution                           [ D       ]
% s         covariance matrix of the test distribution              [ D  x  D ]
%
% M         mean of pred. distribution                              [ E       ]
% S         covariance of the pred. distribution                    [ E  x  E ]
% V         inv(s) times covariance between input and output        [ D  x  E ]
% dMdm      output mean by input mean                               [ E  x  D ]
% dSdm      output covariance by input mean                         [E*E x  D ]
% dVdm      inv(s)*input-output covariance by input mean            [D*E x  D ]
% dMds      ouput mean by input covariance                          [ E  x D*D]
% dSds      output covariance by input covariance                   [E*E x D*D]
% dVds      inv(s)*input-output covariance by input covariance      [D*E x D*D]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth, 
%  Andrew McHutchon, & Joe Hall   2012-07-09

if nargout < 4; [M S V] = gp1(gpmodel, m, s); return; end     % no derivs, gp1
if numel(gpmodel.induce) == 0           % no inducing inputs, back off to gp0d
  [M S V dMdm dSdm dVdm dMds dSds dVds] = gp0d(gpmodel, m, s); return;
end

persistent iK2 beta oldh;
ridge = 1e-6;                        % jitter to make matrix better conditioned
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
E = size(gpmodel.target,2);         % number of examples and number of outputs
h = gpmodel.hyp; input = gpmodel.inputs; target = gpmodel.target;

[np pD pE] = size(gpmodel.induce);     % number of pseudo inputs per dimension
pinput = gpmodel.induce;                                   % all pseudo inputs

if numel(unwrap(h)) ~= numel(oldh) || isempty(iK2) || ... % if necessary
                        any(unwrap(h) ~= oldh) || numel(iK2) ~=E*np^2
  oldh = unwrap(h); iK = zeros(np,n,E); iK2 = zeros(np,np,E); beta = zeros(np,E);
    
  for i=1:E
    pinp = bsxfun(@times,pinput(:,:,min(i,pE)),exp(-h(i).l'));
    inp = bsxfun(@times,input,exp(-h(i).l'));
    Kmm = exp(2*h(i).s-maha(pinp,pinp)/2) + ridge*eye(np);  % add small ridge
    Kmn = exp(2*h(i).s-maha(pinp,inp)/2);
    L = chol(Kmm)';
    V = L\Kmn;                                             % inv(sqrt(Kmm))*Kmn
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
    beta(:,i) = iK(:,:,i)*target(:,i);      
    iB = iAt'*iAt.*exp(2*h(i).n);              % inv(B), [Ed's thesis, p. 40]
    iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
  end
end

k = zeros(np,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);       % allocate
dMds = zeros(E,D,D); dSdm = zeros(E,E,D); dSds = zeros(E,E,D,D);
dVds = zeros(D,E,D,D); T = zeros(D); inp = zeros(np,D,E);

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  inp(:,:,i) = bsxfun(@minus,pinput(:,:,min(i,pE)),m');   % centralize p-inputs
    
  iL = diag(exp(-h(i).l));
  in = inp(:,:,i)*iL;
  B = iL*s*iL+eye(D);  LiBL = iL/B*iL;
  t = in/B;
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*iL;
  tlb = bsxfun(@times,tL,lb);
  c = exp(2*h(i).s)/sqrt(det(B));

  M(i) = c*sum(lb);
  V(:,i) = tL'*lb*c;                   % inv(s) times input-output covariance
  dMds(i,:,:) = c*tL'*tlb/2 - LiBL*M(i)/2;
  for d = 1:D
    dVds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - LiBL*V(d,i)/2 ...
                                                             - V(:,i)*LiBL(d,:);
  end
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
end
dMdm = V'; dVdm = 2*permute(dMds,[2 1 3]);                  % derivatives wrt m


il = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(il,[3,1,2])); % N-by-D-by-E
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(il(:,i)+il(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    
    if i==j
      iKL = iK2(:,:,i).*L; s1iKL = sum(iKL,1); s2iKL = sum(iKL,2);  
      S(i,j) = t*(beta(:,i)'*L*beta(:,i) - sum(s1iKL));
      zi = ii/R; 
      bibLi = L'*beta(:,i).*beta(:,i); cbLi = L'*bsxfun(@times, beta(:,i), zi);
      r = (bibLi'*zi*2 - (s2iKL' + s1iKL)*zi)*t;
      for d = 1:D
        T(d,1:d) = 2*(zi(:,1:d)'*(zi(:,d).*bibLi) + ...
            cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) ...
                                                   - zi(:,1:d)'*(iKL*zi(:,d)));
        T(1:d,d) = T(d,1:d)';
      end
    else
      zi = ii/R; zj = ij/R;
      S(i,j) = beta(:,i)'*L*beta(:,j)*t; S(j,i) = S(i,j);

      bibLj = L*beta(:,j).*beta(:,i); bjbLi = L'*beta(:,i).*beta(:,j);
      cbLi = L'*bsxfun(@times, beta(:,i), zi);
      cbLj = L*bsxfun(@times, beta(:,j), zj);

      r = (bibLj'*zi+bjbLi'*zj)*t;
      for d = 1:D
        T(d,1:d) = zi(:,1:d)'*(zi(:,d).*bibLj) + ...
          cbLi(:,1:d)'*(zj(:,d).*beta(:,j)) + zj(:,1:d)'*(zj(:,d).*bjbLi) + ...
                                             cbLj(:,1:d)'*(zi(:,d).*beta(:,i));
        T(1:d,d) = T(d,1:d)';
      end
    end        
    
    dSdm(i,j,:) = r - M(i)*dMdm(j,:)-M(j)*dMdm(i,:); dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S(i,j)*diag(il(:,i)+il(:,j))/R)/2;
    T = T - squeeze(M(i)*dMds(j,:,:) + M(j)*dMds(i,:,:));
    dSds(i,j,:,:) = T; dSds(j,i,:,:) = T;  
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);
end

S = S - M*M';                                              % centralize moments

dMds = reshape(dMds,[E D*D]);                           % vectorise derivatives
dSds = reshape(dSds,[E*E D*D]); dSdm = reshape(dSdm,[E*E D]);
dVds = reshape(dVds,[D*E D*D]); dVdm = reshape(dVdm,[D*E D]);