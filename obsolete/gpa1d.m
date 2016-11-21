function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpa1d(gpmodel, m, s)

% Compute joint predictions for the FITC sparse approximation to multiple GPs
% with first order additive squared exponential kernels and uncertain inputs.
% If dynmodel.nigp exists, individual noise contributions are added.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters ( P = 2*D+1 )                      [ P  x  E ]
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   induce  inducing inputs                                         [ np x  D ]
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
% dVds      inv(s)*input-output covariance by input covariance      [D*E x
% D*D]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen & Marc Deisenroth 2012-01-15
% Edited by Joe Hall 2012-04-04

if nargout < 4; [M S V] = gpa1(gpmodel, m, s); return; end   % no derivs, gap1
if numel(gpmodel.induce) == 0          % no inducing inputs, back off to gpa0d
  [M S V dMdm dSdm dVdm dMds dSds dVds] = gpa0d(gpmodel, m, s); return;
end

persistent iK iK2 beta oldX;
ridge = 1e-6;                        % jitter to make matrix better conditioned
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
E = size(gpmodel.target,2);         % number of examples and number of outputs
X = gpmodel.hyp; input = gpmodel.inputs; target = gpmodel.target;
D2 = 2*D;

[np pD pE] = size(gpmodel.induce);     % number of pseudo inputs per dimension
pinput = gpmodel.induce;                                   % all pseudo inputs

if numel(X) ~= numel(oldX) || isempty(iK) || isempty(iK2) || ... % if necessary
              sum(any(X ~= oldX)) || numel(iK2) ~=E*np^2 || numel(iK) ~= n*np*E
  oldX = X;                                        % compute K, inv(K), inv(K2)
  iK = zeros(np,n,E); iK2 = zeros(np,np,E); beta = zeros(np,E);
    
  for i=1:E
    pinp = bsxfun(@rdivide,pinput(:,:,min(i,pE)),exp(X(1:D,i)'));
    inp = bsxfun(@rdivide,input,exp(X(1:D,i)'));
    Kmm = ridge*eye(np); Kmn = zeros(np,n);                   % add small ridge
    for d=1:D
      Kmm = Kmm + exp(2*X(D+d,i)-maha(pinp(:,d),pinp(:,d))/2);
      Kmn = Kmn + exp(2*X(D+d,i)-maha(pinp(:,d), inp(:,d))/2);
    end
    L = chol(Kmm)';
    V = L\Kmn;                                             % inv(sqrt(Kmm))*Kmn
    if isfield(gpmodel,'nigp')
      G = sum(exp(2*X(D+1:D2,i))) - sum(V.^2) + gpmodel.nigp(:,i)';
    else
      G = sum(exp(2*X(D+1:D2,i))) - sum(V.^2);
    end
    G = sqrt(1+G/exp(2*X(D2+1,i)));
    V = bsxfun(@rdivide,V,G);
    Am = chol(exp(2*X(D2+1,i))*eye(np) + V*V')';
    At = L*Am;                                    % chol(sig*B) [thesis, p. 40]
    iAt = At\eye(np);
% The following is not an inverse matrix, but we'll treat it as such: multiply
% the targets from right and the cross-covariances left to get predictive mean.
    iK(:,:,i) = ((Am\(bsxfun(@rdivide,V,G)))'*iAt)';
    beta(:,i) = iK(:,:,i)*target(:,i);      
    iB = iAt'*iAt.*exp(2*X(D2+1,i));             % inv(B), [Ed's thesis, p. 40]
    iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
  end
end

M = zeros(E,1); V = zeros(D,E); S = zeros(E);                        % allocate
dMds = zeros(E,D,D); dSdm = zeros(E,E,D);
dSds = zeros(E,E,D,D); dVds = zeros(D,E,D,D);
k = zeros(np,E,D); inp = zeros(np,D,E);

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  inp(:,:,i) = bsxfun(@minus,pinput(:,:,min(i,pE)),m');   % centralize p-inputs
  for d = 1:D
    % 1a) Compute the values **************************************************
    R = s(d,d) + exp(2*X(d,i)); iR = R\1;      %  R =            sdd + lamd [1]
    t = inp(:,d,i)*iR;                         %  t =              inp_d*iR [1]
    l = exp(-inp(:,d,i).*t/2);                 %  l = exp(-ilamd*inp_d.^2/2)[n]
    lb = l.*beta(:,i);                         % lb =               l.*beta [n]
    c = exp(2*X(D+d,i))*sqrt(iR)*exp(X(d,i));  %  c =   ad^2*(R*ilamd)^-0.5 [1]
    
    Md = c*sum(lb); M(i) = M(i) + Md;                          % predicted mean
    V(d,i) = c*t'*lb;                    % inv(s) times input-output covariance
    k(:,i,d) = 2*X(D+d,i) - inp(:,d).^2*exp(-2*X(d,i))/2;          % log-kernel
    
    % 1b) Compute the derivatives *********************************************
    tlb = t.*lb;                               %  tlb = iR*inp_d.*l.*beta   [n]
    dMds(i,d,d) = c*t'*tlb/2 - iR*Md/2;                                     % M
    dVds(d,i,d,d) = c*(t.*t)'*tlb/2 - iR*V(d,i)/2 - V(d,i)*iR;              % V
  end % d
end % i
dMdm = V';                                                                  % M 
dVdm = 2*permute(dMds,[2 1 3]);                                             % V


% 2) Predictive Covariance Matrix *********************************************
for i = 1:E
  ii = bsxfun(@rdivide,inp(:,:,i),exp(2*X(1:D,i)'));

  for j = 1:i
    ij = bsxfun(@rdivide,inp(:,:,j),exp(2*X(1:D,j)'));
    BB = beta(:,i)*beta(:,j)';                        % BB = betai*betaj' [nxn]
    if i==j; BB = BB - iK2(:,:,i); end          % incorporate model uncertainty
    
    for d = 1:D
      for e = 1:D
        % 2a) Compute the value ***********************************************
        if d==e % ----------------- combo of 1D kernels on same input dimension
          p = 1;                                               %     d==e:  p=1
          ii_d = ii(:,d); ij_e = ij(:,e);                      %  ii_/ij_   [n]
          sde = s(d,d);                                        %      sde   [1]
          eXi = exp(-2*X(d,i)); eXj = exp(-2*X(e,j));          % l-scales   [1]
        else % -------------- combo of 1D kernels on different input dimensions
          p = 2;                                               %     d~=e:  p=2
          ii_d=[ii(:,d) zeros(n,1)]; ij_e=[zeros(n,1) ij(:,e)];%  ii_/ij_ [nx2]
          sde = s([d e],[d e]);                                %      sde [2x2]
          eXi = [exp(-2*X(d,i)) 0]; eXj = [0 exp(-2*X(e,j))];  % l-scales [1x2]
        end
        R = sde*diag(eXi + eXj) + eye(p);                         %     R [pxp]
        t = 1/sqrt(det(R));                                       %     t   [1]
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') ...              %     L [nxn]
                                    + maha(ii_d,-ij_e,R\sde/2));
        A = BB.*L;                             %    A = (betai*betaj').*L [nxn]
        ssA = sum(sum(A));                     %  ssA =    betai'*L*betaj   [1]
        S(i,j) = S(i,j) + t*ssA; S(j,i) = S(i,j);        % predicted covariance
          
        % 2b) Compute the derivatives *****************************************
        zi = ii_d/R; zj = ij_e/R;             %  z{i/j} = ii_{d/e}*iR     [nxp]
        T = zeros(p);
        for dd = 1:p
          if dd==1, f=d; else f=e; end
        % ------------------------------------------- derivatives w.r.t m and s
          B = bsxfun(@plus,zi(:,dd),zj(:,dd)').*A;                    % B [nxn]
          dSdm(i,j,f) = dSdm(i,j,f) + t*sum(sum(B));
          T(dd,1:dd) = sum(zi(:,1:dd)'*B,2) + sum(B*zj(:,1:dd))';     % T [pxp]
          T(1:dd,dd) = T(dd,1:dd)';
        end % dd
                  
        % ------------------------------------------------- derivatives w.r.t s
        T = (t*T - t*ssA*diag(eXi + eXj)/R)/2;                        % T [pxp]
        if d==e, dSds(i,j,d,d) = dSds(i,j,d,d) + T;
        else     dSds(i,j,[d e],[d e]) = squeeze(dSds(i,j,[d e],[d e])) + T;
        end
      end % e
    end % d

    % ------------------------------------------- centralise moment derivatives
    dSdm(i,j,:)   =shiftdim(dSdm(i,j,:)  ,1)-M(i)*dMdm(j,:)  -M(j)*dMdm(i,:);
    dSds(i,j,:,:) =shiftdim(dSds(i,j,:,:),1)-M(i)*dMds(j,:,:)-M(j)*dMds(i,:,:);

    % ---------------------------------------------- fill in the symmetric bits
    if i~=j
      dSdm(j,i,:)   = dSdm(i,j,:);
      dSds(j,i,:,:) = dSds(i,j,:,:);
    end
    
  end % j
  S(i,i) = S(i,i) + sum(exp(2*X(D+1:D2,i)));
end % i

S = S - M*M';                                              % centralise moments