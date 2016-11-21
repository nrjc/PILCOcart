function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpc1d(gpmodel, m, s)

% Compute joint predictions and derivatives for GPs with a combination of first
% order additive SE kernels plus a standard SE kernel and uncertain inputs. If 
% dynmodel.nigp exists, individual noise contributions are added. Predictive 
% variances contain uncertainty about the function, but no measurement noise.
%
% inputs:
% X       log-hyper-parameters (p = (3D+2)*E)                           [p    ]
% input   training inputs                                               [n x D]
% target  training targets                                              [n x E]
% m       mean of the test distribution                                 [D    ] 
% s       covariance matrix of the test distribution                    [D x D] 
%
% outputs:
% M       mean of pred. distribution                            [E    ] 
% S       covariance of the pred. distribution                  [E x E]
% V       inv(s) times covariance between input and prediction  [D x E]
% dMdm    output mean   by input mean partial derivatives       [E     x D    ]
% dSdm    output cov    by input mean derivatives               [E x E x D    ]
% dVdm    inv(s)*io-cov by input mean derivatives               [D x E x D    ]
% dMds    ouput mean    by input covariance derivatives         [E     x D x D]
% dSds    output cov    by input covariance derivatives         [E x E x D x D] 
% dVds    inv(s)*io-cov by input covariance derivatives         [D x E x D x D]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen & Marc Deisenroth 2012-01-18
% Edited by Joe Hall 2012-03-21

persistent iK iK2 beta oldX;
ridge = 1e-6;                        % jitter to make matrix better conditioned
[n, D] = size(gpmodel.inputs);    % number of examples and dimension of inputs
E = size(gpmodel.target,2);         % number of examples and number of outputs
X = gpmodel.hyp; input = gpmodel.inputs; target = gpmodel.target;
D2 = D*2; D3 = D*3;
X1 = X([1:D D2+1:D3],:);                % hyperparameters for 1st order kernels
X2 = X([D+1:D2 D3+1],:);                 % hyperparameters for Dth order kernel

[np pD pE] = size(gpmodel.induce);     % number of pseudo inputs per dimension
pinput = gpmodel.induce;                                   % all pseudo inputs

if numel(X) ~= numel(oldX) || isempty(iK) || isempty(iK2) || ... % if necessary
              sum(any(X ~= oldX)) || numel(iK2) ~=E*np^2 || numel(iK) ~= n*np*E
  oldX = X;                                        % compute K, inv(K), inv(K2)
  iK = zeros(np,n,E); iK2 = zeros(np,np,E); beta = zeros(np,E);
    
  for i=1:E
    pinp1 = bsxfun(@rdivide,pinput(:,:,min(i,pE)),exp(X1(1:D,i)'));
    pinp2 = bsxfun(@rdivide,pinput(:,:,min(i,pE)),exp(X2(1:D,i)'));
    inp1 = bsxfun(@rdivide,input,exp(X1(1:D,i)'));
    inp2 = bsxfun(@rdivide,input,exp(X2(1:D,i)'));
    Kmm = ridge*eye(np); Kmn = zeros(np,n);                   % add small ridge
    for d=1:D
      Kmm = Kmm + exp(2*X1(D+d,i)-maha(pinp1(:,d),pinp1(:,d))/2);
      Kmn = Kmn + exp(2*X1(D+d,i)-maha(pinp1(:,d), inp1(:,d))/2);
    end
    Kmm = Kmm + exp(2*X2(D+1,i)-maha(pinp2(:,d),pinp2(:,d))/2);
    Kmn = Kmn + exp(2*X2(D+1,i)-maha(pinp2(:,d), inp2(:,d))/2);
    L = chol(Kmm)';
    V = L\Kmn;                                             % inv(sqrt(Kmm))*Kmn
    if isfield(gpmodel,'nigp')
      G = sum(exp(2*X(D2+1:D3+1,i))) - sum(V.^2) + gpmodel.nigp(:,i)';
    else
      G = sum(exp(2*X(D2+1:D3+1,i))) - sum(V.^2);
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
    iB = iAt'*iAt.*exp(2*X(D3+2,i));             % inv(B), [Ed's thesis, p. 40]
    iK2(:,:,i) = Kmm\eye(np) - iB; % covariance matrix for predictive variances       
  end
end

% initializations
M = zeros(E,1); V = zeros(D,E); S = zeros(E);                        % allocate
dMds = zeros(E,D,D); dSdm = zeros(E,E,D);
dSds = zeros(E,E,D,D); dVds = zeros(D,E,D,D);
k = zeros(np,E,D+1); inp = zeros(np,D,E);


% 1) Predicted Mean and Input-Output Covariance *******************************
for i = 1:E
  inp(:,:,i) = bsxfun(@minus,pinput(:,:,min(i,pE)),m');
    
  % 1a) First Order Additive Kernels ------------------------------------------
  for d = 1:D
  % 1ai) Compute the values
    R = s(d,d) + exp(2*X1(d,i)); iR = R\1;        %  R =         sdd + lamd [1]
    t = inp(:,d,i)*iR;                            %  t =           inp_d*iR [n]
    l = exp(-inp(:,d,i).*t/2);                    %  l = exp(-iR*inp_d.^2/2)[n]
    lb = l.*beta(:,i);                            % lb =            l.*beta [n]
    c = exp(2*X1(D+d,i))*sqrt(iR)*exp(X1(d,i));   %  c =  ad^2*sqrt(R*ilamd)[1]
    
    Md = c*sum(lb); M(i) = M(i) + Md;                          % predicted mean
    V(d,i) = c*t'*lb;                    % inv(s) times input-output covariance
    k(:,i,d) = 2*X1(D+d,i) - inp(:,d).^2*exp(-2*X1(d,i))/2;        % log-kernel
    
  % 1aii) Compute the derivatives
    tlb = t.*lb;                               %  tlb = iR*inp_d.*l.*beta   [n]
    dMds(i,d,d) = c*t'*tlb/2 - iR*Md/2;                                     % M
    dVds(d,i,d,d) = c*(t.*t)'*tlb/2 - iR*V(d,i)/2 - V(d,i)*iR;              % V
  end % d
  
  % 1b) Full Squared Exponential Kernel ---------------------------------------
  % 1bi) Compute the values
  L = diag(exp(-X2(1:D,i))); in = inp*L;
  B = L*s*L+eye(D); iR = L/B*L;              % iR =           inv(s + lam)[DxD]
  t1 = in/B; t = t1*L;                       %  t =                inp*iR [nxD]
  l = exp(-sum(in.*t1,2)/2);                 %  l =    exp(-inp*iR*inp'/2)  [n]
  lb = l.*beta(:,i);                         % lb =               l.*beta   [n]
  c = exp(2*X2(D+1,i))/sqrt(det(B));         %  c = aSE^2*sqrt(det(R*ilam)) [1]
  
  M(i) = M(i) + c*sum(lb);                                     % predicted mean
  Vi = t'*lb*c; V(:,i) = V(:,i) + Vi;    % inv(s) times input-output covariance
  k(:,i,D+1) = 2*X2(D+1,i)-sum(in.*in,2)/2;                        % log-kernel
  
  % 1bii) Compute the derivatives
  tlb = bsxfun(@times,t,lb);                   %  tlb = iR*inp.*(l.*beta) [nxD]
  dMds(i,:,:) = squeeze(dMds(i,:,:)) + c*t'*tlb/2 - iR*c*sum(lb)/2;         % M
  for d = 1:D
    dVds(d,i,:,:) = squeeze(dVds(d,i,:,:)) ...                              % V
            + c*bsxfun(@times,t,t(:,d))'*tlb/2 - iR*Vi(d)/2 - Vi*iR(d,:);
  end
  
end % i
dMdm = V';                                                                  % M 
dVdm = 2*permute(dMds,[2 1 3]);                                             % V


% 2) Predictive Covariance Matrix *********************************************
for i=1:E
  ii1 = bsxfun(@rdivide,inp(:,:,i),exp(2*X1(1:D,i)'));
  ii2 = bsxfun(@rdivide,inp(:,:,i),exp(2*X2(1:D,i)'));
  
  for j=1:i
    ij1 = bsxfun(@rdivide,inp(:,:,j),exp(2*X1(1:D,j)'));
    ij2 = bsxfun(@rdivide,inp(:,:,j),exp(2*X2(1:D,j)'));
    BB = beta(:,i)*beta(:,j)';
    if i==j; BB = BB - iK2(:,:,i); end           % incorporate model uncertainty
    LL = zeros(n);

    % 2a) Combo of First Order Additive Kernels -------------------------------
    for d=1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end     % fill in cross terms
      for e=1:P
    % 2ai) Compute the values
        if d==e % ----------------------------------------- k_id and k_je (d==e)
          sde = s(d,d); p = 1; pp = 1;
          ii_ = ii1(:,d); ij_ = ij1(:,e);
          eXij = exp(-2*X1(d,i)) + exp(-2*X1(e,j));
        else % -------------------------------------------- k_id and k_je (d~=e)
          sde = s([d e],[d e]); p = 2;
          ii_ = [ii1(:,d) zeros(n,1)]; ij_ = [zeros(n,1) ij1(:,e)];
          eXij = [exp(-2*X1(d,i)) exp(-2*X1(e,j))];
        end
        R = sde*diag(eXij) + eye(p);
        t = 1/sqrt(det(R));
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') + maha(ii_,-ij_,R\sde/2));
        LL = LL + pp*t*L;
        
    % 2aii) Compute the derivatives
        A = BB.*L;                          %       A = (betai*betaj').*L [nxn]
        ssA = sum(sum(A));                  %     ssA =    betai'*L*betaj   [1]
        zi = ii_/R; zj = ij_/R;             %  z{i/j} =        i{i/j}_*iR [nxp]
        T = zeros(p);
        for dd = 1:p
          if dd==1, f=d; else f=e; end
          B = bsxfun(@plus,zi(:,dd),zj(:,dd)').*A;                    % B [nxn]
          T(dd,1:dd) = sum(zi(:,1:dd)'*B,2) + sum(B*zj(:,1:dd))';     % T [pxp]
          T(1:dd,dd) = T(dd,1:dd)';
          dSdm(i,j,f) = dSdm(i,j,f) + pp*t*sum(sum(B));
        end % dd
        
        T = (t*T - t*ssA*diag(eXij)/R)/2;                             % T [pxp]
        if d==e, dSds(i,j,d,d) = dSds(i,j,d,d) + T;
        else     dSds(i,j,[d e],[d e]) = squeeze(dSds(i,j,[d e],[d e])) + pp*T;
        end
      end % e
    end % d
    
    % 2b) Combo of First Order and Full Squared Exponential Kernel ------------
    for d=1:D+1
      if i==j, P = 1; pp = 2; else P = 2; pp = 1; end     % fill in cross terms
      if d==D+1, P = 1; pp = 1; end
      for e=1:P
    % 2bi) Compute the values
        if d==D+1 % ------------------------------------------- k_iSE and k_jSE
          ii_ = ii2; ij_ = ij2;
          eXij = exp(-2*X2(1:D,i)) + exp(-2*X2(1:D,j));
          ki = k(:,i,D+1); kj = k(:,j,D+1);
        elseif e==1 % ------------------------------------------ k_id and k_jSE
          ii_ = 0*ii1; ii_(:,d) = ii1(:,d); ij_ = ij2;
          eXij = exp(-2*X2(1:D,j)); eXij(d) = eXij(d) + exp(-2*X1(d,i));
          ki = k(:,i,d); kj = k(:,j,D+1);
        else % ------------------------------------------------- k_iSE and k_jd
          ij_= 0*ij1; ij_(:,d) = ij1(:,d); ii_ = ii2;
          eXij = exp(-2*X2(1:D,i)); eXij(d) = eXij(d) + exp(-2*X1(d,j));
          kj = k(:,j,d); ki = k(:,i,D+1);
        end
        R = s*diag(eXij) + eye(D);
        t = 1/sqrt(det(R));
        L = exp(bsxfun(@plus,ki,kj') + maha(ii_,-ij_,R\s/2));
        LL = LL + pp*t*L;
        
    % 2bii) Compute the derivatives
        A = BB.*L;                          %       A = (betai*betaj').*L [nxn]
        ssA = sum(sum(A));                  %     ssA =    betai'*L*betaj   [1]
        zi = ii_/R; zj = ij_/R;             %  z{i/j} =        i{i/j}_*iR [nxD]
        T = zeros(D);
        for dd = 1:D
          B = bsxfun(@plus,zi(:,dd),zj(:,dd)').*A;                    % B [nxn]
          T(dd,1:dd) = sum(zi(:,1:dd)'*B,2) + sum(B*zj(:,1:dd))';     % T [DxD]
          T(1:dd,dd) = T(dd,1:dd)';
          dSdm(i,j,dd) = dSdm(i,j,dd) + pp*t*sum(sum(B));
        end
        
        T = t*(T - ssA*diag(eXij)/R)/2;                               % T [DxD]
        dSds(i,j,:,:) = squeeze(dSds(i,j,:,:)) + pp*T;
        
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
  S(i,i) = S(i,i) + sum(exp(2*X(D2+1:D3+1,i)));
end % i

S = S - M*M';                                              % centralize moments