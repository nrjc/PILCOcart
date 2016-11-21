function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpa0d(gpmodel, m, s)

% Compute joint predictions and derivatives for GPs with first order additive
% SE kernels and uncertain inputs. If dynmodel.nigp exists, individial noise 
% contributions are added. Predictive variances contain uncertainty about the 
% function, but no measurement noise.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters ( P = 2*D+1 )                      [ P  x  E ]
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
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
% Copyright (C) 2008-2012 by Carl Edward Rasmussen & Marc Deisenroth 2012-01-18
% Edited by Joe Hall 2012-03-21

input = gpmodel.inputs;  target = gpmodel.target; X = gpmodel.hyp;

if nargout < 4; [M, S, V] = gpa0(gpmodel, m, s); return; end

persistent K iK oldX oldIn beta;                         % cache some variables
[n, D] = size(input);            % no. of examples and dimension of input space
[n, E] = size(target);                  % no. of examples and number of outputs
D2 = 2*D; X = reshape(X, D2+1, E);

% if necessary: re-compute cashed variables
if length(X) ~= length(oldX) || isempty(iK) || ...
                                sum(any(X ~= oldX)) || sum(any(oldIn ~= input))
  oldX = X; oldIn = input;                                             
  K = zeros(n,n,E); iK = K; beta = zeros(n,E);

  % compute K and inv(K) and beta
  for i=1:E
    inp = bsxfun(@rdivide,input,exp(X(1:D,i)'));
    for d = 1:D,
      K(:,:,i) = K(:,:,i) + exp(2*X(D+d,i)-maha(inp(:,d),inp(:,d))/2);
    end
    if isfield(gpmodel,'nigp')
      L = chol(K(:,:,i) + exp(2*X(D2+1,i))*eye(n)+diag(gpmodel.nigp(:,i)))';
    else        
      L = chol(K(:,:,i) + exp(2*X(D2+1,i))*eye(n))';
    end
    iK(:,:,i) = L'\(L\eye(n));
    beta(:,i) = L'\(L\gpmodel.target(:,i));
  end
end

% initializations
M    = zeros(E,1);   S    = zeros(E);       V    = zeros(D,E);
dMdm = zeros(E,D);   dSdm = zeros(E,E,D);   dVdm = zeros(D,E,D);
dMds = zeros(E,D,D); dSds = zeros(E,E,D,D); dVds = zeros(D,E,D,D);
k = zeros(n,E,D);

% centralize training inputs
inp = bsxfun(@minus,input,m');

% 1) Predicted Mean and Input-Output Covariance *******************************
for i = 1:E
  for d = 1:D
    % 1a) Compute the values **************************************************
    R = s(d,d) + exp(2*X(d,i)); iR = R\1;      %  R =            sdd + lamd [1]
    t = inp(:,d)*iR;                           %  t =              inp_d*iR [1]
    l = exp(-inp(:,d).*t/2);                   %  l = exp(-ilamd*inp_d.^2/2)[n]
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
  ii = bsxfun(@rdivide,inp,exp(2*X(1:D,i)'));

  for j = 1:i
    ij = bsxfun(@rdivide,inp,exp(2*X(1:D,j)'));
    BB = beta(:,i)*beta(:,j)';                        % BB = betai*betaj' [nxn]
    if i==j; BB = BB - iK(:,:,i); end           % incorporate model uncertainty
    
    for d = 1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end
      for e = 1:P
        % 2a) Compute the value ***********************************************
        if d==e % ----------------- combo of 1D kernels on same input dimension
          p = 1; pp = 1;                                       %     d==e:  p=1
          ii_d = ii(:,d); ij_e = ij(:,e);                      %  ii_/ij_   [n]
          sde = s(d,d);                                        %      sde   [1]
          eXij = exp(-2*X(d,i)) + exp(-2*X(e,j));              % l-scales   [1]
        else % -------------- combo of 1D kernels on different input dimensions
          p = 2;                                               %     d~=e:  p=2
          ii_d=[ii(:,d) zeros(n,1)]; ij_e=[zeros(n,1) ij(:,e)];%  ii_/ij_ [nx2]
          sde = s([d e],[d e]);                                %      sde [2x2]
          eXij = [exp(-2*X(d,i)) exp(-2*X(e,j))];              % l-scales [1x2]
        end
        R = sde*diag(eXij) + eye(p);                              %     R [pxp]
        t = 1/sqrt(det(R));                                       %     t   [1]
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') ...              %     L [nxn]
                                    + maha(ii_d,-ij_e,R\sde/2));
        A = BB.*L;                             %    A = (betai*betaj').*L [nxn]
        ssA = sum(sum(A));                     %  ssA =    betai'*L*betaj   [1]
        S(i,j) = S(i,j) + pp*t*ssA; S(j,i) = S(i,j);     % predicted covariance
          
        % 2b) Compute the derivatives *****************************************
        zi = ii_d/R; zj = ij_e/R;             %  z{i/j} = ii_{d/e}*iR     [nxp]
        T = zeros(p);
        for dd = 1:p
          if dd==1, f=d; else f=e; end
        % ------------------------------------------- derivatives w.r.t m and s
          B = bsxfun(@plus,zi(:,dd),zj(:,dd)').*A;                    % B [nxn]
          dSdm(i,j,f) = dSdm(i,j,f) + pp*t*sum(sum(B));
          T(dd,1:dd) = sum(zi(:,1:dd)'*B,2) + sum(B*zj(:,1:dd))';     % T [pxp]
          T(1:dd,dd) = T(dd,1:dd)';
        end % dd
                  
        % ------------------------------------------------- derivatives w.r.t s
        T = t*(T - ssA*diag(eXij)/R)/2;                               % T [pxp]
        if d==e, dSds(i,j,d,d) = dSds(i,j,d,d) + T;
        else     dSds(i,j,[d e],[d e]) = squeeze(dSds(i,j,[d e],[d e])) + pp*T;
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

dMds = reshape(dMds,[E D*D]);                           % vectorise derivatives
dSds = reshape(dSds,[E*E D*D]); dSdm = reshape(dSdm,[E*E D]);
dVds = reshape(dVds,[D*E D*D]); dVdm = reshape(dVdm,[D*E D]);