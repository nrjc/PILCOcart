function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, dMdi, dSdi, dVdi, ...
                     dMdt, dSdt, dVdt, dMdX, dSdX, dVdX] = gpa2d(gpmodel, m, s)

% Compute joint predictions and derivatives for GPs with first order additive
% squared exponential kernels and uncertain inputs. Predictive variances
% contain neither uncertainty about the function nor measurement noise.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters ( P = 2*D+1 )                      [ P  x  E ]
%   inputs  training inputs                                         [ n  x  D ]
%   target  training targets                                        [ n  x  E ]
%   induce  inducint inputs                                         [ np x  D ]
%   noise   (optional) individual noise variance terms              [ n  x  E ]
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
% dMdi      output mean by inputs                                   [ E  x n*D]
% dSdi      output covariance by inputs                             [E*E x n*D]
% dVdi      inv(s) times input-output covariance by inputs          [D*E x n*D]
% dMdt      output mean by targets                                  [ E  x n*E]
% dSdt      output covariance by targets                            [E*E x n*E]
% dVdt      inv(s) times input-output covariance by targets         [D*E x n*E]
% dMdX      output mean by hyperparameters                          [ E  x P*E]
% dSdX      output covariance by hyperparameters                    [E*E x P*E]
% dVdX      inv(s) times input-output covariance by hyperparameters [D*E x P*E]
%
% Copyright (C) 2008-2012 by Marc Deisenroth and Carl Edward Rasmussen, 
% 2012-01-18, Edited by Joe Hall 2012-03-21

input = gpmodel.inputs;  target = gpmodel.target; X = gpmodel.hyp;

if nargout < 4; [M, S, V] = gpa2(gpmodel, m, s); return; end

persistent K KK iK oldX oldIn beta;                      % cache some variables
[n, D] = size(input);            % no. of examples and dimension of input space
[n, E] = size(target);                  % no. of examples and number of outputs
D2 = 2*D; X = reshape(X, D2+1, E)';

% if necessary: re-compute cashed variables
if length(X) ~= length(oldX) || isempty(iK) || ...
                                sum(any(X ~= oldX)) || sum(any(oldIn ~= input))
  oldX = X; oldIn = input;                                             
  K = zeros(n,n,E); iK = K; beta = zeros(n,E); KK = zeros(n,n,E,D);

  % compute K and inv(K) and beta
  for i=1:E
    inp = bsxfun(@rdivide,input,exp(X(i,1:D)));
    for d = 1:D,
      KK(:,:,i,d) = exp(2*X(i,D+d)-maha(inp(:,d),inp(:,d))/2);
      K(:,:,i) = K(:,:,i) + KK(:,:,i,d);
    end
    L = chol(K(:,:,i)+exp(2*X(i,D2+1))*eye(n))';
    iK(:,:,i) = L'\(L\eye(n));
    beta(:,i) = L'\(L\gpmodel.target(:,i));
  end
end

% initializations
M    = zeros(E,1);      S    = zeros(E);          V    = zeros(D,E);
dMdm = zeros(E,D);      dSdm = zeros(E,E,D);      dVdm = zeros(D,E,D);
dMds = zeros(E,D,D);    dSds = zeros(E,E,D,D);    dVds = zeros(D,E,D,D);
dMdi = zeros(E,n,D);    dSdi = zeros(E,E,n,D);    dVdi = zeros(D,E,n,D);
dMdt = zeros(E,n,E);    dSdt = zeros(E,E,n,E);    dVdt = zeros(D,E,n,E);
dMdX = zeros(E,D2+1,E); dSdX = zeros(E,E,D2+1,E); dVdX = zeros(D,E,D2+1,E);
k = zeros(n,E,D); bdX = zeros(n,E,D2); kdX = zeros(n,E,D2); Kdi = zeros(n,n,D);

% centralize training inputs
inp = bsxfun(@minus,input,m');

% 1) Predicted Mean and Input-Output Covariance *******************************
for i = 1:E
  inp2 = bsxfun(@rdivide,input,exp(X(i,1:D)));   % inp2 = input*sqrt(ilam)[nxD]
  ii = bsxfun(@rdivide,input,exp(2*X(i,1:D)));   %   ii = input*     ilam [nxD]
  for d = 1:D,                             % derivatives d(beta)/dX and d(K)/di
    KdX = KK(:,:,i,d).*bsxfun(@minus,inp2(:,d),inp2(:,d)').^2;
    bdX(:,i,d) = -iK(:,:,i)*KdX*beta(:,i);
    bdX(:,i,D+d) = -iK(:,:,i)*(2*KK(:,:,i,d))*beta(:,i);
    Kdi(:,:,d) = KK(:,:,i,d).*bsxfun(@minus,ii(:,d),ii(:,d)');
  end
  
  for d = 1:D
    % 1a) Compute the values **************************************************
    R = s(d,d) + exp(2*X(i,d)); iR = R\1;      %  R =            sdd + lamd [1]
    t = inp(:,d)*iR;                           %  t =              inp_d*iR [1]
    l = exp(-inp(:,d).*t/2);                   %  l = exp(-ilamd*inp_d.^2/2)[n]
    lb = l.*beta(:,i);                         % lb =               l.*beta [n]
    c = exp(2*X(i,D+d))*sqrt(iR)*exp(X(i,d));  %  c =   ad^2*(R*ilamd)^-0.5 [1]
    
    Md = c*sum(lb); M(i) = M(i) + Md;                          % predicted mean
    V(d,i) = c*t'*lb;                    % inv(s) times input-output covariance
    k(:,i,d) = 2*X(i,D+d) - inp(:,d).^2*exp(-2*X(i,d))/2;          % log-kernel
    
    % 1b) Compute the derivatives *********************************************
    tliK = t'*bsxfun(@times,l,iK(:,:,i));      % tliK = iR*inp_d'* (iK.*l)[1xn]
    liK = iK(:,:,i)*l;                         %  liK =             iK *l   [n]
    tlb = t.*lb;                               %  tlb = iR*inp_d.*l.*beta   [n]
    
    % ----------------------------------------------------- derivatives w.r.t s
    dMds(i,d,d) = c*t'*tlb/2 - iR*Md/2;                                     % M
    dVds(d,i,d,d) = c*(t.*t)'*tlb/2 - iR*V(d,i)/2 - V(d,i)*iR;              % V
    
    % --------------------------------------- derivatives w.r.t training-inputs
    for dd = 1:D                                                  % ... in beta
      tlbdi = Kdi(:,:,dd)*liK.*beta(:,i) ...                     %  tlbdi   [n]
                                    + Kdi(:,:,dd)*beta(:,i).*liK;
      tlbdi2 = -tliK*(-bsxfun(@times,Kdi(:,:,dd),beta(:,i))' ... % tlbdi2 [1xn]
                                  - diag(Kdi(:,:,dd)*beta(:,i)));
      dMdi(i,:,dd) = dMdi(i,:,dd) + c*tlbdi';                               % M
      dVdi(d,i,:,dd) = c*tlbdi2;                                            % V
    end                                                           % ... and c*l
    dMdi(i,:,d) = dMdi(i,:,d) - c*tlb';                                     % M
    dVdi(d,i,:,d) = dVdi(d,i,:,d) + reshape(c*(iR*lb' - t'.*tlb'),[1 1 n]); % V
    
    % -------------------------------------- derivatives w.r.t training-targets
    dMdt(i,:,i) = dMdt(i,:,i) + c*liK';                                     % M
    dVdt(d,i,:,i) = dVdt(d,i,:,i) + reshape(c*tliK,[1 1 n]);                % V
    
    % -------------------------- derivatives w.r.t length-scales and prior covs   
    kdX(:,i,d) = inp(:,d).^2*exp(-2*X(i,d));                            % log(k)
    kdX(:,i,D+d) = 2*ones(1,n);                                         % log(k)
    cdX_d = c*(1 - exp(2*X(i,d))*iR);                                   %     c
    tdX_d = inp(:,d)*( -iR*2.*exp(2*X(i,d))*iR );                       %     t
    ldX_d = exp(2*X(i,d))*l.*t.*t;                                      %     l
    for dd = 1:D                                                  % ... in beta
      dMdX(i,dd,i) = dMdX(i,dd,i) + c*sum(l.*bdX(:,i,dd));                  % M
      dMdX(i,D+dd,i) = dMdX(i,D+dd,i) + c*sum(l.*bdX(:,i,D+dd));            % M
      dVdX(d,i,dd,i) = c*(l.*bdX(:,i,dd))'*t;                               % V
      dVdX(d,i,D+dd,i) = c*(l.*bdX(:,i,D+dd))'*t;                           % V
    end                                                           % ... and c*l
    dMdX(i,d,i) = dMdX(i,d,i) + c*beta(:,i)'*ldX_d + sum(lb)*cdX_d;         % M
    dMdX(i,D+d,i) = dMdX(i,D+d,i) + 2*Md;                                   % M
    dVdX(d,i,d,i) = dVdX(d,i,d,i) ...                                       % V
                + c*(lb'*tdX_d + (ldX_d.*beta(:,i))'*t) + sum(lb.*t)*cdX_d;
    dVdX(d,i,D+d,i) = dVdX(d,i,D+d,i) + 2*V(d,i);                           % V
    
    % ------------------------------------------------- derivatives w.r.t noise
    dMdX(i,D2+1,i) = dMdX(i,D2+1,i) ...
                  - c*sum( l.*(2*exp(2*X(i,D2+1))*(iK(:,:,i)*beta(:,i))) ); % M
    dVdX(d,i,D2+1,i) = ...
                  -((l.*(2*exp(2*X(i,D2+1))*(iK(:,:,i)*beta(:,i))))'*t*c)'; % V
  end % d
end % i
% --------------------------------------------------------- derivatives w.r.t m
dMdm = V';                                                                  % M 
dVdm = 2*permute(dMds,[2 1 3]);                                             % V


% 2) Predictive Covariance Matrix *********************************************
for i = 1:E
  ii = bsxfun(@rdivide,inp,exp(2*X(i,1:D)));

  for j = 1:i
    ij = bsxfun(@rdivide,inp,exp(2*X(j,1:D)));
    BB = beta(:,i)*beta(:,j)';                        % BB = betai*betaj' [nxn]
    
    stL = zeros(n); stbLiKi = zeros(n,1); stbLiKj = stbLiKi;
    for d = 1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end
      %pp = 1; P = D;
      for e = 1:P
        % 2a) Compute the value ***********************************************
        if d==e % ----------------- combo of 1D kernels on same input dimension
          p = 1; pp = 1;                                       %     d==e:  p=1
          ii_ = ii(:,d); ij_ = ij(:,e);                        %  ii_/ij_   [n]
          sde = s(d,d);                                        %      sde   [1]
          eXi = exp(-2*X(i,d)); eXj = exp(-2*X(j,e));          % l-scales   [1]
        else % -------------- combo of 1D kernels on different input dimensions
          p = 2;                                               %     d~=e:  p=2
          ii_= [ii(:,d) zeros(n,1)]; ij_= [zeros(n,1) ij(:,e)];%  ii_/ij_ [nx2]
          sde = s([d e],[d e]);                                %      sde [2x2]
          eXi = [exp(-2*X(i,d)) 0]; eXj = [0 exp(-2*X(j,e))];  % l-scales [1x2]
        end
        R = sde*diag(eXi + eXj) + eye(p); iR = R\eye(p);          %     R [pxp]
        t = 1/sqrt(det(R));                                       %     t   [1]
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') ...              %     L [nxn]
                                    + maha(ii_,-ij_,R\sde/2));
        A = BB.*L;                             %    A = (betai*betaj').*L [nxn]
        ssA = sum(sum(A));                     %  ssA =    betai'*L*betaj   [1]
        S(i,j) = S(i,j) + pp*t*ssA; S(j,i) = S(i,j);     % predicted covariance
        
        % 2b) Compute the derivatives *****************************************
        zzi= ii_*(R\sde); zzj= ij_*(R\sde);    % zz{i/j} = i{i/j}_*iR*sde [nxp]
        zi = ii_/R; zj = ij_/R;                %  z{i/j} = i{i/j}_*iR     [nxp]
        bLiKi = iK(:,:,j)*(L'*beta(:,i));      %   bLiKi =   iKj*L'*betai   [n]
        bLiKj = iK(:,:,i)*(L *beta(:,j));      %   bLiKj =   iKi*L *betaj   [n]
        
        stL = stL + pp*t*L;                   %     stL = sum_{de} t*L      [1]
        stbLiKi = stbLiKi + pp*t*bLiKi;       % stbLiKi = sum_{de} t*bLiKi  [n]
        stbLiKj = stbLiKj + pp*t*bLiKj;       % stbLiKj = sum_{de} t*bLiKj  [n]
          
        Q2 = R\sde/2;                         %     Q2 =         iR*sde/2 [pxp]
        aQ = ii_*Q2; bQ = ij_*Q2;             % {a/b}Q = i{i/j}_*iR*sde/2 [nxp]
        
        T = zeros(p); SdXi = zeros(1,p); SdXj = SdXi;
        for dd = 1:p
          if dd==1, f=d; else f=e; end
        % ------------------------------------------- derivatives w.r.t m and s
          B = bsxfun(@plus,zi(:,dd),zj(:,dd)').*A;                    % B [nxn]
          dSdm(i,j,f) = dSdm(i,j,f) + pp*t*sum(sum(B));
          T(dd,1:dd) = sum(zi(:,1:dd)'*B,2) + sum(B*zj(:,1:dd))';     % T [pxp]
          T(1:dd,dd) = T(dd,1:dd)';
          
        % ---------------------------- derivatives w.r.t training-inputs (in L) 
          Z = eXi(dd)*(A*zzj(:,dd) + sum(A,2).*(zzi(:,dd)-inp(:,f)))...
           + eXj(dd)*(zzi(:,dd)'*A + sum(A,1).*(zzj(:,dd)-inp(:,f))')';
          dSdi(i,j,:,f) = squeeze(dSdi(i,j,:,f)) + pp*Z*t;
          
        % ------------------------------ derivatives w.r.t length-scales (in L)
          if i==j && d==e
            RTi =sde*(-2*eXi-2*eXj);                     %  RTi =  d(R)/dXi [1]
            diRi = -R\(RTi*iR);                          % diRi = d(iR)/dXi [1]
          else
            RTi = bsxfun(@times,sde,-2*eXi);                       %  RTi [pxp]
            RTj = bsxfun(@times,sde,-2*eXj);                       %  RTi [pxp]
            diRi = -R\bsxfun(@times,RTi(:,dd),iR(dd,:));           % diRi [pxp]
            diRj = -R\bsxfun(@times,RTj(:,dd),iR(dd,:));           % diRi [pxp]
            QdXj = diRj*sde/2;                                     % QdXi [pxp]
          end
          QdXi = diRi*sde/2;                             % QdXi = d(Q2)/dXi [1]
          
          if i==j && d==e
            aQdi = ii_*QdXi - 2*aQ;                   %  aQdi = d(ii*Q)/dXi [n]
            saQdi = aQdi.*ii_ - 2.*aQ.*ii_;           % saQdi               [n]
            saQdj = saQdi;                            % saQdj               [n]
            sbQdi = saQdi;                            % sbQdi               [n]
            sbQdj = sbQdi;                            % sbQdj               [n]
            m2dXi = -2*aQdi*ii_' + 2*(bsxfun(@times,aQ,ii_') ...
                     + bsxfun(@times,ii_,aQ'));  % 2nd bit of d(maha)/dXi [nxn]
            m2dXj = m2dXi;                       % 2nd bit of d(maha)/dXj [nxn]
          else
            aQdi = ii_*QdXi + bsxfun(@times,-2*ii_(:,dd),Q2(dd,:));     % [nxp]
            aQdj = ii_*QdXj;                                            % [nxp]
            bQdi = ij_*QdXi;                                            % [nxp]
            bQdj = ij_*QdXj + bsxfun(@times,-2*ij_(:,dd),Q2(dd,:));     % [nxp]
            
            saQdi = sum(aQdi.*ii_,2) - 2.*aQ(:,dd).*ii_(:,dd);          %   [n]
            saQdj = sum(aQdj.*ii_,2);                                   %   [n]
            sbQdi = sum(bQdi.*ij_,2);                                   %   [n]
            sbQdj = sum(bQdj.*ij_,2) - 2.*bQ(:,dd).*ij_(:,dd);          %   [n]
            m2dXi = -2*aQdi*ij_';                % 2nd bit of d(maha)/dXi [nxn]
            m2dXj = -2*ii_*bQdj';                % 2nd bit of d(maha)/dXj [nxn]
          end
      
          m1dXi = bsxfun(@plus,saQdi,sbQdi');    % 1st bit of d(maha)/dXi [nxn]
          m1dXj = bsxfun(@plus,saQdj,sbQdj');    % 1st bit of d(maha)/dXj [nxn]
          mdXi = m1dXi - m2dXi;                      % mdXi = d(maha)/dXi [nxn]
          mdXj = m1dXj - m2dXj;                      % mdXj = d(maha)/dXj [nxn]
          
          if i==j && d==e
            LdXi = L.*(mdXi + bsxfun(@plus,kdX(:,i,f),kdX(:,j,f)'));    % [nxn]
            SdXi = beta(:,i)'*LdXi*beta(:,j);                           %   [1]
          else
            LdXi = L.*(mdXi + bsxfun(@plus,kdX(:,i,f),zeros(n,1)'));    % [nxn]
            LdXj = L.*(mdXj + bsxfun(@plus,zeros(n,1),kdX(:,j,f)'));    % [nxn]
            SdXi(dd) = beta(:,i)'*LdXi*beta(:,j);                       % [1xp]
            SdXj(dd) = beta(:,i)'*LdXj*beta(:,j);                       % [1xp]
          end
        end % dd
        
        % ------------------------------ derivatives w.r.t length-scales (in t)
        if i==j && d==e
          tdX = -0.5*t*sum(iR'.*bsxfun(@times,sde,-2*eXi-2*eXj));  %  tdX   [1]
          dSdX(i,i,d,i) = dSdX(i,i,d,i) + t*SdXi + tdX*ssA;
        else
          tdXi = -0.5*t*sum(iR'.*bsxfun(@times,sde,-2*eXi));       % tdXi [1xp]
          tdXj = -0.5*t*sum(iR'.*bsxfun(@times,sde,-2*eXj));       % tdXj [1xp]
          dSdX(i,j,d,i) = dSdX(i,j,d,i) + pp*t*SdXi(1) + pp*tdXi(1)*ssA;
          dSdX(i,j,e,j) = dSdX(i,j,e,j) + pp*t*SdXj(p) + pp*tdXj(p)*ssA;
        end
                  
        % ------------------------------------------------- derivatives w.r.t s
        T = (t*T - t*ssA*diag(eXi + eXj)/R)/2;                        % T [pxp]
        if d==e, dSds(i,j,d,d) = dSds(i,j,d,d) + T;
        else     dSds(i,j,[d e],[d e]) = squeeze(dSds(i,j,[d e],[d e])) + pp*T;
        end
        
        % --------------------------------- derivatives w.r.t prior covs (in t)
        dSdX(i,j,D+d,i) = dSdX(i,j,D+d,i) + pp*2*t*ssA;
        dSdX(i,j,D+e,j) = dSdX(i,j,D+e,j) + pp*2*t*ssA;
        
      end % e
    end % d
    
    % ----------------------------- derivatives w.r.t training-inputs (in beta) 
    ZZ = zeros(n,D);  
    for dd = 1:D
      Q = bsxfun(@minus,inp(:,dd),inp(:,dd)');                        % Q [nxn]
      B = KK(:,:,i,dd).*Q;                                            % B [nxn]
      ZZ(:,dd) = exp(-2*X(i,dd))*(B*beta(:,i).*stbLiKj+beta(:,i).*(B*stbLiKj));
      if i~=j; B = KK(:,:,j,dd).*Q; end
      ZZ(:,dd) = ZZ(:,dd) ...
               + exp(-2*X(j,dd))*(stbLiKi.*(B*beta(:,j))+B*stbLiKi.*beta(:,j));            
    end
    dSdi(i,j,:,:) = shiftdim(dSdi(i,j,:,:),2) + ZZ;
    
    % -------- derivatives w.r.t training-targets and hyperparameters (in beta)
    if i==j
      dSdt(i,i,:,i) = iK(:,:,i)*(stL+stL')*beta(:,i);
      dSdX(i,i,1:D2,i) = squeeze(dSdX(i,i,1:D2,i)) ...
                              + reshape(bdX(:,i,:),n,D2)'*(stL+stL')*beta(:,i);
    else
      dSdt(i,j,:,i) = iK(:,:,i)*stL *beta(:,j);
      dSdt(i,j,:,j) = iK(:,:,j)*stL'*beta(:,i);
      dSdX(i,j,1:D2,i) = squeeze(dSdX(i,j,1:D2,i)) ...
                                  + reshape(bdX(:,i,:),n,D2)'*(stL *beta(:,j));
      dSdX(i,j,1:D2,j) = squeeze(dSdX(i,j,1:D2,j)) ...
                                  + reshape(bdX(:,j,:),n,D2)'*(stL'*beta(:,i));
    end
    dSdX(i,j,D2+1,i)=dSdX(i,j,D2+1,i) -2*exp(2*X(i,D2+1))*(beta(:,i)'*stbLiKj);
    dSdX(i,j,D2+1,j)=dSdX(i,j,D2+1,j) -2*exp(2*X(j,D2+1))*(beta(:,j)'*stbLiKi);
    
    % ------------------------------------------- centralise moment derivatives
    dSdm(i,j,:)   =shiftdim(dSdm(i,j,:)  ,1)-M(i)*dMdm(j,:)  -M(j)*dMdm(i,:);
    dSds(i,j,:,:) =shiftdim(dSds(i,j,:,:),1)-M(i)*dMds(j,:,:)-M(j)*dMds(i,:,:);
    dSdi(i,j,:,:) =shiftdim(dSdi(i,j,:,:),1)-M(i)*dMdi(j,:,:)-M(j)*dMdi(i,:,:);
    dSdt(i,j,:,i) =shiftdim(dSdt(i,j,:,i),1)-M(j)*dMdt(i,:,i);
    dSdt(i,j,:,j) =shiftdim(dSdt(i,j,:,j),1)-M(i)*dMdt(j,:,j);
    dSdX(i,j,:,i) =shiftdim(dSdX(i,j,:,i),1)-M(j)*dMdX(i,:,i);
    dSdX(i,j,:,j) =shiftdim(dSdX(i,j,:,j),1)-M(i)*dMdX(j,:,j);

    % ---------------------------------------------- fill in the symmetric bits
    if i~=j
      dSdm(j,i,:)   = dSdm(i,j,:);
      dSds(j,i,:,:) = dSds(i,j,:,:);
      dSdi(j,i,:,:) = dSdi(i,j,:,:);
      dSdt(j,i,:,:) = dSdt(i,j,:,:);
      dSdX(j,i,:,:) = dSdX(i,j,:,:);
    end
    
  end % j
end % i

S = S - M*M' + 1e-6*eye(E);               % centralise moments...and add jitter

dMds=reshape(dMds,[E D*D]);                             % vectorise derivatives
dSdm=reshape(dSdm,[E*E D]); dSds=reshape(dSds,[E*E D*D]);
dVdm=reshape(dVdm,[D*E D]); dVds=reshape(dVds,[D*E D*D]);
dMdi=reshape(dMdi,E,[]);  dMdt=reshape(dMdt,E,[]);  dMdX=reshape(dMdX,E,[]);
dSdi=reshape(dSdi,E*E,[]);dSdt=reshape(dSdt,E*E,[]);dSdX=reshape(dSdX,E*E,[]);
dVdi=reshape(dVdi,D*E,[]);dVdt=reshape(dVdt,D*E,[]);dVdX=reshape(dVdX,D*E,[]);