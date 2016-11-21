function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
                                                   = conGaussAddd(policy, m, s)
% Compute joint predictions for multiple Radial Basis Functions (RBFs) with
% first order additive squared exponential kernels and uncertain inputs.
%
% inputs:
% policy    policy parameter struct
%   .p      policy parameters being optimized
%     .cen  centres of the basis functions                          [ n  x  D ]
%     .ll   log lengthscales for basis functions (1 per dim)        [ D  x  E ]
%     .ldw  log sqrt weights for each dimension                     [ D  x  E ]
%     .w    basis function weights                                  [ n  x  E ]
% m         mean of the test distribution                           [ D       ] 
% s         covariance matrix of the test distribution              [ D  x  D ] 
%
% outputs:
% M       mean of control distribution                              [ E       ] 
% S       covariance of the control distribution                    [ E  x  E ]
% C       inv(s) times covariance between input and controls        [ D  x  E ]
% dMdm    output mean   by input mean partial derivatives           [ E  x  D ]
% dSdm    output cov    by input mean derivatives                   [E*E x  D ]
% dCdm    inv(s)*io-cov by input mean derivatives                   [D*E x  D ]
% dMds    ouput mean    by input covariance derivatives             [ E  x D*D]
% dSds    output cov    by input covariance derivatives             [E*E x D*D] 
% dCds    inv(s)*io-cov by input covariance derivatives             [D*E x D*D]
%
% dMdp    output mean   by policy parameters                        [ E  x  P ] 
% dSdp    output cov    by policy parameters                        [E*E x  P ] 
% dCdp    inv(s)*io-cov by policy parameters                        [D*E x  P ]
%
% where P = n*E + 2*D*E ( + n*D if centres are not fixed)
%
% Copyright (C) 2008-2012 by Marc Deisenroth and Carl Edward Rasmussen, 
% 2012-01-18, Edited by Joe Hall & Andrew McHutchon 2012-06-26

if nargout < 4; [M S C] = conGaussAdd(policy,m,s); return; end % no derivatives
w = policy.p.w; ll = policy.p.ll; ldw = policy.p.ldw;
if isfield(policy,'cen'); cen = policy.cen; fxdcen = 1; 
else cen = policy.p.cen; fxdcen = 0; end

[n, D] = size(cen);                 % no. of bases and dimension of input space
E = size(w,2);                                             % number of controls
D2 = 2*D;

% initializations
M = zeros(E,1); S = zeros(E); C = zeros(D,E); k = zeros(n,E,D);
dSdm = zeros(E,E,D); dMds = zeros(E,D,D); dSds = zeros(E,E,D,D); 
dCds = zeros(D,E,D,D); dMdc = zeros(E,n,D); dSdc = zeros(E,E,n,D);
dCdc = zeros(D,E,n,D); dMdldw = zeros(E,D,E); dSdldw = zeros(E,E,D,E);
dCdldw = zeros(D,E,2,E); dMdll = zeros(E,D,E); dSdll = zeros(E,E,D,E);
dCdll = zeros(D,E,D,E); dMdw = zeros(E,n,E); dSdw = zeros(E,E,n,E);
dCdw = zeros(D,E,n,E); kdX = zeros(n,E,D2); sde = zeros(2); eXi = zeros(1,2);
eXj = zeros(1,2); R = zeros(2); iR = zeros(2); ii_ = zeros(n,2);
ij_ = zeros(n,2); zzi = zeros(n,2); zzj = zeros(n,2); zi = zeros(n,2);
zj = zeros(n,2); T = zeros(2); Q2 = zeros(2); aQ = zeros(n,2); bQ = zeros(n,2);
RTi = zeros(2); diRi = zeros(2); RTj = zeros(2); diRj = zeros(2);
QdXi = zeros(2); QdXj = zeros(2); aQdi = zeros(n,2); aQdj = zeros(n,2);
bQdi = zeros(n,2); bQdj = zeros(n,2); SdXi = zeros(1,2); SdXj = zeros(1,2);
tdlli = zeros(1,2); tdllj = zeros(1,2);

inp = bsxfun(@minus,cen,m');      % move basis function centres about input mean

% 1) Predicted Mean and Input-Output Covariance *******************************
for i = 1:E
  for d = 1:D
    % 1a) Compute the values **************************************************
    RR = s(d,d) + exp(2*ll(d,i)); iRR = RR\1;  % RR =            sdd + lamd [1]
    tt = inp(:,d)*iRR;                         % tt =             inp_d*iRR [1]
    l = exp(-inp(:,d).*tt/2);                  %  l = exp(-ilamd*inp_d.^2/2)[n]
    lb = l.*w(:,i);                         % lb =               l.*w [n]
    c = exp(2*ldw(d,i))*sqrt(iRR)*exp(ll(d,i));%  c =  ad^2*(RR*ilamd)^-0.5 [1]
    
    Md = c*sum(lb); M(i) = M(i) + Md;                          % predicted mean
    C(d,i) = c*tt'*lb;                   % inv(s) times input-output covariance
    k(:,i,d) = 2*ldw(d,i) - inp(:,d).^2*exp(-2*ll(d,i))/2;         % log-kernel
    
    % 1b) Compute the derivatives *********************************************
    tlb = tt.*lb;                                %  tlb = iRR*inp_d.*l.*w   [n]
    
    % ----------------------------------------------------- derivatives w.r.t s
    dMds(i,d,d) = c*tt'*tlb/2 - iRR*Md/2;                                   % M
    dCds(d,i,d,d) = c*(tt.*tt)'*tlb/2 - iRR*C(d,i)/2 - C(d,i)*iRR;          % V
    
    % -------------------------------- derivatives w.r.t basis-function weights
    dMdw(i,:,i) = dMdw(i,:,i) + c*l';                                       % M
    dCdw(d,i,:,i) = c*tt.*l;                                                % V
    
    % -------------------------- derivatives w.r.t length-scales and prior covs
    kdX(:,i,d) = inp(:,d).^2*exp(-2*ll(d,i));                          % log(k)
    kdX(:,i,D+d) = 2*ones(1,n);                                        % log(k)
    cdX_d = c*(1 - exp(2*ll(d,i))*iRR);                                 %     c
    tdX_d = inp(:,d)*( -iRR*2.*exp(2*ll(d,i))*iRR );                    %     t
    ldX_d = exp(2*ll(d,i))*l.*tt.*tt;                                   %     l
    
    dMdll(i,d,i) = c*w(:,i)'*ldX_d + sum(lb)*cdX_d;                         % M
    dMdldw(i,d,i) = 2*Md;                                                   % M
    dCdll(d,i,d,i) =c*(lb'*tdX_d+(ldX_d.*w(:,i))'*tt)+sum(lb.*tt)*cdX_d;    % V
    dCdldw(d,i,d,i) = 2*C(d,i);                                             % V
    
    % --------------------------------------- derivatives w.r.t bases' centres
    if ~fxdcen
      dMdc(i,:,d) = -c*tlb';                                                % M
      dCdc(d,i,:,d) = c*(iRR*lb' - tt'.*tlb');                              % V
    end
    
  end % d
end % i

% --------------------------------------------------------- derivatives w.r.t m
dMdm = C';                                                                  % M 
dCdm = 2*permute(dMds,[2 1 3]);                                             % V


% 2) Control Covariance Matrix ************************************************
for i = 1:E
  ii = bsxfun(@rdivide,inp,exp(2*ll(:,i)'));

  for j = 1:i
    ij = bsxfun(@rdivide,inp,exp(2*ll(:,j)'));
    ww = w(:,i)*w(:,j)';                                                % [nxn]
    
    stL = zeros(n);
    for d = 1:D
      if i==j, P = d; pp = 2; else P = D; pp = 1; end
      for e = 1:P
        % 2a) Compute the value ***********************************************
        if d==e % ----------------- combo of 1D kernels on same input dimension
          p = 1; pp = 1;                                       %     d==e:  p=1
          ii_(:,1) = ii(:,d); ij_(:,1) = ij(:,e);              %  ii_/ij_   [n]
          sde(1,1) = s(d,d);                                   %      sde   [1]
          eXi(1) = exp(-2*ll(d,i)); eXj(1) = exp(-2*ll(e,j));  % l-scales   [1]
          R(1,1) = sde(1,1)*diag(eXi(1)+eXj(1))+1; iR(1,1) = 1/R(1,1);% R [pxp]
        else % -------------- combo of 1D kernels on different input dimensions
          p = 2;                                               %     d~=e:  p=2
          ii_= [ii(:,d) zeros(n,1)]; ij_= [zeros(n,1) ij(:,e)];%  ii_/ij_ [nx2]
          sde = s([d e],[d e]);                                %      sde [2x2]
          eXi = [exp(-2*ll(d,i)) 0]; eXj = [0 exp(-2*ll(e,j))];% l-scales [1x2]
          R = sde*diag(eXi+eXj)+eye(2); iR=R\eye(2);           %        R [pxp]
        end
        t = 1/sqrt(det(R(1:p,1:p)));                              %     t   [1]
        L = exp(bsxfun(@plus,k(:,i,d),k(:,j,e)') ...              %     L [nxn]
                    + maha(ii_(:,1:p),-ij_(:,1:p),R(1:p,1:p)\sde(1:p,1:p)/2));
        A = ww.*L;                                   %    A = (wi*wj').*L [nxn]
        ssA = sum(sum(A));                           %  ssA =    wi'*L*wj   [1]
        S(i,j) = S(i,j) + pp*t*ssA; S(j,i) = S(i,j);     % predicted covariance
        
        % 2b) Compute the derivatives *****************************************
        zzi(:,1:p) = ii_(:,1:p)*(R(1:p,1:p)\sde(1:p,1:p));
        zzj(:,1:p) = ij_(:,1:p)*(R(1:p,1:p)\sde(1:p,1:p));
        zi(:,1:p) = ii_(:,1:p)/R(1:p,1:p);
        zj(:,1:p) = ij_(:,1:p)/R(1:p,1:p);
        stL = stL + pp*t*L;                            % stL = sum_{de} t*L [1]
          
        Q2(1:p,1:p) = R(1:p,1:p)\sde(1:p,1:p)/2;
        aQ(:,1:p) = ii_(:,1:p)*Q2(1:p,1:p);
        bQ(:,1:p) = ij_(:,1:p)*Q2(1:p,1:p);
        
        for dd = 1:p
          if dd==1, f=d; else f=e; end
        % ------------------------------------------- derivatives w.r.t m and s
          B = bsxfun(@plus,zi(:,dd),zj(:,dd)').*A;                    % B [nxn]
          dSdm(i,j,f) = dSdm(i,j,f) + pp*t*sum(sum(B));
          T(dd,1:dd) = sum(zi(:,1:dd)'*B,2) + sum(B*zj(:,1:dd))';     % T [pxp]
          T(1:dd,dd) = T(dd,1:dd)';
          
        % ------------------------------------ derivatives w.r.t centres (in L)
          Z = eXi(dd)*(A*zzj(:,dd) + sum(A,2).*(zzi(:,dd)-inp(:,f)))...
           + eXj(dd)*(zzi(:,dd)'*A + sum(A,1).*(zzj(:,dd)-inp(:,f))')';
          dSdc(i,j,:,f) = squeeze(dSdc(i,j,:,f)) + pp*Z*t;
          
        % ------------------------------ derivatives w.r.t length-scales (in L)
          if i==j && d==e
            RTi(1,1) = sde(1,1)*(-2*eXi(1)-2*eXj(1));
            diRi(1,1) = -R(1,1)\(RTi(1,1)*iR(1,1));
          else
            RTi(1:p,1:p) = bsxfun(@times,sde(1:p,1:p),-2*eXi(1:p));
            RTj(1:p,1:p) = bsxfun(@times,sde(1:p,1:p),-2*eXj(1:p));
            diRi(1:p,1:p) =-R(1:p,1:p)\bsxfun(@times,RTi(1:p,dd),iR(dd,1:p));
            diRj(1:p,1:p) =-R(1:p,1:p)\bsxfun(@times,RTj(1:p,dd),iR(dd,1:p));
            QdXj(1:p,1:p) = diRj(1:p,1:p)*sde(1:p,1:p)/2;
          end
          QdXi(1:p,1:p) = diRi(1:p,1:p)*sde(1:p,1:p)/2;
          
          if i==j && d==e
            aQdi(:,1) = ii_(:,1)*QdXi(1,1) - 2*aQ(:,1);
            saQdi = aQdi(:,1).*ii_(:,1) - 2.*aQ(:,1).*ii_(:,1);
            saQdj = saQdi;
            sbQdi = saQdi;
            sbQdj = sbQdi;
            m2dXi = -2*aQdi(:,1)*ii_(:,1)' + 2*(bsxfun(@times,aQ(:,1),...
                                ii_(:,1)') + bsxfun(@times,ii_(:,1),aQ(:,1)'));
            m2dXj = m2dXi;
          else
            aQdi(:,1:p) = ii_(:,1:p)*QdXi(1:p,1:p) ...
                                      + bsxfun(@times,-2*ii_(:,dd),Q2(dd,1:p));
            aQdj(:,1:p) = ii_(:,1:p)*QdXj(1:p,1:p);
            bQdi(:,1:p) = ij_(:,1:p)*QdXi(1:p,1:p);
            bQdj(:,1:p) = ij_(:,1:p)*QdXj(1:p,1:p) ...
                                      + bsxfun(@times,-2*ij_(:,dd),Q2(dd,1:p));
            
            saQdi = sum(aQdi(:,1:p).*ii_(:,1:p),2) - 2.*aQ(:,dd).*ii_(:,dd);
            saQdj = sum(aQdj(:,1:p).*ii_(:,1:p),2);
            sbQdi = sum(bQdi(:,1:p).*ij_(:,1:p),2);
            sbQdj = sum(bQdj(:,1:p).*ij_(:,1:p),2) - 2.*bQ(:,dd).*ij_(:,dd);
            m2dXi = -2*aQdi(:,1:p)*ij_(:,1:p)';  % 2nd bit of d(maha)/dXi [nxn]
            m2dXj = -2*ii_(:,1:p)*bQdj(:,1:p)';  % 2nd bit of d(maha)/dXj [nxn]
          end
      
          m1dXi = bsxfun(@plus,saQdi,sbQdi');    % 1st bit of d(maha)/dXi [nxn]
          m1dXj = bsxfun(@plus,saQdj,sbQdj');    % 1st bit of d(maha)/dXj [nxn]
          mdXi = m1dXi - m2dXi;                      % mdXi = d(maha)/dXi [nxn]
          mdXj = m1dXj - m2dXj;                      % mdXj = d(maha)/dXj [nxn]
          
          if i==j && d==e
            LdXi = L.*(mdXi + bsxfun(@plus,kdX(:,i,f),kdX(:,j,f)'));    % [nxn]
            SdXi(1) = w(:,i)'*LdXi*w(:,j);                              %   [1]
          else
            LdXi = L.*(mdXi + bsxfun(@plus,kdX(:,i,f),zeros(n,1)'));    % [nxn]
            LdXj = L.*(mdXj + bsxfun(@plus,zeros(n,1),kdX(:,j,f)'));    % [nxn]
            SdXi(dd) = w(:,i)'*LdXi*w(:,j);                             % [1xp]
            SdXj(dd) = w(:,i)'*LdXj*w(:,j);                             % [1xp]
          end
        end % dd
        
        % --------------------------- derivatives w.r.t log lengthscales (in t)
        if out > 1
          if i==j && d==e
            tdll = -0.5*t*sum(iR(1,1)'...
                                 .*bsxfun(@times,sde(1,1),-2*eXi(1)-2*eXj(1)));
            dSdll(i,i,d,i) = dSdll(i,i,d,i) + t*SdXi(1) + tdll*ssA;
          else
            tdlli(1:p) = -0.5*t*sum(iR(1:p,1:p)'...
                   .*bsxfun(@times,sde(1:p,1:p),-2*eXi(1:p)));     % tdXi [1xp]
            tdllj(1:p) = -0.5*t*sum(iR(1:p,1:p)'...
                   .*bsxfun(@times,sde(1:p,1:p),-2*eXj(1:p)));     % tdXj [1xp]
            dSdll(i,j,d,i) = dSdll(i,j,d,i) + pp*t*SdXi(1) + pp*tdlli(1)*ssA;
            dSdll(i,j,e,j) = dSdll(i,j,e,j) + pp*t*SdXj(p) + pp*tdllj(p)*ssA;
          end
          
          dSdldw(i,j,d,i) = dSdldw(i,j,d,i) + pp*2*t*ssA;
          dSdldw(i,j,e,j) = dSdldw(i,j,e,j) + pp*2*t*ssA;
        end
                  
        % ------------------------------------------------- derivatives w.r.t s
        T(1:p,1:p) = (t*T(1:p,1:p) - t*ssA*diag(eXi(1:p) ...
                                          + eXj(1:p))/R(1:p,1:p))/2;  % T [pxp]
        if d==e, dSds(i,j,d,d) = dSds(i,j,d,d) + T(1,1);
        else     dSds(i,j,[d e],[d e]) = squeeze(dSds(i,j,[d e],[d e])) + pp*T;
        end
        
      end % e
    end % d
    
    % -------------------------------- derivatives w.r.t basis function weights
    if i==j
      dSdw(i,i,:,i) = (stL+stL')*w(:,i);
    else
      dSdw(i,j,:,i) = stL *w(:,j);
      dSdw(i,j,:,j) = stL'*w(:,i);
    end
    
    % ------------------------------------------- centralise moment derivatives
    dSdm(i,j,:)   =shiftdim(dSdm(i,j,:)  ,1)-M(i)*dMdm(j,:)  -M(j)*dMdm(i,:);
    dSds(i,j,:,:) =shiftdim(dSds(i,j,:,:),1)-M(i)*dMds(j,:,:)-M(j)*dMds(i,:,:);
    dSdc(i,j,:,:) =shiftdim(dSdc(i,j,:,:),1)-M(i)*dMdc(j,:,:)-M(j)*dMdc(i,:,:);
    dSdw(i,j,:,i) =shiftdim(dSdw(i,j,:,i),1)-M(j)*dMdw(i,:,i);
    dSdw(i,j,:,j) =shiftdim(dSdw(i,j,:,j),1)-M(i)*dMdw(j,:,j);
    dSdll(i,j,:,i) =shiftdim(dSdll(i,j,:,i),1)-M(j)*dMdll(i,:,i);
    dSdll(i,j,:,j) =shiftdim(dSdll(i,j,:,j),1)-M(i)*dMdll(j,:,j);
    dSdldw(i,j,:,i) =shiftdim(dSdldw(i,j,:,i),1)-M(j)*dMdldw(i,:,i);
    dSdldw(i,j,:,j) =shiftdim(dSdldw(i,j,:,j),1)-M(i)*dMdldw(j,:,j);

    % ---------------------------------------------- fill in the symmetric bits
    if i~=j
      dSdm(j,i,:)   = dSdm(i,j,:);
      dSds(j,i,:,:) = dSds(i,j,:,:);
      dSdc(j,i,:,:) = dSdc(i,j,:,:);
      dSdw(j,i,:,:) = dSdw(i,j,:,:);
      dSdll(j,i,:,:) = dSdll(i,j,:,:);
      dSdldw(j,i,:,:) = dSdldw(i,j,:,:);
    end
    
  end % j
end % i

S = S - M*M' + 1e-6*eye(E);               % centralise moments...and add jitter

% concatentate derivatives
dMds = reshape(dMds,[E D*D]);                           % vectorise derivatives
dSdm = reshape(dSdm,[E*E D]); dSds = reshape(dSds,[E*E D*D]);
dCdm = reshape(dCdm,[D*E D]); dCds = reshape(dCds,[D*E D*D]);
dMdp = [reshape(dMdldw,[E D*E])   reshape(dMdll,[E D*E])   ...
                                                        reshape(dMdw,[E n*E])];
dSdp = [reshape(dSdldw,[E*E D*E]) reshape(dSdll,[E*E D*E]) ...
                                                      reshape(dSdw,[E*E n*E])];
dCdp = [reshape(dCdldw,[D*E D*E]) reshape(dCdll,[D*E D*E]) ...
                                                      reshape(dCdw,[D*E n*E])];
if ~fxdcen; 
  dMdp = [reshape(dMdc,[E n*D])   dMdp];
  dSdp = [reshape(dSdc,[E*E n*D]) dSdp];
  dCdp = [reshape(dCdc,[D*E n*D]) dCdp];
end