function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpasd(gpmodel, m, s)

% Compute single predictions for multiple GPs with uncertain inputs and
% obtain the joint distribution using an approximation of joint Gaussianity
% between input and output. This approximates the output covariance matrix.
% If dynmodel.nigp exists, individual noise contributions are added.
% Predictive variances contain uncertainty about the function, but no
% noise.
%
% dynmodel  dynamics model struct
%   hyp     log-hyper-parameters                               [D+2 x  E ]
%   inputs  training inputs                                    [ n  x  D ]
%   target  training targets                                   [ n  x  E ]
%   nigp    (optional) individual noise variance terms         [ n  x  E ]
% m         mean of the test distribution                      [ D       ]
% s         covariance matrix of the test distribution         [ D  x  D ]
%
% M         mean of pred. distribution                         [ E       ]
% S         covariance of the pred. distribution               [ E  x  E ]
% V         inv(s) times covariance between input and output   [ D  x  E ]
% dMdm      output mean by input mean                          [ E  x  D ]
% dSdm      output covariance by input mean                    [E*E x  D ]
% dVdm      inv(s)*input-output covariance by input mean       [D*E x  D ]
% dMds      ouput mean by input covariance                     [ E  x D*D]
% dSds      output covariance by input covariance              [E*E x D*D]
% dVds      inv(s)*input-output covariance by input covariance [D*E x D*D]
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
%                                Joe Hall and Andrew McHutchon, 2014-01-17

if isfield(gpmodel,'induce') && numel(gpmodel.induce)>0; x = gpmodel.induce; 
else x = gpmodel.inputs; end

[~, D, pE] = size(x); E = size(gpmodel.beta,2);
h = gpmodel.hyp; iK = gpmodel.iK; beta = gpmodel.beta;

M = zeros(E,1); V = zeros(D,E); Si = zeros(E,1);      % initialize
dMds = zeros(E,D,D); dSidm = zeros(E,D); M1 = zeros(E,1);
dSids = zeros(E,D^2); dVds = zeros(D,E,D,D); T = zeros(D);
dM1dm = zeros(E,D);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E
iL = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(iL,[3,1,2])); % N-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  xi = x(:,:,min(i,pE));
  il = diag(exp(-h(i).l));                  % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;               % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D); liBl = il/B*il;       % liBl = (Lambda + S)^-1
  
  t = in/B;                                 % in.*t = (x-m) (S+L)^-1 (x-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il; tlb = bsxfun(@times,tL,lb);
  c = exp(2*h(i).s)/sqrt(det(B));           % sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;        % predicted mean
  V1 = tL'*lb*c; V(:,i) = V1 + h(i).m;   % inv(s) times input-output covariance
  dMds(i,:,:) = c*tL'*tlb/2 - liBl*M1(i)/2;
  for d = 1:D
    dVds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - liBl*V1(d)/2 - V1*liBl(d,:);
  end
   k = 2*h(i).s-sum(in.*in,2)/2;
   xm = xi'*lb*c; L2iBL = diag(exp(2*h(i).l))*liBl; 
   LiBLxm = liBl*xm; sLx = s*liBl*xi';
   a = L2iBL*m*M1(i) + s*LiBLxm;
   dadm = L2iBL*(M1(i)*eye(D) + m*V1') + s*liBl*xi'*bsxfun(@times,tL,lb)*c;
   dads = -bsxfun(@times,L2iBL,permute(liBl*m,[2,3,1]))*M1(i) ...
                                          + bsxfun(@times,L2iBL*m,dMds(i,:,:));
   for d=1:D; dads(d,d,:) = dads(d,d,:) + permute(LiBLxm,[2,3,1]); end
   tLtlb = bsxfun(@times,tL,permute(tlb,[1,3,2])); 
   Lbc = bsxfun(@times,lb,permute(liBl*c,[3,1,2]));
   dads = dads - bsxfun(@times,s*liBl,permute(LiBLxm,[2,3,1]))...
                                + etprod('123',sLx,'14',c*tLtlb-Lbc,'423')/2;
   
   dM1dm(i,:) = V1;

  % 2. Compute predictive covariance (non-central moments) ---------------
  R = 2*s*diag(iL(:,i))+eye(D); t = 1/sqrt(det(R));
  L = exp(bsxfun(@plus,k,k')+maha(inpil(:,:,i),-inpil(:,:,i),R\s/2));
  iKL = iK(:,:,i).*L; s1iKL = sum(iKL,1); s2iKL = sum(iKL,2);
  S1 = t*(beta(:,i)'*L*beta(:,i) - sum(s1iKL));            % covariance
  Si(i) = S1 + 2*h(i).m'*(a - m*M1(i));
  
  zi = inpil(:,:,i)/R;
  bibLi = L'*beta(:,i).*beta(:,i); cbLi = L'*bsxfun(@times,beta(:,i),zi);
  dSidm(i,:) = (bibLi'*zi*2 - (s2iKL' + s1iKL)*zi)*t;               % dSdm
  dSidm(i,:) = dSidm(i,:) + 2*h(i).m'*(dadm - M1(i)*eye(D) - m*dM1dm(i,:));
  
  for d = 1:D
    T(d,1:d) = 2*(zi(:,1:d)'*(zi(:,d).*bibLi) + ...
       cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) ...
                                              - zi(:,1:d)'*(iKL*zi(:,d)));
    T(1:d,d) = T(d,1:d)';
  end
  dSids(i,:) = reshape( (t*T-S1*diag(2*iL(:,i))/R)/2,[1 D*D]);% dSds
  dSids(i,:) = dSids(i,:) + 2*h(i).m'*reshape(dads,D,D^2)...
          - 2*reshape(h(i).m'*m*dMds(i,:,:),1,D^2) ...
          + reshape(h(i).m*h(i).m',1,D^2);
end

dVTdm = reshape(2*dMds,[E*D D]);                   % vectorise derivatives
dVTds = reshape(permute(dVds,[2 1 3 4]),[D*E D*D]);
dMdm = V'; dVdm = reshape( 2*permute(dMds,[2 1 3]), [D*E D]);
dMds = reshape(dMds,[E D*D]); dVds = reshape(dVds,[D*E D*D]);
dSidm = dSidm - 2*bsxfun(@times,M1,dM1dm);           % centralise covariance
dSids = dSids - 2*bsxfun(@times,M1,dMds);

% 3. Compute cross covariances -------------------------------------------
S = V'*s*V; S = (S+S')/2;                    % symmetrize
S(eye(E)==1) = Si + exp(2*[h.s]') - M1.^2 + diag([h.m]'*s*[h.m]);

dSdm = kron(eye(E),V'*s)*dVdm + kron(V'*s,eye(E))*dVTdm;
dSds = kron(eye(E),V'*s)*dVds + kron(V'*s,eye(E))*dVTds + kron(V,V)';
dSdm(1:E+1:E*E,:) = dSidm; dSds(1:E+1:E*E,:) = dSids;

X = reshape(1:E*E,[E E]); XT = X'; I = triu(ones(E),1);    % symmetrise dS
ij=X(I==1)'; ji=XT(I==1)'; dSdm(ji,:)=dSdm(ij,:);  dSds(ji,:)=dSds(ij,:);