function [M, S, dMdm, dSdm, dMds, dSds, dMdp, dSdp,a,dadl] = gpD(gpmodel, m, s)
% Compute joint predictions for multiple GPs with uncertain inputs. This version
% uses a linear + bias mean function. Predictive variances contain uncertainty 
% about the function, but no noise.
%
% dynmodel  dynamics model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%     .m    D-by-1 linear weights for the GP mean
%     .b    1-by-1 biases for the GP mean
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%   iK      n-by-n-by-E, inverse covariance matrix
%   beta    n-by-E, iK*(targets - mean function of inputs)
%
% x         n-by-D or n-by-D-by-E, training inputs or FITC inducing inputs 
% m         D-by-1, mean of the test distribution
% s         D-by-D, covariance matrix of the test distribution
%
% M         E-by-1, mean of pred. distribution 
% S         E-by-E, covariance of the pred. distribution             
% V         D-by-E, inv(s) times covariance between input and output 
%
% Copyright (C) 2008-2012 by Carl Edward Rasmussen, Marc Deisenroth,
% Andrew McHutchon, & Joe Hall 2014-03-10

x = gpmodel.inputs;
[n, D, pE] = size(x); E = size(gpmodel.beta,2);
h = gpmodel.hyp; iK = gpmodel.iK; beta = gpmodel.beta; 
p = h; [p.beta] = deal(zeros(n,1)); Np = length(unwrap(p)); idx = rewrap(p,1:Np);

% Initialisations
M1 = zeros(E,1); M = M1; S = zeros(E); C1 = zeros(D,E); k = zeros(n,E); 
dM1dm = zeros(E,D); dMdm = zeros(E,D); dSdm = zeros(E,E,D); dkdl = zeros(n,D,E); 
dMds = zeros(E,D,D); dSds = zeros(E,E,D,D); dCds = zeros(D,E,D,D); T = zeros(D);
dMdp = zeros(E,Np); dSdp = zeros(E,E,Np); dCdp = zeros(D,E,Np);

inp = bsxfun(@minus,x,m');                % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l));                  % D-by-D, L^-1/2
  in = inp(:,:,min(i,pE))*il;               % n-by-D, (X - m)*L^-1/2
  B = il*s*il+eye(D);               % D-by-D, B = (L^-1/2 * S * L^-1/2 + I)
  liBl = il/B*il;                           % D-by-D, liBl = (L + S)^-1
  
  t = in/B;                    % n-by-D, t = (X-m)*[L^-1/2 * S + L^-1/2]^-1
  l = exp(-sum(in.*t,2)/2);          % n-by-1, in.*t = (X-m) (S+L)^-1 (X-m)
  lb = l.*beta(:,i);    % n-by-1, lb = exp(-0.5*(X-m) (S+L)^-1 (X-m)).*beta
  tL = t*il; tlb = bsxfun(@times,tL,lb);      % n-by-D, tL = (X-m)*(S+L)^-1
  c = exp(2*h(i).s)/sqrt(det(B));             % c = sf2/sqrt(det(S*iL + I))
  
  M1(i) = sum(lb)*c; M(i) = M1(i) + h(i).m'*m + h(i).b;    % predicted mean
  C1(:,i) = tL'*lb*c;
  
  % dM1dm, dMdm, dMds, and dCds
  dM1dm(i,:) = tL'*lb*c;
  dMdm(i,:) = dM1dm(i,:) + h(i).m';
  dMds(i,:,:) = c*tL'*tlb/2 - liBl*M1(i)/2;
  for d = 1:D
    dCds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - liBl*C1(d,i)/2 ...
                                                    - C1(:,i)*liBl(d,:);
  end
  
  % Pre-computations for calculating S
  in2 = in.^2; k(:,i) = 2*h(i).s-sum(in2,2)/2;
   
  % Derivatives w.r.t. the parameters p
  dMdp(i,idx(i).beta) = c*l';
  dCdp(:,i,idx(i).beta) = bsxfun(@times,tL,c*l)';
  
  dMdp(i,idx(i).s) = 2*M1(i);
  dCdp(:,i,idx(i).s) = 2*C1(:,i);
  
  dMdp(i,idx(i).m) = m; dMdp(i,idx(i).b) = 1;
  
  L = diag(exp(2*h(i).l));
  lbdl = bsxfun(@times,tL*L.*tL,lb);                    % n-by-D
  cdl = c*(1 - diag(liBl*L));                           % D-by-1
  dMdp(i,idx(i).l) = sum(lbdl,1)*c + sum(lb)*cdl';      % E-by-D
  dkdl(:,:,i) = in2;                                    % n-by-D-by-E
  tLdl = -2*bsxfun(@times,permute(tL*L,[1,3,2]),permute(liBl,[3,2,1])); % n-by-D-by-D
  dCdp(:,i,idx(i).l) = reshape(lb'*tLdl(:,:),D,D)*c + tL'*lbdl*c + tL'*lb*cdl';
end
dCdm = 2*permute(dMds,[2 1 3]);

iL = exp(-2*[h.l]); inpil = bsxfun(@times,inp,permute(iL,[3,1,2])); % N-by-D-by-E
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); 
    t = 1/sqrt(det(R)); ij = inpil(:,:,j); iRs = R\s;
    L = exp(bsxfun(@plus,k(:,i),k(:,j)') + maha(ii,-ij,iRs/2));
    bLb = beta(:,i)'*L*beta(:,j);
    
    if i==j
      iKL = iK(:,:,i).*L; s1iKL = sum(iKL,1); ssiKL = sum(s1iKL);
      S1 = t*(bLb - ssiKL);
      
      zi = ii/R; s2iKL = sum(iKL,2);
      bibLi = L'*beta(:,i).*beta(:,i); cbLi = L'*bsxfun(@times, beta(:,i), zi);
      r = (bibLi'*zi*2 - (s2iKL' + s1iKL)*zi)*t;
      for d = 1:D
        T(d,1:d) = 2*(zi(:,1:d)'*(zi(:,d).*bibLi) + ...
            cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) ...
                                                   - zi(:,1:d)'*(iKL*zi(:,d)));
        T(1:d,d) = T(d,1:d)';
      end
      
    else
      S1 = t*bLb;
      
      zi = ii/R; zj = ij/R;
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
    S(i,j) = S1;
    S(j,i) = S(i,j);
    
    % dSdm & dSds derivatives
    dSdm(i,j,:) = r - M1(i)*dM1dm(j,:)-M1(j)*dM1dm(i,:);
    dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S1*diag(iL(:,i)+iL(:,j))/R)/2;
    T = T - reshape(M1(i)*dMds(j,:,:) + M1(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T;
    dSds(j,i,:,:) = permute(dSds(i,j,:,:),[1,2,4,3]);
    
    % dSdp derivatives
    dSdp(i,j,idx(i).beta) = L*beta(:,j)*t; 
    dSdp(i,j,idx(j).beta) = permute(dSdp(i,j,idx(j).beta),[4,3,1,2]) + beta(:,i)'*L*t;   
    
    dSdp(i,j,idx(i).s) = 2*S1;
    dSdp(i,j,idx(j).s) = dSdp(i,j,idx(j).s) + 2*S1;
    
    tdli = diag(iRs).*iL(:,i)*t; tdlj = diag(iRs).*iL(:,j)*t; 
    
    iiij = bsxfun(@plus,permute(ii,[1,3,2]),permute(ij,[3,1,2]));
    ijiRs = reshape(reshape(iiij,n^2,D)*iRs,n,n,D);         % n-by-n-by-D
    mahadli = -2*bsxfun(@times,permute(ii,[1,3,2]),ijiRs) ...
        + bsxfun(@times,ijiRs.^2,permute(iL(:,i),[2,3,1])); % n-by-n-by-D
    mahadlj = -2*bsxfun(@times,permute(ij,[3,1,2]),ijiRs) ...
        + bsxfun(@times,ijiRs.^2,permute(iL(:,j),[2,3,1]));
    
    dLdli = bsxfun(@times,bsxfun(@plus,permute(dkdl(:,:,i),[1,3,2]),mahadli),L); % n-by-n-by-D
    dLdlj = bsxfun(@times,bsxfun(@plus,permute(dkdl(:,:,j),[3,1,2]),mahadlj),L); % n-by-n-by-D
    
    dS1dli = reshape(beta(:,i)'*dLdli(:,:),n,D)'*beta(:,j)*t + bLb*tdli;
    dS1dlj = reshape(beta(:,i)'*dLdlj(:,:),n,D)'*beta(:,j)*t + bLb*tdlj;
    dSdp(i,j,idx(i).l) = dS1dli; 
    dSdp(i,j,idx(j).l) = squeeze(dSdp(i,j,idx(j).l)) + dS1dlj;
    
    if i==j; 
        dSdiK = -t*L; iKdn = -2*exp(2*h(i).n)*iK(:,:,i)*iK(:,:,i);
        dSdp(i,i,idx(i).n) = dSdiK(:)'*iKdn(:);
        iKdlsf = -2*iK(:,:,i)*gpmodel.Kclean(:,:,i)*iK(:,:,i);
        dSdp(i,i,idx(i).s) = dSdp(i,i,idx(i).s) + dSdiK(:)'*iKdlsf(:) + 2*exp(2*h(i).s);
        dSdp(i,i,idx(i).l) = squeeze(dSdp(i,i,idx(i).l))' - ...
                t*reshape(iK(:,:,i),1,n^2)*reshape(dLdli+dLdlj,n^2,D) ...
                    - L(:)'*gpmodel.iKdl(:,:,i)*t - ssiKL*(tdli + tdlj)';
    end
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);            % add signal variance to diagonal
end

S = S - M1*M1' + [h.m]'*s*[h.m] + C1'*s*[h.m] + [h.m]'*s*C1;

dMds = reshape(dMds,[E D*D]); 

% Finish dSdm: deriv{ C1'*s*[h.m] + [h.m]'*s*C1 }
msCdm = reshape([h.m]'*s*reshape(dCdm,D,E*D),E,E,D);  % E-by-E-by-D
dSdm = dSdm + msCdm + permute(msCdm,[2,1,3]);
dSdm = reshape(dSdm,[E*E D]);               % finally vectorise derivative

% Finish dSds: deriv{ [h.m]'*s*[h.m] + C1'*s*[h.m]+[h.m]'*s*C1 }
msCds = reshape([h.m]'*s*reshape(dCds,D,E*D^2),E,E,D,D);  % E-by-E-by-D-by-D
dSds = dSds + bsxfun(@times,permute([h.m],[2,3,1,4]),permute([h.m],[3,2,4,1]))...
    + bsxfun(@times,permute(C1,[2,3,1,4]),permute([h.m],[3,2,4,1])) ...
    + bsxfun(@times,permute([h.m],[2,3,1,4]),permute(C1,[3,2,4,1])) ...
    + msCds + permute(msCds,[2,1,3,4]);
dSds = reshape(dSds,[E*E D*D]);                         % finally vectorise

% Finish dSdp: deriv{ - M1*M1' + [h.m]'*s*[h.m] + C1'*s*[h.m]+[h.m]'*s*C1 }
i = repmat(triu(true(E,E)),[1,1,Np]);        % first fill in symmetric half
dSdpt = permute(dSdp,[2,1,3]); dSdp(i) = dSdpt(i);
dM2dp = bsxfun(@times,M1,permute(dMdp,[3,1,2]));        % E-by-E-by-Np
msCdp = reshape([h.m]'*s*reshape(dCdp,D,E*Np),E,E,Np);  % E-by-E-by-Np
dSdp = bsxfun(@minus,dSdp,dM2dp + permute(dM2dp,[2,1,3])) ...
                                    + msCdp + permute(msCdp,[2,1,3]);
ms = [h.m]'*s; Cs = C1'*s; t = zeros(E,E,D,E); 
for i=1:D; for j=1:E; t(:,j,i,j) = ms(:,i) + Cs(:,i); end; end
dSdp(:,:,[idx.m]) = reshape(t+permute(t,[2,1,3,4]),[E,E,E*D]);
dSdp(:,:,[idx.b]) = 0; % set back to 0, we used dMdp earlier not dM1dp
dSdp = reshape(dSdp,E^2,Np);                % finally vectorise derivative
