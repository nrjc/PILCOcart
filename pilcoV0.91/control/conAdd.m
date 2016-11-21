function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] = ...
                                                            conAdd(policy, m, s)
% Utility function to add a number of controllers together, each of which
% can be using a different control function and operating on a different
% part of the state.
%
% Inputs
%   policy          policy struct
%      .fcn         @conAdd - called to arrive here
%      .sub{n}      cell array of sub controllers to add together
%         .fcn      handle to sub function
%         .poli     indices of variables to be passed to controller
%         .< >      all fields in sub will be passed onto the sub function
%      .p{n}        parameters for the n'th controller
%      .maxU        the maximum control values allowed (used to find # controls)
%
%   m               input distribution mean
%   s               input distribution variance
% 
% Andrew McHutchon, September 2012

% Dimensions and Initialisations                                                       
Npol = length(policy.sub); D = length(m); E = length(policy.maxU);
Np = length(unwrap(policy.p)); pidx = 0;
M = zeros(E,1); S = zeros(E); C = zeros(D,E);
dMdm = zeros(E,D); dSdm = zeros(E^2,D); dCdm = zeros(D*E,D);
dMds = zeros(E,D^2); dSds = zeros(E^2,D^2); dCds = zeros(D*E,D^2);
dMdp = zeros(E,Np); dSdp = zeros(E^2,Np); dCdp = zeros(D*E,Np);

for n=1:Npol                                 % Loop over each of the sub-polices
    pol = policy.sub{n}; pol.p = policy.p{n}; pol.maxU = policy.maxU;    % slice
    i = pol.poli; pidx = pidx(end) + (1:length(unwrap(pol.p))); % the sub-policy
    
    % Call the sub policy
    if nargout > 3
        [Mi, Si, Ci, Mdm, Sdm, Cdm, Mds, Sds, Cds, Mdp, Sdp, Cdp] = ...
                                                  pol.fcn(pol, m(i), s(i,i));
    else
        [Mi, Si, Ci] = pol.fcn(pol, m(i), s(i,i));
    end
    
    M = M + Mi;
    S = S + Si + Ci'*s(i,:)*C + C'*s(:,i)*Ci; % V(a+b) = V(a)+V(b)+C(a,b)+C(b,a)
    
    if nargout > 3
        % Sort out all the derivatives
        ii = sub2ind2(D,i,i); ij = sub2ind2(D,i,1:D); ji = sub2ind2(D,1:D,i); ik = sub2ind2(D,i,1:E);
        dMdm(:,i) = dMdm(:,i) + Mdm; dMds(:,ii) = dMds(:,ii) + Mds; 
        
        kICis = kron(eye(E),Ci'*s(i,:)); kCsI = kron(C'*s(:,i),eye(E));
        kICs = kron(eye(E),C'*s(:,i)); kCisI = kron(Ci'*s(i,:),eye(E)); 
        Ti = Tvec(length(i),E); T = Tvec(D,E);
        kCis = kCisI*T + kICis; kCs = kCsI*Ti + kICs;
        
        dSdm(:,i) = dSdm(:,i) + Sdm + kCs*Cdm; dSdm = dSdm + kCis*dCdm;
        dSds(:,ii) = dSds(:,ii) + Sds + kCs*Cds; dSds = dSds + kCis*dCds;
        dSds(:,ij) = dSds(:,ij) + kron(C,Ci)';
        dSds(:,ji) = dSds(:,ji)  + kron(Ci,C)'; 
        
        dMdp(:,pidx) = dMdp(:,pidx) + Mdp; 
        dSdp(:,pidx) = dSdp(:,pidx) + Sdp + kCs*Cdp; 
        dSdp = dSdp + kCis*dCdp;
        
    end
    
    % Input - Output covariance update
    C(i,:) = C(i,:) + Ci;                  % must be after S and its derivatives
    
    if nargout > 3
        dCdm(ik,i) = dCdm(ik,i) + Cdm;               % must be after dSdm & dSds
        dCds(ik,ii) = dCds(ik,ii) + Cds;
        dCdp(ik,pidx) = dCdp(ik,pidx) + Cdp;
    end
end

function idx = sub2ind2(D,i,j)
% D = #rows, i = row subscript, j = column subscript
i = i(:); j = j(:)';
idx =  reshape(bsxfun(@plus,D*(j-1),i),1,[]);

function Tmn = Tvec(m,n)
% Builds the orthogonal transpose vectorization matrix of an m by n matrix.
% Tmn = TvecMat(m,n)
%
% Tmn is an orthogonal permutation matrix called the "vectorized transpose
% matrix" of an m#n matrix. For example, if A is m#n then Tmn*vec(A) =
% vec(A.'), where vec(A) is the "vectorization of A" defined as the column
% vector formed by concatenating all the columns of A. vec(A) = A(:).
%
% Other uses are in the tensor product, where B(x)A = Tpm( A(x)B )Tnq where
% A is m#n and B is p#q and (x) denotes the tensor product.
%
%
% %%% ZCD Feb 2010 %%%
%

if nargin == 1; n = m; end
d = m*n;
Tmn = zeros(d,d);

i = 1:d;
rI = 1+m.*(i-1)-(m*n-1).*floor((i-1)./n);
I1s = sub2ind([d d],rI,1:d);
Tmn(I1s) = 1;
Tmn = Tmn';