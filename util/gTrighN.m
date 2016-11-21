function [M, S, CC, V, dMdm, dSdm, dCdm, dVdm, dMds, dSds, dCds, dVds, ...
  dMdv, dSdv, dCdv, dVdv] = gTrighN(m, s, v, i, e)

% gTrigN is similar to gTrig, except the outputs are complete matrices, not
% just the incremental parts supplied by gTrig.
%
% m      D x 1    mean-of-mean Gaussian vector
% s      D x D    mean-covariance matrix
% v      D x D    covariance matrix
% i               length vector of indices of elements to augment
% e               length optional scale vector (defaults to unity)
% M      E x 1
% S      E x E
% C      D x E
% dMdm   E x D
% dSdm  EE x D
% dCdm  DE x D
% dVdm  EE x D
% dMds   E x DD
% dSds  EE x DD
% dCds  DE x DD
% dVds  EE x DD
% dMdv   E x DD
% dSdv  EE x DD
% dCdv  DE x DD
% dVdv  EE x DD
%
% Rowan McAllister 2014-12-09

D = length(m); E = D + 2*length(i); n = 1:D; a = D+1:E;
if nargin == 4, e = ones(length(i),1); end

% non-derivatives case
M = m; S = s; CC = eye(D); V = v;
if nargout <= 4
  if isempty(i); return; end
  [M(a), S(a,a), C, V(a,a)] = gTrigh(m, s, v, i, e);
  S(n,a) = s*C; S(a,n) = S(n,a)';
  V(n,a) = v*C; V(a,n) = V(n,a)';
  CC = [CC, C];
  return
end

% derivatives-case
DD = D*D; EE = E*E; DE = D*E;
aa = sub2ind2(E,a,a); na = sub2ind2(E,n,a); an = sub2ind2(E,a,n);
nn = sub2ind2(E,n,n); NA = sub2ind2(D,n,a);

dMdm = eye(E,D);
dSdm = zeros(EE,D);
dCdm = zeros(DE,D);
dVdm = zeros(EE,D);
dMds = zeros(E,DD);
dSds = zeros(EE,DD); dSds(nn,:) = eye(DD);
dCds = zeros(DE,DD);
dVds = zeros(EE,DD);
dMdv = zeros(E,DD);
dSdv = zeros(EE,DD);
dCdv = zeros(DE,DD);
dVdv = zeros(EE,DD); dVdv(nn,:) = eye(DD);
if isempty(i); return; end

[M(a), S(a,a), C, V(a,a), ...
  dMdm(a,:), dSdm(aa,:), Cdm, dVdm(aa,:), ...
  dMds(a,:), dSds(aa,:), Cds, dVds(aa,:), ...
  dMdv(a,:), dSdv(aa,:), Cdv, dVdv(aa,:)] = gTrigh(m, s, v, i, e);
S(n,a) = s*C; S(a,n) = S(n,a)';
V(n,a) = v*C; V(a,n) = V(n,a)';
CC = [CC, C];

dCdm(NA,:) = Cdm;
Ctdm = transposed(Cdm,D);
dSdm(na,:) = prodd(s,Cdm);
dSdm(an,:) = prodd([],Ctdm,s);
dVdm(na,:) = prodd(v,Cdm);
dVdm(an,:) = prodd([],Ctdm,v);

dCds(NA,:) = Cds;
Ctds = transposed(Cds,D);
dSds(na,:) = prodd([],'eye',C) + prodd(s,Cds);
dSds(an,:) = prodd(C','eye') + prodd([],Ctds,s);
dSds = symmetrised(dSds,1);
dVds(na,:) = prodd(v,Cds);
dVds(an,:) = prodd([],Ctds,v);

dCdv(NA,:) = Cdv;
Ctdv = transposed(Cdv,D);
dSdv(na,:) = prodd(s,Cdv);
dSdv(an,:) = prodd([],Ctdv,s);
dVdv(na,:) = prodd([],'eye',C) + prodd(v,Cdv);
dVdv(an,:) = prodd(C','eye') + prodd([],Ctdv,v);
dVdv = symmetrised(dVdv,1);