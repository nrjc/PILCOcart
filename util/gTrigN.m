function [M, S, CC, dMdm, dSdm, dCdm, dMds, dSds, dCds] = gTrigN(m, s, i, e)

% gTrigN is similar to gTrig, except the outputs are complete matrices, not
% just the incremental parts supplied by gTrig.
%
% m      D x 1
% s      D x D
% i      A x 1
% m      E x 1
% s      E x E
% C      D x E
% dMdm   E x D
% dSdm  EE x D
% dCdm  DE x D
% dMds   E x DD
% dSds  EE x DD
% dCds  DE x DD
%
% Carl Edward Rasmussen and Rowan McAllister 2014-12-09

D = length(m); E = D + 2*length(i); n = 1:D; a = D+1:E;
if nargin == 3, e = ones(length(i),1); end

% non-derivatives case
M = m; S = s; CC = eye(D);
if nargout <= 3
  if isempty(i); return; end
  [M(a), S(a,a), C] = gTrig(m, s, i, e);
  q = s*C; S(n,a) = q; S(a,n) = q';
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
dMds = zeros(E,DD);
dSds = zeros(EE,DD); dSds(nn,:) = eye(DD);
dCds = zeros(DE,DD);
if isempty(i); return; end

[M(a), S(a,a), C, dMdm(a,:), dSdm(aa,:), Cdm, dMds(a,:), dSds(aa,:), Cds] = ...
  gTrig(m, s, i, e);
q = s*C; S(n,a) = q; S(a,n) = q';
CC = [CC, C];

dSdm(na,:) = prodd(s,Cdm);
dSdm(an,:) = prodd([],transposed(Cdm,D),s);
dCdm(NA,:) = Cdm;
dSds(na,:) = prodd([],'eye',C) + prodd(s,Cds);
dSds(an,:) = prodd(C','eye') + prodd([],transposed(Cds,D),s);
dCds(NA,:) = Cds;