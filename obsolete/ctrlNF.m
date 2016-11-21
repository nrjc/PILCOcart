function [uM, uS, uC, s, duMds, duSds, duCds, dsds, ...
  duMdp, duSdp, duCdp, dsdp] = ctrlNF(s, ctrl, plant, dsds, dsdp)

% Controller with No Filter. The state is given either as a point m or as a
% distribution N(s.m,s.s). First augment with trignometric functions if
% necessary, then call the policy, and finally (optionally) call the actuate
% function. There is no filter, so no updates to any state structure filter
% fields are required.
%
% s           .       state structure
%   m         D x 1   state mean
%   s         D x D   (optional) (noisy) state variance (default zero)
% ctrl        .       controller structure
%   U                 number of control outputs
%   on        D x 1   log observation noise
%   policy    .       policy structure
%     fcn     @       policy function
%   actuate   @       (optional) call this function with the calculated action
% plant               plant structure
%   angi              angular variabels indices
%   poli              policy input indices
% uM          U x 1   control signal mean
% uS          U x U   control signal variance
% uC          D x U   inv(s) times input output covariance
% duMds       U x S   derivatives of outputs wrt input state struct
% duSds     U*U x S
% duCds     U*D x S
% dsds        S x S   ouput state derivative wrt input state
% duMdp       U x P   P is the total number of parameters is the policy
% duSdp     U*U x P
% duCdp     U*D x P
% dsdp        S x P   ouput state derivative wrt policy parameters
%
% Copyright (C) 2014 by Carl Edward Rasmussen and Rowan McAllister 2014-09-18

% no filter to reset
if strcmp(ctrl, 'ResetFilter');
  uM = nan; uS = nan; uC = nan; return;
end

angi = plant.angi; poli = plant.poli; U = ctrl.U; A = length(angi);
derivativesRequested = nargout > 4;
ns = plant.ns; is = plant.is;
D = length(s.m); DD = D*D;
D1 = D + 2*A;
i=1:D;
if isfield(s,'s'), ss = s.s; else ss = zeros(D); end
M = zeros(D1,1); M(i) = s.m; S = zeros(D1); S(i,i) = ss;
if derivativesRequested
  idx = @(i,j,I) bsxfun(@plus, I*(i'-1), j);
  Mds = zeros(D1,ns); Mds(i,is.m) = eye(D);
  Sds = zeros(D1*D1,ns); Sds(:,is.s) = kron(Mds(:,is.m),Mds(:,is.m));
end

% augment with trig functions
i = 1:D; k = D+1:D1;
if ~derivativesRequested
  [M(k), S(k,k), c] = gTrig(M(i), S(i,i), angi);
else
  kk = idx(k,k,D1); ik = idx(i,k,D1); ki = idx(k,i,D1);
  [M(k), S(k,k), c, Mds(k,is.m), Sds(kk,is.m), Cdm, ...
    Mds(k,is.s), Sds(kk,is.s), Cds] = gTrig(M(i), S(i,i), angi);
  qdm = reshape(S(i,i)*reshape(Cdm,D,[]),[],D);
  Sds(ik,is.m) = qdm; Sds(ki',is.m) = qdm;
  qds = reshape(S(i,i)*reshape(Cds,D,[]),[],DD) + kron(c',eye(D));
  Sds(ik,is.s) = qds; Sds(ki',is.s) = qds;
end
q = S(i,i)*c; S(i,k) = q; S(k,i) = q';

% compute control signal
if ~derivativesRequested
  [uM, uS, uC] = ctrl.policy.fcn(ctrl.policy, M(poli), S(poli,poli));
else
  [uM, uS, uC, mdm, sdm, cdm, mds, sds, cds, duMdp, duSdp, duCdp] = ...
    ctrl.policy.fcn(ctrl.policy, M(poli), S(poli,poli));
end
if isfield(ctrl, 'actuate'), ctrl.actuate(uM); end         % actuate controller

eDc = [eye(D) c]; eDcp = eDc(:,poli); eDc = reshape(eDcp,D,[]);
if derivativesRequested
  j = idx(poli,poli,D1); l = length(poli); k = l+(1-2*A:0);
  duMds = mdm*Mds(poli,:) + mds*Sds(j,:);
  duSds = sdm*Mds(poli,:) + sds*Sds(j,:);
  duCds = nan(U*D,ns);
  duCds(:,is.m) = reshape(reshape(Cdm',DD,2*A)*uC(k,:),D,D*U)' + ...
    reshape(eDc*reshape(cdm*Mds(poli,is.m)+cds*Sds(j,is.m),l,[]),[],D);
  duCds(:,is.s) = reshape(reshape(Cds',D*DD,2*A)*uC(k,:),DD,D*U)' + ...
    reshape(eDc*reshape(cdm*Mds(poli,is.s)+cds*Sds(j,is.s),l,[]),[],DD);
  duCdp = reshape(eDc*reshape(duCdp,l,[]),D*U,[]);
end
uC = eDcp*uC;                                 % correct input output covariance