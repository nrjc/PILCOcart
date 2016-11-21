function [state, M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, ...
                                              dCdp] = ctrlNF(ctrl, plant, m, s)

% Controller with No Filter. The state is given either as a point m or as a
% distribution N(m,s). First augment with trignometric functions if necessary,
% then call the policy, and finally (optionally) call the actuate function. The
% controller is state free, so no updates to the ctrl structure are required.
%
% ctrl                 controller structure
%   policy             policy structure
%     fcn     @        policy function
%   actuate   @        (optional) call this function with the calculated action
%   state     empty    this controller has an empty state
% plant                plant structure
%   angi               angular variabels indices
%   poli               policy input indices
% m           D x 1    state mean
% s           D x D    (optional) state variance (default zero)
% state       empty    this controller has no state, and thus returns nothing     
% M           U x 1    action mean
% S           U x U    action variance
% C           D x U    inv(s+on) times input output covariance 
% dMdm        U x D    derivatives of outputs wrt inputs (as matrices)
% dSdm      UxU x D
% dCdm      DxU x D
% dMds        U x DxD
% dSds      UxU x DxD
% dCds      DxU x DxD
% dMdp        U x P    P is the total number of parameters is the policy
% dSdp      UxU x P
% dCdp      DxU x P
%
% Copyright (C) 2014 by Carl Edward Rasmussen, 2014-04-07

angi = plant.angi; poli = plant.poli; U = ctrl.U; D = length(m); DD = D*D;
A = length(angi); D1 = D+2*A;
if nargin == 3, s = zeros(D,D); end
idx = @(i,j,I) bsxfun(@plus, I*(i'-1), j);

i = 1:D; k = D+1:D1; j = idx(k,k,D1);             % augment with trig functions
mdm = zeros(D1,D); mdm = eye(D); sdm = zeros(D1*D1,D); mds = zeros(D1,DD);
sds = zeros(D1*D1,DD); sds(idx(i,i,D1)',1:DD) = eye(DD);

[m(k), s(k,k), c, mdm(k,i), sdm(j,i), cdm, mds(k,1:DD), sds(j,1:DD), cds] = ...
                                                             gTrig(m, s, angi);
q = s(i,i)*c; s(i,k) = q; s(k,i) = q';
qdm = reshape(s(i,i)*reshape(cdm,D,[]),[],D);
sdm(idx(i,k,D1),i) = qdm; sdm(idx(k,i,D1)',i) = qdm; 
qds = reshape(s(i,i)*reshape(cds,D,[]),[],DD) + kron(c',eye(D));
sds(idx(i,k,D1),1:DD) = qds; sds(idx(k,i,D1)',1:DD) = qds; 

[M, S, C, Mdm, Sdm, Cdm, Mds, Sds, Cds, dMdp, dSdp, dCdp] = ...   % call policy
                           ctrl.policy.fcn(ctrl.policy, m(poli), s(poli,poli));

j = idx(poli,poli,D1); l = length(poli); k = l+(1-2*A:0);
eDc = [eye(D) c]; eDcp = eDc(:,poli); eDc = reshape(eDcp,D,[]);
dMdm = Mdm*mdm(poli,i) + Mds*sdm(j,i);
dSdm = Sdm*mdm(poli,i) + Sds*sdm(j,i);
dCdm = reshape(reshape(cdm',DD,2*A)*C(k,:),D,D*U)' + ...
                  reshape(eDc*reshape(Cdm*mdm(poli,i)+Cds*sdm(j,i),l,[]),[],D);
dMds = Mdm*mds(poli,1:DD) + Mds*sds(j,1:DD);
dSds = Sdm*mds(poli,1:DD) + Sds*sds(j,1:DD);
dCds = reshape(reshape(cds',D*DD,2*A)*C(k,:),DD,D*U)' + ...
                 reshape(eDc*reshape(Cdm*mds(poli,:)+Cds*sds(j,:),l,[]),[],DD);
dCdp = reshape(eDc*reshape(dCdp,l,[]),D*U,[]);
C = eDcp*C;                                   % correct input output covariance

if isfield(ctrl, 'actuate'), ctrl.actuate(M); end     % call actuation function

state = [];                                              % return (empty) state
