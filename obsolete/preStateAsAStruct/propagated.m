function [state, Mo, So, dMdm, dSdm, dMds, dSds, dMdp, dSdp] = ...
                                        propagated(m, s, plant, dynmodel, ctrl)

% Propagate the state distribution one time step forward with derivatives
%
% dMdm    output mean wrt input mean                                [ E  x  D ]
% dMds    output mean wrt input covariance matrix                   [ E  x D*D]
% dSdm    output covariance matrix wrt input mean                   [E*E x  D ]
% dSds    output cov wrt input cov                                  [E*E x D*D]
% dMdp    output mean wrt policy parameters                         [ E  x  P ]
% dSdp    output covariance matrix wrt policy parameters            [E*E x  P ]
%
% where P is the number of policy parameters.
%
% Copyright (C) 2008-2014 by Marc Deisenroth, Carl Edward Rasmussen, Henrik
% Ohlsson, Andrew McHutchon and Joe Hall, 2014-04-25

if nargout < 4                                   % just predict, no derivatives
  [state, Mo, So] = propagate(m, s, plant, dynmodel, ctrl);
  return
end

angi = plant.angi; dyni = plant.dyni;

D0 = length(m);                                        % size of the input mean
D1 = D0 + 2*length(angi);          % length after mapping all angles to sin/cos
D2 = D1 + ctrl.U;                          % length after computing ctrl signal
D3 = D2 + D0;                                         % length after predicting
M = zeros(D3,1); M(1:D0) = m; S = zeros(D3); S(1:D0,1:D0) = s;   % init M and S

M0dm = [eye(D0); zeros(D3-D0,D0)]; S0dm = zeros(D3*D3,D0);
M0ds = zeros(D3,D0*D0); S0ds = kron(M0dm,M0dm);
X = reshape(1:D3*D3,[D3 D3]); XT = X'; S0ds = (S0ds + S0ds(XT(:),:))/2;
X = reshape(1:D0*D0,[D0 D0]); XT = X'; S0ds = (S0ds + S0ds(:,XT(:)))/2;

% 1) Augment state distribution with trigonometric functions ------------------
i = 1:D0; j = 1:D0; k = D0+1:D1;
[M(k), S(k,k), C, mdm, sdm, Cdm, mds, sds, Cds] = gTrig(M(i), S(i,i), angi);
[S, Mdm, Mds, Sdm, Sds] = ...
  fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,M0dm,S0dm,M0ds,S0ds,[ ],[ ],[ ],i,j,k,D3);

% 2) Compute distribution of the control signal -------------------------------
i = 1:D0; j = 1:D1; k = D1+1:D2; kk = sub2ind2(D3,k,k); kj = sub2ind2(D3,k,j); 
ji = sub2ind2(D3,j,i); jk = sub2ind2(D3,j,k); ii = sub2ind2(D3,i,i);
[state, M(k), S(k,k), C, mdm, sdm, Cdm, mds, sds, Cds, Mdp, Sdp, Cdp] = ...
            ctrl.fcn(ctrl, plant, M(i), S(i,i)+diag(exp(2*[dynmodel.hyp.on])));
[S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = ...
      fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,Mdp,Sdp,Cdp,i,j,k,D3);

% 3) Compute distribution of the change in state ------------------------------
ii = [dyni D1+1:D2]; j = 1:D2;
if isfield(dynmodel,'sub'), Nf = length(dynmodel.sub); else Nf = 1; end
for n=1:Nf                               % potentially multiple dynamics models
  [dyn, i, k] = sliceModel(dynmodel,n,ii,D1,D2,D3); j = setdiff(j,k);
  
  [M(k), S(k,k), C, mdm, sdm, Cdm, mds, sds, Cds] = dyn.fcn(dyn, M(i), S(i,i));
  
  S(k,k) = S(k,k) + diag(exp(2*[dyn.hyp.pn]));              % add process noise
  
  [S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = ...
      fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,Mdp,Sdp,[ ],i,j,k,D3);
  
  j = [j k];                                   % update 'previous' state vector
end

% 4) Select distribution of the next state -----------------------------------
i = D2+1:D3; ii = sub2ind2(D3,i,i); 
Mo = M(i); So = S(i,i); So = (So+So')/2;

dMdm = Mdm(i,:); dMds = Mds(i,:); dMdp = Mdp(i,:);
dSdm = Sdm(ii,:); dSds = Sds(ii,:); dSdp = Sdp(ii,:);

X = reshape(1:D0*D0,[D0 D0]); XT = X';  XT = XT(:);             % symmetrise dS
dSdm = (dSdm + dSdm(XT,:))/2; dMds = (dMds + dMds(:,XT))/2;
dSds = (dSds + dSds(XT,:))/2; dSds = (dSds + dSds(:,XT))/2;
dSdp = (dSdp + dSdp(XT,:))/2;


% A1) Separate multiple dynamics models ---------------------------------------
function [dyn, i, k] = sliceModel(dynmodel,n,ii,D1,D2,D3) % separate sub-dynamics
if isfield(dynmodel,'sub')
  dyn = dynmodel.sub{n}; do = dyn.dyno; D = length(ii)+D1-D2;
  if isfield(dyn,'dyni'), di=dyn.dyni; else di=[]; end
  if isfield(dyn,'dynu'), du=dyn.dynu; else du=[]; end
  if isfield(dyn,'dynj'), dj=dyn.dynj; else dj=[]; end
  i = [ii(di) D1+du D2+dj]; k = D2+do;
  dyn.inputs = [dynmodel.inputs(:,[di D+du]) dynmodel.target(:,dj)];   % inputs
  dyn.target = dynmodel.target(:,do);                                 % targets
else
    dyn = dynmodel; k = D2+1:D3; i = ii;
end

% A2) Apply chain rule and fill out cross covariance terms --------------------
function [S, Mdm, Mds, Sdm, Sds, Mdp, Sdp] = ...
       fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,Mdm,Sdm,Mds,Sds,Mdp,Sdp,dCdp,i,j,k,D)

if isempty(k), return; end
   
aa = length(k); bb = numel(S(k,k)); cc = numel(C);    % reshape new derivatives
sdm=reshape(sdm,bb,[]); Cdm=reshape(Cdm,cc,[]); mds=reshape(mds,aa,[]);
sds=reshape(sds,bb,[]); Cds=reshape(Cds,cc,[]);
    
 
ii = sub2ind2(D,i,i); kk = sub2ind2(D,k,k);  % vectorised indices
ji = sub2ind2(D,j,i); jk = sub2ind2(D,j,k);
kj = kron(k,ones(1,length(j))) + kron(ones(1,length(k)),(j-1)*D);

Mdm(k,:)  = mdm*Mdm(i,:) + mds*Sdm(ii,:);                           % chainrule
Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:);
Sdm(kk,:) = sdm*Mdm(i,:) + sds*Sdm(ii,:);
Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:);
dCdm      = Cdm*Mdm(i,:) + Cds*Sdm(ii,:);
dCds      = Cdm*Mds(i,:) + Cds*Sds(ii,:);
if isempty(dCdp) && nargout > 5
  Mdp(k,:)  = mdm*Mdp(i,:) + mds*Sdp(ii,:);
  Sdp(kk,:) = sdm*Mdp(i,:) + sds*Sdp(ii,:);
  dCdp      = Cdm*Mdp(i,:) + Cds*Sdp(ii,:);
elseif nargout > 5
  mdp = zeros(D,size(Mdp,2)); sdp = zeros(D*D,size(Mdp,2));
  mdp(k,:)  = reshape(Mdp,aa,[]); Mdp = mdp;
  sdp(kk,:) = reshape(Sdp,bb,[]); Sdp = sdp;
  Cdp       = reshape(dCdp,cc,[]); dCdp = Cdp;
end

q = S(j,i)*C; S(j,k) = q; S(k,j) = q';                           % off-diagonal
SS = kron(eye(length(k)),S(j,i)); CC = kron(C',eye(length(j)));
Sdm(jk,:) = SS*dCdm + CC*Sdm(ji,:); Sdm(kj,:) = Sdm(jk,:);
Sds(jk,:) = SS*dCds + CC*Sds(ji,:); Sds(kj,:) = Sds(jk,:);
if nargout > 5; Sdp(jk,:) = SS*dCdp + CC*Sdp(ji,:); Sdp(kj,:) = Sdp(jk,:); end

function idx = sub2ind2(D,i,j)
% D = #rows, i = row subscript, j = column subscript
i = i(:); j = j(:)';
idx =  reshape(bsxfun(@plus,D*(j-1),i),1,[]);
