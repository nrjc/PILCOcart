function [uM, uS, uC, s, duMds, duSds, duCds, dsds, ...
  duMdp, duSdp, duCdp, dsdp] = ctrlBF(s, ctrl, plant, dynmodel, dsds, dsdp)

% Controller with Bayes Filter. There are two states, the system state given
% by N(s.m,s.s), and the internal filter state N(N(s.zm,s.zs),s.v), also
% returned in "state". First call the policy, then augment with trignometric
% functions if necessary, and finally (optionally) call the actuate function.
% See ctrlBF.pdf for details.
%
% s           .       state structure
%   m         D x 1   state mean
%   s         D x D   (optional) (noisy) state variance (default zero)
%   zm        D x 1   filter mean-of-mean
%   zs        D x D   filter variance-of-mean
%   zc        D x D   covariance of state and filter mean
%   v         D x D   filter variance
% ctrl                controller structure
%   U                 number of control outputs
%   on        D x 1   log observation noise
%   dynmodel  .       dynamics model structure
%     fcn     @       dynamics function
%   policy    .       policy structure
%     fcn     @       policy function
%   actuate   @       (optional) call this function with the calculated action
% plant               plant structure
%   angi              angular variabels indices
%   poli              policy input indices
%   is        .       state structure whose members are members' indexes
% dynmodel            propagates's dynmodel
% uM          U x 1   control signal mean
% uS          U x U   control signal variance
% uC          D x U   inv(s.s) times input-state output-control covariance
% duMds       U x S   derivatives of outputs wrt input state struct
% duSds     U*U x S
% duCds     U*D x S
% dsds        S x S   ouput state derivative wrt input state
% duMdp       U x P   P is the total number of parameters is the policy
% duSdp     U*U x P
% duCdp     U*D x P
% dsdp        S x P   ouput state derivative wrt policy parameters
%
% See also <a href="ctrlBF.pdf">ctrlBF.pdf</a>, CTRLBFT.M.
% Copyright (C) 2014 by Carl Edward Rasmussen and Rowan McAllister 2014-09-27

if strcmp(ctrl, 'ResetFilter')        % reset the Bayes filter to a broad prior
  D = numel(s.m); uM = nan; uS = nan; uC = nan;
  s.('zm') = zeros(size(s.m)); s.('zc') = zeros(D);
  s.('zs') = 1e6*eye(D); s.('v') = 1e5*eye(D);
  return
end

% 0) Initialisations ----------------------------------------------------------
angi = plant.angi; poli = plant.poli; dyni = plant.dyni; A = length(angi);
U = ctrl.U; ns = plant.ns; is = plant.is; derivativesRequested = nargout > 4;
D = length(s.zm); D0 = D; DD = D*D;               % length of input filter mean
D1 = D0 + D;                                        % length after latent state
D2 = D1 + 2*A;                            % length after augmented latent state
D3 = D2 + D;                                 % length after updated filter mean
D4 = D3 + 2*A;                     % length after augmented updated filter mean
D5 = D4 + U;                                      % length after control signal
D6 = D5 + D0; L = D6;                      % length after predicted filter mean
prod_a_dx = @(a,dx) (reshape(a*reshape(dx,size(a,2),[]),[],size(dx,2)));
prod_dx_b = @(dx,b) (reshape(reshape(dx',[],size(b,1))*b,size(dx,2),[])');
prod_a_dx_b = @(a,dx,b) (prod_a_dx(a,prod_dx_b(dx,b)));
i = 1:D1; i0 = 1:D0; i1 = D0+1:D1;
M = zeros(L,1); M(i) = [s.zm ; s.m];
S = zeros(L); S(i,i) = [s.zs , s.zc' ; s.zc, s.s];
V = zeros(L); V(i0,i0) = s.v;
if derivativesRequested
  idx3 = @(D,i,j,k) (bsxfun(@plus, i(:), D*(j(:)'-1)) + D*D*(k-1));
  Mds = zeros(L,ns); Mds(i0,is.zm) = eye(D0); Mds(i1,is.m) = eye(D0);
  Sds = zeros(L*L,ns); Sds(idx3(L,i0,i0,is.zs)) = 1; Sds(idx3(L,i1,i1,is.s))=1;
  Sds(idx3(L,i1,i0,is.zc)) = 2; Vds = zeros(L*L,ns); Vds(idx3(L,i0,i0,is.v))=1;
  XS = [is.s, is.zs, is.v]; XST = [is.s', is.zs', is.v'];
  XT = reshape(1:L*L,[L L])'; XT = XT(:);
end

% 1) Bayes-filter update step -------------------------------------------------
i = 1:D1; i0 = 1:D0; i1 = D0+1:D1; k = D2+1:D3; kk = sub2ind2(L,k,k);
ki=sub2ind2(L,k,i); ki0 = sub2ind2(L,k,i0); ki1 = sub2ind2(L,k,i1);
n = diag(exp(2*ctrl.on)); v = V(i0,i0); I = eye(D);
w1 = n/(v+n); w2 = v/(v+n); w = [w1 w2];        % filter 'weights' for updating
M(k) = w*M(i);
SNw = S(i,i)*w';
S(k,k) = w*SNw;
%S(k,i) = w*(S(i,i)-blkdiag(0*n,n)); S(i,k) = S(k,i)';                          % TODO: verify deleteable.
S(k,i) = w*S(i,i); S(i,k) = S(k,i)';
V(k,k) = n/(v+n)*v;
if derivativesRequested
  Mds(k,[is.zm ; is.m]) = w;
  Mds(k,is.v) = kron((M(i1)-M(i0))'/(v+n),w1);
  kr11 = kron(w1,w1); kr2Iw = 2*kron(I,w);
  Sds(kk,is.zs) = kr11;
  Sds(kk,is.s) = kron(w2,w2);
  Sds(kk,is.zc) = 2*kron(w1,w2);         % '2's allocate for both off-diagonals
  Sds(kk,is.v) = 2*kron((SNw(i1,:)-SNw(i0,:))'/(v+n),w1);
  Sds(ki0,[is.zs; is.zc]) = kr2Iw;
  Sds(ki1,[is.zc';is.s]) = kr2Iw;
  %Sds(ki,is.v) = 2*kron((S(i1,i)-[0*n,n]-S(i0,i))'/(v+n),w1);                  % TODO: verify deleteable.
  Sds(ki,is.v) = 2*kron((S(i1,i)-S(i0,i))'/(v+n),w1);
  Vds(kk,is.v) = kr11;
  [Mds, Sds, Vds] = symmetrise(XS, XST, XT, Mds, Sds, Vds);
end

% 2) Augment latent state -----------------------------------------------------
i = D0+1:D1; j = [1:D1,D2+1:D3]; k = D1+1:D2;
if derivativesRequested
  [M(k), S(k,k), C, mdm, sdm, cdm, mds, sds, cds] = gTrig(M(i), S(i,i), angi);
  [S,~,Mds,Sds] = fillIn(i,j,k,L,S,C,[], ...
    mdm,sdm,[],cdm,mds,sds,[],cds,[],[],[],[], ...
    Mds,Sds,[],[],[],[],[]);
else
  [M(k), S(k,k), C] = gTrig(M(i), S(i,i), angi);
  S = fillIn(i,j,k,L,S,C,[]);
end

% % 3) Augment updated-filter ---------------------------------------------------
i = D2+1:D3; j = 1:D3; k = D3+1:D4;
if derivativesRequested
  [M(k), S(k,k), Ca, V(k,k), ...
    mdm, sdm, cadm, vdm, mds, sds, cads, vds, mdv, sdv, cadv, vdv] = ...
    gTrigh(M(i), S(i,i), V(i,i), angi);
  [S,V,Mds,Sds,Vds] = fillIn(i,j,k,L,S,Ca,V, ...
    mdm,sdm,vdm,cadm,mds,sds,vds,cads,mdv,sdv,vdv,cadv, ...
    Mds,Sds,Vds,[],[],[],[]);
else
  [M(k), S(k,k), Ca, V(k,k)] = gTrigh(M(i), S(i,i), V(i,i), angi);
  [S,V] = fillIn(i,j,k,L,S,Ca,V);
end

% 4) Compute distribution of the control signal -------------------------------
i = D2+poli; j = 1:D4; k = D4+1:D5; kk = sub2ind2(L,k,k); ii = sub2ind2(L,i,i);
if derivativesRequested
  [M(k), S(k,k), C, mdm, sdm, cdm, mds, sds, cds, Mdp, Sdp, Cdp] = ...
    ctrl.policy.fcn(ctrl.policy, M(i), S(i,i));
  if isfield(ctrl, 'actuate'), ctrl.actuate(M(k)); end     % actuate controller
  [S,~,Mds,Sds,~,~,Mdp,Sdp,~] = fillIn(i,j,k,L,S,C,[], ...
    mdm,sdm,[],cdm,mds,sds,[],cds,[],[],[],[], ...
    Mds,Sds,[],Mdp,Sdp,[],Cdp);
  uM = M(k); uS = S(k,k);
  duMds = Mds(k,:); duSds = Sds(kk,:);
  duMdp = Mdp(k,:); duSdp = Sdp(kk,:);
  se = [s.s\s.zc, I]; ec = [I Ca]; ec = ec(:,poli);
  uC = se * w' * ec * C;
  duCdp = prod_a_dx(se*w'*ec, Cdp);
  % most duCds chain terms:
  ia = D2+1:D3; iia = sub2ind2(L,ia,ia);
  dec = [zeros(DD,ns); cadm*Mds(ia,:) + cads*Sds(iia,:) + cadv*Vds(iia,:)];
  dec = dec(sub2ind2(D,1:D,poli),:);
  duCds = prod_a_dx_b(se*w', dec, C) + ...
    prod_a_dx(se*w'*ec, cdm*Mds(i,:)+cds*Sds(ii,:));
  % extra duCds chain terms:
  duCds(:,is.s) = duCds(:,is.s) - kron((s.zc*w1'*ec*C)'/s.s,inv(s.s));
  duCds(:,is.zc) = duCds(:,is.zc) + kron((w1'*ec*C)',inv(s.s));
  dwtdv = kron(w1,[-I;I]/(v+n));
  duCds(:,is.v) = duCds(:,is.v) + prod_a_dx_b(se,dwtdv,ec*C);
else
  [M(k), S(k,k), C] = ctrl.policy.fcn(ctrl.policy, M(i), S(i,i));
  if isfield(ctrl, 'actuate'), ctrl.actuate(M(k)); end     % actuate controller
  uM = M(k); uS = S(k,k); uC=nan(D,U);
  if nargout>=3 && ~all(all(s.s==0))
    ec=[I Ca]; ec = ec(:,poli); uC=[s.s\s.zc, I]*w'*ec*C;
  end
  if nargout<=3; return; end
  S = fillIn(i,j,k,L,S,C,[]);
end

% 5) Bayes-filter predict step ------------------------------------------------
i = [D2+dyni, D4+1:D5]; j = 1:D5; k = D5+1:D6;
if derivativesRequested
  [M(k),S(k,k),C,V(k,k),mdm,sdm,cdm,vdm,mds,sds,cds,vds,mdv,sdv,cdv,vdv] = ...
    ctrl.dynmodel.fcn(ctrl.dynmodel, M(i), S(i,i), V(i,i));
  [S,V,Mds,Sds,Vds,~,Mdp,Sdp,Vdp,~] = fillIn(i,j,k,L,S,C,V, ...
    mdm,sdm,vdm,cdm,mds,sds,vds,cds,mdv,sdv,vdv,cdv, ...
    Mds,Sds,Vds,Mdp,Sdp,[],[]);
else
  [M(k),S(k,k),C,V(k,k)] = ctrl.dynmodel.fcn(ctrl.dynmodel,M(i),S(i,i),V(i,i));
  [S,V] = fillIn(i,j,k,L,S,C,V);
end

% 6) Select distribution of predicted-filter ----------------------------------
k = D5+1:D6; kk = sub2ind2(L,k,k);
s.zm = M(k); s.zs = S(k,k); s.v = V(k,k);
if derivativesRequested
  [Mds,Sds,Vds,Sdp,Vdp,duCds]= symmetrise(XS,XST,XT,Mds,Sds,Vds,Sdp,Vdp,duCds);
  dsds(is.zm,:) = Mds(k,:); dsds(is.zs,:) = Sds(kk,:); dsds(is.v,:) =Vds(kk,:);
  dsdp(is.zm,:) = Mdp(k,:); dsdp(is.zs,:) = Sdp(kk,:); dsdp(is.v,:) =Vdp(kk,:);
end

% 7) Finally, covariance of next-state and predicted filter -------------------
% Idea: Save on computation by combine with propagate's dyn-model step?         % TODO
% Idea: use above step to debug: check the z-pred output is the same.           % TODO
if isfield(ctrl,'computeZC') && ~ctrl.computeZC; s=rmfield(s,'zc'); return; end
d = length(dyni); E = size(ctrl.dynmodel.beta,2); EE = E*E;
dynX = dynmodel; dynZ = ctrl.dynmodel;
% dyn-hyps:
for e=1:E                                                          % D --> 2d+U
  dynX.hyp(e).l = [dynX.hyp(e).l(1:d); inf(  d,1); dynX.hyp(e).l(d+1:end)];
  dynX.hyp(e).m = [dynX.hyp(e).m(1:d); zeros(d,1); dynX.hyp(e).m(d+1:end)];
  dynZ.hyp(e).l = [                    inf(  d,1); dynZ.hyp(e).l         ];
  dynZ.hyp(e).m = [                    zeros(d,1); dynZ.hyp(e).m         ];
end
% dyn-inputs:                                                                   % TODO handle nx ~= nz. Perhaps by zeroing corresponding beta values.
x = dynX.inputs; [nx,~,pE] = size(x);
x = cat(2,x(:,1:d,:),zeros(nx,d,pE),x(:,d+1:end,:));
x = repmat(x,[1,1,E/pE]);
z = dynZ.inputs; [nz,~,pE] = size(z);
z = cat(2,zeros(nz,d,pE),z);
z = repmat(z,[1,1,E/pE]);
dyn.hyp = [dynX.hyp,dynZ.hyp];                % 1 x 2E              % E -- > 2E % TODO handle sub-dynmodels.
dyn.inputs = cat(3,x,z);                      % nxDx2E                          % TODO leave as 'inputs'?
dyn.iK = cat(3,dynX.iK,dynZ.iK);              % nxnx2E
dyn.beta = cat(2,dynX.beta,dynZ.beta);        % n x 2E
% call gph to compute s.zc
i = [D0+dyni, D2+dyni, D4+1:D5]; ii = sub2ind2(L,i,i);
S_ = S; % i1 = D0+1:D1; S_(i1,i1) = S(i1,i1)-n;         % de-noise: Y_t --> X_t % TODO: reverting to Sx causes non-PSD problems in gphd below, but it should eb using Sx?!
ix = 1:E; iz = E+1:2*E;
if derivativesRequested
  [~,zc,~,~,~,sdm,~,~,~,sds,~,~,~,sdv] = gphd(dyn,M(i),S_(i,i),V(i,i));
  ixz = sub2ind2(2*E,ix,iz);
  sdm = reshape(sdm,4*EE,[]); sdm = sdm(ixz,:);
  sds = reshape(sds,4*EE,[]); sds = sds(ixz,:);
  sdv = reshape(sdv,4*EE,[]); sdv = sdv(ixz,:);
  dsds(is.zc,:) = sdm*Mds(i,:) + sds*Sds(ii,:) + sdv*Vds(ii,:);
  dsdp(is.zc,:) = sdm*Mdp(i,:) + sds*Sdp(ii,:) + sdv*Vdp(ii,:);
else
  [~,zc] = gph(dyn,M(i),S_(i,i),V(i,i));
end
s.zc = zc(ix,iz);

% FUNCTIONS -------------------------------------------------------------------

% A) Apply chain rule and fill out cross covariance terms ---------------------
function [S, V, Mds, Sds, Vds, Cds, Mdp, Sdp, Vdp, Cdp] = ...
  fillIn(i,j,k,L,S,C,V, ...
  mdm,sdm,vdm,cdm,mds,sds,vds,cds,mdv,sdv,vdv,cdv, ...
  Mds,Sds,Vds,Mdp,Sdp,Vdp,Cdp)

q = S(j,i)*C; S(j,k) = q; S(k,j) = q';                           % off-diagonal
if ~isempty(V); q = V(j,i)*C; V(j,k) = q; V(k,j) = q'; end       % off-diagonal
if nargout <= 2, return, end;

if isempty(k), return; end
if isempty(vdm), vdm=0*sdm; end; if isempty(vds), vds=0*sds; end
if isempty(mdv), mdv=0*mds; end; if isempty(sdv), sdv=0*sds; end
if isempty(vdv), vdv=0*sds; end; if isempty(cdv), cdv=0*cds; end
if isempty(Vds), Vds=0*Sds; end; if isempty(Vdp), Vdp=0*Sdp; end

a = length(k); b = numel(S(k,k)); c = numel(C);       % reshape new derivatives
mdm=reshape(mdm,a,[]); mds=reshape(mds,a,[]); mdv=reshape(mdv,a,[]);
sdm=reshape(sdm,b,[]); sds=reshape(sds,b,[]); sdv=reshape(sdv,b,[]);
vdm=reshape(vdm,b,[]); vds=reshape(vds,b,[]); vdv=reshape(vdv,b,[]);
cdm=reshape(cdm,c,[]); cds=reshape(cds,c,[]); cdv=reshape(cdv,c,[]);

ii = sub2ind2(L,i,i); kk = sub2ind2(L,k,k);                % vectorised indices
ji = sub2ind2(L,j,i); jk = sub2ind2(L,j,k);
kj = kron(k,ones(1,length(j))) + kron(ones(1,length(k)),(j-1)*L);

Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:) + mdv*Vds(ii,:);           % chainrule
Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:) + sdv*Vds(ii,:);
Vds(kk,:) = vdm*Mds(i,:) + vds*Sds(ii,:) + vdv*Vds(ii,:);
Cds       = cdm*Mds(i,:) + cds*Sds(ii,:) + cdv*Vds(ii,:);
if isempty(Cdp) && nargout > 6
  Mdp(k,:)  = mdm*Mdp(i,:) + mds*Sdp(ii,:) + mdv*Vdp(ii,:);
  Sdp(kk,:) = sdm*Mdp(i,:) + sds*Sdp(ii,:) + sdv*Vdp(ii,:);
  Vdp(kk,:) = vdm*Mdp(i,:) + vds*Sdp(ii,:) + vdv*Vdp(ii,:);
  Cdp       = cdm*Mdp(i,:) + cds*Sdp(ii,:) + cdv*Vdp(ii,:);
elseif nargout > 6
  mdp = zeros(L,size(Mdp,2)); sdp = zeros(L*L,size(Mdp,2)); vdp = sdp;
  mdp(k,:)  = reshape(Mdp,a,[]); Mdp = mdp;
  sdp(kk,:) = reshape(Sdp,b,[]); Sdp = sdp;
  vdp(kk,:) = reshape(Vdp,b,[]); Vdp = vdp;
  cdp       = reshape(Cdp,c,[]); Cdp = cdp;
end

SS = kron(eye(a),S(j,i)); CC = kron(C',eye(length(j)));
Sds(jk,:) = SS*Cds + CC*Sds(ji,:); Sds(kj,:) = Sds(jk,:);
if ~isempty(V);
  VV=kron(eye(a),V(j,i));
  Vds(jk,:) = VV*Cds + CC*Vds(ji,:); Vds(kj,:) = Vds(jk,:);
end
if nargout > 6;
  Sdp(jk,:) = SS*Cdp + CC*Sdp(ji,:); Sdp(kj,:) = Sdp(jk,:);
  if ~isempty(V); Vdp(jk,:) = VV*Cdp + CC*Vdp(ji,:); Vdp(kj,:) = Vdp(jk,:); end
end

% B) Get linear indexes from all combinations of two subscripts ---------------
function idx = sub2ind2(D,i,j)
% D = #rows, i = row subscript, j = column subscript
i = i(:); j = j(:)';
idx =  reshape(bsxfun(@plus,D*(j-1),i),1,[]);

% C) Symmetrise the numerator (row) and denomiator (column) cross covariances
function [Mds, Sds, Vds, Sdp, Vdp, duCds] = ...
  symmetrise(XS, XST, XT, Mds, Sds, Vds, Sdp, Vdp, duCds)
Mds(:,XS) = (Mds(:,XS) + Mds(:,XST))/2;
Sds = (Sds + Sds(XT,:))/2; Sds(:,XS) = (Sds(:,XS) + Sds(:,XST))/2;
Vds = (Vds + Vds(XT,:))/2; Vds(:,XS) = (Vds(:,XS) + Vds(:,XST))/2;
if nargout >= 4, Sdp = (Sdp + Sdp(XT,:))/2; end
if nargout >= 5, Vdp = (Vdp + Vdp(XT,:))/2; end
if nargout >= 6, duCds(:,XS) = (duCds(:,XS) + duCds(:,XST))/2; end