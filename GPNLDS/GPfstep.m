function [M, S, nlp, nlpdp, Mdp, Sdp] = GPfstep(varargin)
% Edited Rowan McAllister 2014-11-7

if nargout < 4
    [M, S, nlp] = GPfilt(varargin{:});
else
    [M, S, nlp, nlpdp, Mdp, Sdp] = GPfiltd(varargin{:});
end


function [M, S, nlp] = GPfilt(p,dyn,plant,m,s,y,u)
% function to compute the filter distribution at the current time step
%       p(x_t | z_1:t)
% given a distribution over the previous state and a measurement of the current
% state.

E = length(m); Sp = exp(2*[p.pn]); So = diag(exp(2*[p.on]));

if ~isempty(dyn)
    Du = length(u); 
    
    % augment with trigonometric functions
    Da = length(plant.angi); i = E+1:E+2*Da;
    [m(i), s(i,i), C] = gTrig(m,s,plant.angi);
    s(1:E,i) = s(1:E,1:E)*C; s(i,1:E) = s(1:E,i)';
    
    % add the controls
    m = [m;u];
    s = [s zeros(E+2*Da,Du);zeros(Du,E+2*Da) 1e-9*eye(Du)];
    
    % find the predictive distribution for the next time step
    [Mx, Sx] = gpm(dyn,dyn.inputs,m,s);

    % Add process noise to latent state variance
    Sx = Sx + diag(Sp);
else
    Mx = m; Sx = s;
end

% find the predicted distribution of observation at the next time step
Mz = Mx; Sz = Sx + So; Cxz = Sx; % same mean, expanded variance

% calculate the negative log probability of the observation
ismy = Sz\(Mz-y);
nlp = E/2*log(2*pi) + log(det(Sz))/2 + (Mz-y)'*ismy/2;

% condition on the actual observation. This is equivilant to multiplying
% the predicted distribution with the observation distribution: N(y_t,S_n)
isym = Sz\(y-Mz);
M = Mx + Cxz*isym;
S = So - So*(Sz\So);    % a more stable version of S = Sx - Cxz*(Sz\Cxz');

function [M, S, nlp, nlpdp, Mdp, Sdp] = GPfiltd(p,dyn,plant,m,s,y,u,mdp,sdp)
% function to compute the filter distribution at the current time step
%       p(x_t | z_1:t)
% given a distribution over the previous state and a measurement of the current
% state.

E = length(m); Sp = diag(exp(2*[p.pn])); So = diag(exp(2*[p.on]));
Np = length(unwrap(p)); idx = rewrap(p,1:Np); I = diag(true(1,E));
if nargin < 8; mdp = zeros(E,Np); sdp = zeros(E^2,Np); end

if ~isempty(dyn)
    Du = length(u); Da = length(plant.angi); D1 = E + 2*Da;
    mdm = eye(E); mds = zeros(E,E^2);
    sdm = zeros(D1^2,E); sds = zeros(D1^2,E^2);
    ii = sub2ind2(D1,1:E,1:E); sds(ii,:) = eye(E^2);
    
    % augment with trigonometric functions
    i = 1:E; k = E+1:E+2*Da; kk = sub2ind2(D1,k,k);
    [m(k), s(k,k), C, mdm(k,:), sdm(kk,:), Cdm, mds(k,:), sds(kk,:), Cds] = ...
                                                gTrig(m,s,plant.angi);
    s(i,k) = s(i,i)*C; s(k,i) = s(i,k)';
    ik = sub2ind2(D1,i,k); ki = sub2ind2(D1,k,i);
    sdm(ik,:) = reshape(s(i,i)*reshape(Cdm,E,2*Da*E),2*Da*E,E);
    sCds = zeros(E,2*Da,E,E); for p=1:E;for q=1:E; sCds(p,:,p,q) = C(q,:); end; end
    sds(ik,:) = reshape(s(i,i)*reshape(Cds,E,2*Da*E^2),2*Da*E,E^2) ...
                            + reshape(sCds,E*2*Da,E^2);
    sdm(ki,:) = reshape(permute(reshape(sdm(ik,:),E,2*Da,E),[2,1,3]),2*E*Da,E);
    sds(ki,:) = reshape(permute(reshape(sds(ik,:),E,2*Da,E^2),[2,1,3]),2*E*Da,E^2);
    
    % add the controls
    m = [m;u];
    s = [s zeros(D1,Du);zeros(Du,D1) 1e-9*eye(Du)];
    
    % find the predictive distribution for the next time step
    [Mx, Sx, Mxdm, Sxdm, Mxds, Sxds, Mxdp, Sxdp] = gpD(dyn,m,s);
    ii = sub2ind2(D1+Du,1:D1,1:D1);
    dMxdm = Mxdm(:,1:D1)*mdm + Mxds(:,ii)*sdm; 
    dMxds = Mxdm(:,1:D1)*mds + Mxds(:,ii)*sds;
    dSxdm = Sxdm(:,1:D1)*mdm + Sxds(:,ii)*sdm; 
    dSxds = Sxdm(:,1:D1)*mds + Sxds(:,ii)*sds;
    Mxdp = dMxdm*mdp + dMxds*sdp + Mxdp;
    Sxdp = dSxdm*mdp + dSxds*sdp + Sxdp;

    % Add process noise to latent state variance
    Sx = Sx + Sp; 
    Sxdp(I,[idx.pn]) = Sxdp(I,[idx.pn]) + 2*Sp;
else
    Mx = m; Sx = s; Mxdp = mdp; Sxdp = sdp;
end

% find the predicted distribution of observation at the next time step
Mz = Mx; Sz = Sx + So; Cxz = Sx; % same mean, expanded variance
Szdp = Sxdp; Szdp(I,[idx.on]) = Szdp(I,[idx.on]) + 2*So;

% calculate the negative log probability of the observation & derivatives
ismy = Sz\(Mz-y);
nlp = E/2*log(2*pi) + log(det(Sz))/2 + (Mz-y)'*ismy/2;
nlpdM = ismy; nlpdSz = Sz\eye(E)/2 - ismy*ismy'/2;
nlpdp = nlpdM'*Mxdp + nlpdSz(:)'*Szdp;

% condition on the actual observation. This is equivilant to multiplying
% the predicted distribution with the observation distribution: N(y_t,S_n)
isym = Sz\(y-Mz);
M = Mx + Cxz*isym;
% S = Sx - Cxz*(Sz\Cxz');
S = So - So*(Sz\So);
CiS = Cxz/Sz;
sxdp1 = reshape(Sxdp,E,[]); sxdp2 = reshape(Sxdp',[],E);
szdp2 = reshape(Szdp',[],E);
Mdp = Mxdp + reshape(sxdp2*isym,[],E)' - CiS*reshape(szdp2*isym,[],E)' ...
            - CiS*Mxdp;
Sdp = Sxdp - reshape(sxdp2*CiS',[],E^2)' + ...
        reshape(CiS*reshape(reshape(szdp2*CiS',[],E^2)',E,[]),E^2,[])...
        - reshape(CiS*sxdp1,E^2,[]);
    
function dp = dpStruct(dp,dyn)
% fields of dp are of size Dp-by-*, where Dp is the vectorised quantity
% being differentiated (e.g. output mean, variance, etc) and * are the
% dimensions of the parameter held in p(i).<parametername>
names = fieldnames(dp); Ddp = size(dp(1).(names{1}),1);
[N, D] = size(dyn.inputs);

if ~isfield(dp,'b'); [dp.b] = deal(zeros(Ddp,1)); end
if ~isfield(dp,'beta'); [dp.beta] = deal(zeros(Ddp,N)); end
if ~isfield(dp,'l'); [dp.l] = deal(zeros(Ddp,D)); end
if ~isfield(dp,'m'); [dp.m] = deal(zeros(Ddp,D)); end
if ~isfield(dp,'n'); [dp.n] = deal(zeros(Ddp,1)); end
if ~isfield(dp,'on'); [dp.on] = deal(zeros(Ddp,1)); end
if ~isfield(dp,'pn'); [dp.pn] = deal(zeros(Ddp,1)); end
if ~isfield(dp,'s'); [dp.s] = deal(zeros(Ddp,1)); end  