function [mj, Sj, dynmodel, dmjdm, dSjdm, dmjds, dSjds, dmjdp, dSjdp] = ...
                                 propagatecd(mu, Sigma, plant, dynmodel, policy)

% Propagate the state distribution one time step forward with derivatives
%
% dm6dmu   derivative of output mean wrt input mean (ExD matrix)
% dm6dS    derivative of output mean wrt input covariance matrix (ExDxD matrix)
% dS6dmu   derivative of output covariance matrix wrt input mean (ExExD matrix)
% dS6dS    derivative of output cov wrt input cov (ExExDxD matrix)
% dm6dp    derivative of output mean wrt policy parameters
% dS6dp    derivative of output covariance matrix wrt policy parameters
%
% Copyright (C) 2008-2011 by Marc Deisenroth, Carl Edward Rasmussen, Henrik
% Ohlsson, Andrew McHutchon and Joe Hall, 2011-11-30

if nargout <= 3                                  % just predict, no derivatives
  [mj, Sj, dynmodel] = propagatec(mu, Sigma, plant, dynmodel, policy);
  return
end

angi = plant.angi; poli = plant.poli; dyni = plant.dyni; difi = plant.difi;

Do = length(plant.dyno); Du = length(plant.maxU);
D0 = length(mu);                          % size of joint distribution, Di + Do
D1 = D0 + 2*length(angi);          % length after mapping all angles to sin/cos
D2 = D1 + Du;                    % length after computing unsquashed ctrl signal
D3 = D2 + Du;                               % length after squashing ctrl signal
D4 = D3 + Do;                                         % length after predicting
m = zeros(D4,1); m(1:D0) = mu; S = zeros(D4); S(1:D0,1:D0) = Sigma; % init m, S
Dold = D0 - Do;                 % on first time step this = 0, otherwise = dyni
angi = Dold + angi; poli = Dold + poli; dyni = Dold + dyni; difi = Dold + difi;

% 1) augment the state distribution with trigonometric functions
[m(1:D1), S(1:D1,1:D1), dmdm, dmds, dsdm, dsds] = ...
                                          trigaug(m(1:D0), S(1:D0,1:D0), angi);
noise = zeros(D0); noise(Dold+1:D0,Dold+1:D0) = diag(exp(2*dynmodel.hyp(end,:))/2);
[mm, SS] = trigaug(m(1:D0), S(1:D0,1:D0) + noise, angi);

% 2) compute the distribution of the unsquashed control signal
i = poli; j = 1:D1; k = D1+1:D2;
[m(k) S(k,k) C mdm sdm Cdm mds sds Cds dmdp(k,:) dsdp(k,k,:) Cdp] = ...
                                            policy.fcn(policy, mm(i), SS(i,i));
[S dmdm dmds dsdm dsds dmdp dsdp] = ...
     fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,dmdm,dsdm,dmds,dsds,dmdp,dsdp,Cdp,i,j,k);
Cu = {C, Cdm, Cds};

% 3) squash the control signal
i = 1:D2; j = i; k = D2+1:D3;
[m(k) S(k,k) C mdm sdm Cdm mds sds Cds] = gSin(m(i), S(i,i), D1+1:D2, plant.maxU);

[S dmdm dmds dsdm dsds dmdp dsdp] = ...
      fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,dmdm,dsdm,dmds,dsds,dmdp,dsdp,[],i,j,k);
Csu = {C, Cdm, Cds};
try(chol(S([Dold+1:D0 D2+1:D3],[Dold+1:D0 D2+1:D3]))); 
catch; fprintf('propd3: S not pos def.\n'); keyboard; end

% 4) compute the distribution of the change in state
i = [dyni D2+1:D3]; j = 1:D3; k = D3+1:D4;
[m(k) S(k,k) C mdm sdm Cdm mds sds Cds] = dynmodel.fcn(dynmodel, m(i), S(i,i));

[S dmdm dmds dsdm dsds dmdp dsdp] = ...
      fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,dmdm,dsdm,dmds,dsds,dmdp,dsdp,[],i,j,k);
C = {C, Cdm, Cds};
try(chol(S([Dold+1:D0 D2+1:D4],[Dold+1:D0 D2+1:D4]))); 
catch; fprintf('propd4: S not pos def.\n'); keyboard; end
  
% 5) correct variance due to previous state transition
[S dynmodel, sdm sds] = correctVar(m,S,dynmodel,plant,Cu,Csu,C);
% k = [dyni D2+1:D4]; i = 1:D3;
% p = {dmdm(i,:), dsdm(i,i,:), dmds(i,:,:), dsds(i,i,:,:)};
% [dsdm(k,k,:) dsds(k,k,:,:)] = chainrule(sdm(k,k,i),sds(k,k,i,i),p,2);
% p = {dmdp, dsdp};
% dsdp(k,k,:) = chainrule(sdm(k,k,:), sds(k,k,:,:), p, 2);
p = {dmdm, dsdm, dmds, dsds};
[dsdm dsds] = chainrule(sdm,sds,p,2);
p = {dmdp, dsdp};
dsdp = chainrule(sdm, sds, p, 2);
try(chol(S([Dold+1:D0 D2+1:D4],[Dold+1:D0 D2+1:D4]))); 
catch; fprintf('propd5: S not pos def.\n'); keyboard; end

% 6) compute the distribution of the next state
i = D3+1:D4; di = difi-Dold; nd = setdiff(1:Do,di); % di,nd relative to just current state
m6 = m(i); m6(di) = m6(di) + m(difi);
S6 = S(i,i); 
S6(di,di) = S6(di,di) + S(difi,i(di)) + S(i(di),difi) + S(difi,difi);
S6(nd,di) = S6(nd,di) + S(i(nd),difi); S6(di,nd) = S6(di,nd) + S(di,i(nd));

dm6dm = dmdm(i,:); dm6dm(di,:) = dm6dm(di,:) + dmdm(difi,:);
dm6ds = dmds(i,:,:); dm6ds(di,:,:) = dm6ds(di,:,:) + dmds(difi,:,:);

dS6dm = dsdm(i,i,:);
dS6dm(di,di,:) = dS6dm(di,di,:) + dsdm(difi,i(di),:) + ...
                                      dsdm(i(di),difi,:) + dsdm(difi,difi,:);
dS6dm(nd,di,:) = dS6dm(nd,di,:) + dsdm(i(nd),difi,:);
dS6dm(di,nd,:) = dS6dm(di,nd,:) + dsdm(difi,i(nd),:);

dS6ds = dsds(i,i,:,:);
dS6ds(di,di,:,:) = dS6ds(di,di,:,:) + dsds(difi,i(di),:,:) + ...
                                  dsds(i(di),difi,:,:) + dsds(difi,difi,:,:);
dS6ds(nd,di,:,:) = dS6ds(nd,di,:,:) + dsds(i(nd),difi,:,:);
dS6ds(di,nd,:,:) = dS6ds(di,nd,:,:) + dsds(difi,i(nd),:,:);

dm6dp = dmdp(i,:);
dS6dp = dsdp(i,i,:);
dS6dp(di,di,:) = dS6dp(di,di,:) + dsdp(difi,i(di),:) + dsdp(i(di),difi,:);
dS6dp(nd,di,:,:) = dS6dp(nd,di,:,:) + dsdp(i(nd),difi,:,:);
dS6dp(di,nd,:,:) = dS6dp(di,nd,:,:) + dsdp(difi,i(nd),:,:);

% Compute covariance between Sigma(dyni) and S6
i = [dyni D2+1:D3]; Css6 = zeros(length(i),Do);
Css6(:,nd) = S(i,D3+nd);                           % C(x, y), non-difi variables
Css6(:,di) = S(i,D3+di) + S(i,difi); % C([w; x], x+y) = [C(w,x); V(x)] + C([w; x],y)
dCdm = zeros(length(i),Do,D0); dCds = zeros(length(i),Do,D0,D0);
dCdm(:,nd,:) = dsdm(i,D3+nd,:); dCdm(:,di,:) = dsdm(i,D3+di,:) + dsdm(i,difi,:);
dCds(:,nd,:,:) = dsds(i,D3+nd,:,:); dCds(:,di,:,:) = dsds(i,D3+di,:,:) + dsds(i,difi,:,:);
dCdp(:,nd,:) = dsdp(i,D3+nd,:); dCdp(:,di,:) = dsdp(i,D3+di,:) + dsdp(i,difi,:);

mj = [m(i); m6]; Sj = [S(i,i) Css6; Css6' S6];

dmjdm = [dmdm(i,:); dm6dm]; dmjds = [dmds(i,:,:); dm6ds]; dmjdp = [dmdp(i,:); dm6dp];
dSjdm = [dsdm(i,i,:) dCdm; permute(dCdm,[2,1,3]) dS6dm];
dSjds = [dsds(i,i,:,:) dCds; permute(dCds,[2,1,3,4]) dS6ds];
dSjdp = [dsdp(i,i,:) dCdp; permute(dCdp,[2,1,3]) dS6dp];

try(chol(Sj)); catch; fprintf('Sj not pos def.\n'); keyboard; end

function [S dmdm dmds dsdm dsds dmdp dsdp] = ...
         fillIn(S,C,mdm,sdm,Cdm,mds,sds,Cds,dmdm,dsdm,dmds,dsds,dmdp,dsdp,dCdp,i,j,k)

p = {dmdm(i,:), dsdm(i,i,:), dmds(i,:,:), dsds(i,i,:,:)};
[dmdm(k,:), dmds(k,:,:)] = chainrule(mdm, mds, p, 1);
[dsdm(k,k,:), dsds(k,k,:,:)] = chainrule(sdm, sds, p, 2);
[dCdm, dCds] = chainrule(Cdm, Cds, p, 2);
if isempty(dCdp)
    p = {dmdp(i,:), dsdp(i,i,:)};
    dmdp(k,:) = chainrule(mdm, mds, p, 1);
    dsdp(k,k,:) = chainrule(sdm, sds, p, 2);
    dCdp = chainrule(Cdm, Cds, p, 2);
end

q = S(j,i)*C; S(j,k) = q; S(k,j) = q';                           % off-diagonal
dsdm(j,k,:) = SdC(S(j,i),dCdm) + dSC(dsdm(j,i,:),C);
dsdm(k,j,:) = permute(dsdm(j,k,:),[2 1 3]);
dsds(j,k,:,:) = SdC(S(j,i),dCds) + dSC(dsds(j,i,:,:),C);
dsds(k,j,:,:) = permute(dsds(j,k,:,:),[2 1 3 4]);
dsdp(j,k,:) = SdC(S(j,i),dCdp) + dSC(dsdp(j,i,:),C);
dsdp(k,j,:) = permute(dsdp(j,k,:),[2 1 3]);


function dx = SdC(S,dC)                % multiply matrix/vector with derivative
cx = ndims(dC);
dx = etprod([1 3:cx+1],S,[1 2],dC,2:2+cx-1);


function dx = dSC(dS,C)                % multiply derivative with matrix/vector
sx = ndims(dS);
cx = ndims(C);
dx = etprod([1 3:3+sx-1], dS, [1:2 2+cx:cx+sx-1], C, 2:2+cx-1);


function [dxdm, dxdS] = chainrule(dxda, dxdB, p, sx)  % chainrule for Gaussians
dadm = p{1}; dBdm = p{2};
sa = max(ndim(dxda)-sx,1);                % dim of tensor a (dim dx/da - dim x)
sB = max(ndim(dxdB)-sx,2);                % dim of tensor B (dim dx/dB - dim x)
sm = max(ndim(dadm)-sa,1);                % dim of tensor m (dim da/dm - dim a)

                    % compute tensor product: dx/dm = dx/da*da/dm + dx/dB*dB/dm
dxdm = etprod([1:sx sx+sa+1:sx+sa+sm], dxda, 1:sx+sa, dadm, sx+1:sx+sa+sm) ...
         + etprod([1:sx sx+sB+1:sx+sB+sm], dxdB, 1:sx+sB, dBdm, sx+1:sx+sB+sm);

if nargout > 1
  dadS = p{3}; dBdS = p{4};
  sS = max(ndim(dBdS)-sB,2);
                   % compute tensor product: dx/dS = dx/da*da/dS + dx/dB*dB/dS
  dxdS = etprod([1:sx sx+sa+1:sx+sa+sS], dxda,1:sx+sa, dadS, sx+1:sx+sa+sS) ...
         + etprod([1:sx sx+sB+1:sx+sB+sS], dxdB, 1:sx+sB, dBdS, sx+1:sx+sB+sS);
end


function sx = ndim(x)
if length(size(x))==2 && size(x,2)==1
  sx = 1;
else
  sx = length(size(x));
end
