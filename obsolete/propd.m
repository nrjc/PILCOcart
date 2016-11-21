function [m6, S6, dm6dmu, dS6dmu, dm6dS, dS6dS, dm6dp, dS6dp] = ...
                                       propd(mu, Sigma, plant, dynmodel, policy)

% Propagate the distribution of the state one time step forward.
%
% Copyright (C) 2008-2011 by Carl Edward Rasmussen, Marc Deisenroth,
%                     Henrik Ohlsson, Andrew McHutchon and Joe Hall, 2011-12-06

if nargout <= 2                                  % just predict, no derivatives
  [m6, S6] = prop(mu, Sigma, plant, dynmodel, policy);
  return
end

angi = plant.angi; poli = plant.poli; dyni = plant.dyni; difi = plant.difi;

D1 = length(mu); D0 = D1/2; D2 = D1 + D0; D3 = D2 + D0; 
D4 = D3 + 2*length(angi); D5 = D4 + 2*length(angi); D6 = D5 + 2*length(angi);
D7 = D6 + length(plant.maxU); D8 = D7 + 2*length(plant.maxU); 
D9 = D8 + D0; Da = D9 + D0;

m = zeros(Da,1); m(1:D1) = mu; S = zeros(Da); S(1:D1,1:D1) = Sigma; % init m, S
dmdm = zeros(D3,D1); dsds = zeros(D3,D3,D1,D1);              % init derivatives 
for i=1:D1, dmdm(i,i) = 1; for j=1:D1, dsds(i,j,i,j) = 1; end, end
dmds = zeros(D3,D1,D1); dsdm = zeros(D3,D3,D1);

i = 1:D0; j = D0+1:D1; ij = [1:D1]; k = D1+1:D2; l = D2+1:D3;  % useful indices
Ss = S(i,i); Sr = S(j,j); Sc = S(i,j); Sn = diag(exp(2*dynmodel.hyp(end,:))/2);
Zr = Sr/(Sr+Sn); Zn = Sn/(Sr+Sn);

% 1) add the belief variable, b
m(k) = [Zr Zn]*m(ij); dmdm(k,ij) = [Zr Zn];                              % mean
dmds(k,j,j) = etprod('123',Zn,'12',(Sr+Sn)\(m(i)-m(j)),'3');
S(k,k) = [Zr Zn]*[Ss+Sn Sc; Sc' Sr]*[Zr Zn]';                        % variance
dsds(k,k,ij,ij) = etprod('1234',[Zr Zn],'13',[Zr Zn]','42');
q = etprod('1234',Zn,'13',(Sr+Sn)\((Ss+Sn-Sc')*Zr'+(Sc-Sr)*Zn'),'42');
dsds(k,k,j,j) = dsds(k,k,j,j) + q + permute(q,[2 1 3 4]);
q = Zr*Ss + Zn*Sc'; S(k,i) = q; S(i,k) = q';                         % cov(b,s)
dsds(k,i,i,i) = etprod('1234',Zr,'13',eye(D0),'42');
dsds(k,i,j,i) = etprod('1234',Zn,'13',eye(D0),'42');
dsds(k,i,j,j) = etprod('1234',Zn,'13',(Sr+Sn)\(Ss-Sc'),'42');
dsds(i,k,ij,ij) = permute(dsds(k,i,ij,ij),[2 1 3 4]);      % symmetric elements
q = Zr*Sc + Zn*Sr; S(k,j) = q; S(j,k) = q';                          % cov(b,r)
dsds(k,j,i,j) = etprod('1234',Zr,'13',eye(D0),'42');
dsds(k,j,j,j) = etprod('1234',Zn,'13',(Sr+Sn)\(Sc+Sn),'42');
dsds(j,k,ij,ij) = permute(dsds(k,j,ij,ij),[2 1 3 4]);      % symmetric elements

% 2) add ctrl input variable, a
m(l) = m(k); dmdm(l,ij) = dmdm(k,ij); dmds(l,j,j) = dmds(k,j,j);         % mean
S(l,l) = Zr*(Ss+Sn)*Zr';                                             % variance
dsds(l,l,i,i) = etprod('1234',Zr,'13',Zr','42');
q = etprod('1234',Zn,'13',(Sr+Sn)\(Ss+Sn)*Zr','42');
dsds(l,l,j,j) = q + permute(q,[2 1 3 4]);
S(l,i) = Zr*Ss; S(i,l) = S(l,i)';                                    % cov(a,s)
dsds(l,i,i,i) = etprod('1234',Zr,'13',eye(D0),'42');
dsds(l,i,j,j) = etprod('1234',Zn,'13',(Sr+Sn)\Ss,'42');
dsds(i,l,ij,ij) = permute(dsds(l,i,ij,ij),[2 1 3 4]);      % symmetric elements
S(l,j) = Zr*Sc; S(j,l) = S(l,j)';                                    % cov(a,r)
dsds(l,j,i,j) = etprod('1234',Zr,'13',eye(D0),'42');
dsds(l,j,j,j) = etprod('1234',Zn,'13',(Sr+Sn)\Sc,'42');
dsds(j,l,ij,ij) = permute(dsds(l,j,ij,ij),[2 1 3 4]);
S(l,k) = S(l,l); S(k,l) = S(l,l)';                                   % cov(a,b)
dsds(l,k,ij,ij) = dsds(l,l,ij,ij);
dsds(k,l,ij,ij) = permute(dsds(l,k,ij,ij),[2 1 3 4]);      % symmetric elements

% 3) augment with three sets of trigonometric functions
[m(1:D6) S(1:D6,1:D6) mdm mds sdm sds] = ...
                        trigaug(m(1:D3), S(1:D3,1:D3), [angi D1+angi D2+angi]);

p = {dmdm, dsdm, dmds, dsds};
[dmdm, dmds] = chainrule(mdm, mds, p, 1);
[dsdm, dsds] = chainrule(sdm, sds, p, 2);

% 4) compute distribution of unsquashed control signal
i = [D2+poli+(D5-D3)*(poli>D0)]; j = [1:D2 D3+1:D5]; k = [D6+1:D7];
[m(k) S(k,k) C dmdm2 dsdm2 dCdm2 dmds2 dsds2 dCds2 dmdp(k,:) dsdp(k,k,:) ...
                                      dCdp] = policy.fcn(policy, m(i), S(i,i));

p = {dmdm(i,:), dsdm(i,i,:), dmds(i,:,:), dsds(i,i,:,:)};
[dmdm(k,:) dmds(k,:,:)] = chainrule(dmdm2, dmds2, p, 1);
[dsdm(k,k,:) dsds(k,k,:,:)] = chainrule(dsdm2, dsds2, p, 2);
[dCdm dCds] = chainrule(dCdm2, dCds2, p, 2); 

q = S(j,i)*C; S(j,k) = q; S(k,j) = q';                          % off diag term
dsdm(j,k,:) = SdC(S(j,i),dCdm) + dSC(dsdm(j,i,:),C);
dsdm(k,j,:) = permute(dsdm(j,k,:),[2 1 3]);                         % symmetric
dsds(j,k,:,:) = SdC(S(j,i),dCds) + dSC(dsds(j,i,:,:),C);
dsds(k,j,:,:) = permute(dsds(j,k,:,:),[2 1 3 4]); 
dsdp(j,k,:) = SdC(S(j,i),dCdp);
dsdp(k,j,:) = permute(dsdp(j,k,:),[2 1 3]);

% 5) squash control signal
[m(1:D8) S(1:D8,1:D8) mdm mds sdm sds] = ...
                           trigaug(m(1:D7), S(1:D7,1:D7), D6+1:D7, plant.maxU);

p = {dmdm, dsdm, dmds, dsds};
[dmdm, dmds] = chainrule(mdm, mds, p, 1);
[dsdm, dsds] = chainrule(sdm, sds, p, 2);
p = {dmdp, dsdp};
dmdp = chainrule(mdm, mds, p, 1);
dsdp = chainrule(sdm, sds, p, 2);

% 6) compute new states
i = [dyni+(D3-D0)*(dyni>D0) D7+1:2:D8]; j = 1:D1; k = D8+1:D9;
[m(k) S(k,k) C mdm sdm Cdm mds sds Cds] = dynmodel.fcn(dynmodel, m(i), S(i,i));
p = {dmdm(i,:), dsdm(i,i,:), dmds(i,:,:), dsds(i,i,:,:)};
[dmdm(k,:), dmds(k,:,:)] = chainrule(mdm, mds, p, 1);
[dsdm(k,k,:), dsds(k,k,:,:)] = chainrule(sdm, sds, p, 2);
[dCdm, dCds] = chainrule(Cdm, Cds, p, 2);
p = {dmdp(i,:), dsdp(i,i,:)};
dmdp(k,:) = chainrule(mdm, mds, p, 1);
dsdp(k,k,:) = chainrule(sdm, sds, p, 2);
dCdp = chainrule(Cdm, Cds, p, 2);

q = S(j,i)*C; S(j,k) = q; S(k,j) = q';
dsdm(j,k,:) = SdC(S(j,i),dCdm) + dSC(dsdm(j,i,:),C);
dsdm(k,j,:) = permute(dsdm(j,k,:),[2 1 3]);
dsds(j,k,:,:) = SdC(S(j,i),dCds) + dSC(dsds(j,i,:,:),C);
dsds(k,j,:,:) = permute(dsds(j,k,:,:),[2 1 3 4]);
dsdp(j,k,:) = SdC(S(j,i),dCdp) + dSC(dsdp(j,i,:),C);
dsdp(k,j,:) = permute(dsdp(j,k,:),[2 1 3]);

l = [D1+dyni+(D4-D2)*(dyni>D0) D7+1:2:D8]; j = 1:D1; k = D9+1:Da;
[m(k) S(k,k) D mdm sdm Ddm mds sds Dds] = dynmodel.fcn(dynmodel, m(l), S(l,l));
p = {dmdm(l,:), dsdm(l,l,:), dmds(l,:,:), dsds(l,l,:,:)};
[dmdm(k,:), dmds(k,:,:)] = chainrule(mdm, mds, p, 1);
[dsdm(k,k,:), dsds(k,k,:,:)] = chainrule(sdm, sds, p, 2);
[dDdm, dDds] = chainrule(Ddm, Dds, p, 2);
p = {dmdp(l,:), dsdp(l,l,:)};
dmdp(k,:) = chainrule(mdm, mds, p, 1);
dsdp(k,k,:) = chainrule(sdm, sds, p, 2);
dDdp = chainrule(Ddm, Dds, p, 2);

q = S(j,l)*D; S(j,k) = q; S(k,j) = q';
dsdm(j,k,:) = SdC(S(j,l),dDdm) + dSC(dsdm(j,l,:),D);
dsdm(k,j,:) = permute(dsdm(j,k,:),[2 1 3]);
dsds(j,k,:,:) = SdC(S(j,l),dDds) + dSC(dsds(j,l,:,:),D);
dsds(k,j,:,:) = permute(dsds(j,k,:,:),[2 1 3 4]);
dsdp(j,k,:) = SdC(S(j,l),dDdp) + dSC(dsdp(j,l,:),D);
dsdp(k,j,:) = permute(dsdp(j,k,:),[2 1 3]);

q = C'*S(i,l)*D; S(D8+1:D9,D9+1:Da) = q; S(D9+1:Da,D8+1:D9) = q';
q = etprod('134',dCdm,'214',S(i,l)*D,'23') + ...
                                  etprod('134',C'*S(i,l),'12',dDdm,'234') + ... 
            etprod('134',etprod('134',C','12',dsdm(i,l,:),'234'),'124',D,'23');
dsdm(D8+1:D9,D9+1:Da,:) = q; dsdm(D9+1:Da,D8+1:D9,:) = permute(q,[2 1 3]);
q = etprod('1345',dCds,'2145',S(i,l)*D,'23') + ...
                                etprod('1345',C'*S(i,l),'12',dDds,'2345') + ...
      etprod('1345',etprod('1345',C','12',dsds(i,l,:,:),'2345'),'1245',D,'23');
dsds(D8+1:D9,D9+1:Da,:,:) = q; 
dsds(D9+1:Da,D8+1:D9,:,:) = permute(q,[2 1 3 4]);
q = etprod('134',dCdp,'214',S(i,l)*D,'23') + ...
                                  etprod('134',C'*S(i,l),'12',dDdp,'234') + ... 
            etprod('134',etprod('134',C','12',dsdp(i,l,:),'234'),'124',D,'23');
dsdp(D8+1:D9,D9+1:Da,:) = q; dsdp(D9+1:Da,D8+1:D9,:) = permute(q,[2 1 3]);

% 7) compute the distribution of the next state
i = D8+1:Da; difi = [difi D0+difi]; j = setdiff(1:D1,difi);
m6 = m(i); m6(difi) = m6(difi) + m(difi);
S6 = S(i,i); 
S6(difi,difi) = S6(difi,difi) + S(difi,i(difi)) + S(i(difi),difi) + ...
                                                                  S(difi,difi);
S6(j,difi) = S6(j,difi) + S(i(j),difi); S6(difi,j) = S6(difi,j) + S(difi,i(j));

dm6dmu = dmdm(i,:); dm6dmu(difi,:) = dm6dmu(difi,:) + dmdm(difi,:);
dm6dS = dmds(i,:,:); dm6dS(difi,:,:) = dm6dS(difi,:,:) + dmds(difi,:,:);

dS6dmu = dsdm(i,i,:);
dS6dmu(difi,difi,:) = dS6dmu(difi,difi,:) + dsdm(difi,i(difi),:) + ...
                                      dsdm(i(difi),difi,:) + dsdm(difi,difi,:);
dS6dmu(j,difi,:) = dS6dmu(j,difi,:) + dsdm(i(j),difi,:);
dS6dmu(difi,j,:) = dS6dmu(difi,j,:) + dsdm(difi,i(j),:);

dS6dS = dsds(i,i,:,:);
dS6dS(difi,difi,:,:) = dS6dS(difi,difi,:,:) + dsds(difi,i(difi),:,:) + ...
                                  dsds(i(difi),difi,:,:) + dsds(difi,difi,:,:);
dS6dS(j,difi,:,:) = dS6dS(j,difi,:,:) + dsds(i(j),difi,:,:);
dS6dS(difi,j,:,:) = dS6dS(difi,j,:,:) + dsds(difi,i(j),:,:);

dm6dp = dmdp(i,:);
dS6dp = dsdp(i,i,:);
dS6dp(difi,difi,:) = dS6dp(difi,difi,:) + dsdp(difi,i(difi),:) + ...
                                     dsdp(i(difi),difi,:) + dsdp(difi,difi,:);
dS6dp(j,difi,:,:) = dS6dp(j,difi,:,:) + dsdp(i(j),difi,:,:);
dS6dp(difi,j,:,:) = dS6dp(difi,j,:,:) + dsdp(difi,i(j),:,:);


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
if length(size(x))==2 & size(x,2)==1
  sx = 1;
else
  sx = length(size(x));
end
