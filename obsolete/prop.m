function [m6, S6] = prop(mu, Sigma, plant, dynmodel, policy)

% Propagate the distribution of the state one time step forward.
%
% Copyright (C) 2008-2011 by Carl Edward Rasmussen, Marc Deisenroth,
%                     Henrik Ohlsson, Andrew McHutchon and Joe Hall, 2011-12-06

angi = plant.angi; poli = plant.poli; dyni = plant.dyni; difi = plant.difi;

D1 = length(mu); D0 = D1/2; D2 = D1 + D0; D3 = D2 + D0; 
D4 = D3 + 2*length(angi); D5 = D4 + 2*length(angi); D6 = D5 + 2*length(angi);
D7 = D6 + length(plant.maxU); D8 = D7 + 2*length(plant.maxU); 
D9 = D8 + D0; Da = D9 + D0;

m = zeros(Da,1); m(1:D1) = mu; S = zeros(Da); S(1:D1,1:D1) = Sigma; % init m, S
Ss = S(1:D0,1:D0); Sr = S(D0+1:D1,D0+1:D1); Sc = S(1:D0,D0+1:D1);
Sn = diag(exp(2*dynmodel.hyp(end,:))/2); Zr = Sr/(Sr+Sn); Zn = Sn/(Sr+Sn);

% 1) add the belief variable
m(D1+1:D2) = Zr*m(1:D0) + Zn*m(D0+1:D1);                                 % mean
S(D1+1:D2,D1+1:D2) = Zn*Sr*Zn'+Zr*(Ss+Sn)*Zr'+Zn*Sc'*Zr'+Zr*Sc*Zn';  % variance
q = Zr*Ss + Zn*Sc'; S(D1+1:D2,1:D0) = q; S(1:D0,D1+1:D2) = q';     % cov with s
q = Zr*Sc + Zn*Sr; S(D1+1:D2,D0+1:D1) = q; S(D0+1:D1,D1+1:D2) = q';    % with r

% 2) add ctrl input variable
m(D2+1:D3) = m(D1+1:D2);                                                 % mean
S(D2+1:D3,D2+1:D3) = Zr*(Ss+Sn)*Zr';                                 % variance
q = Zr*Ss; S(D2+1:D3,1:D0) = q; S(1:D0,D2+1:D3) = q';              % cov with s
q = Zr*Sc; S(D2+1:D3,D0+1:D1) = q; S(D0+1:D1,D2+1:D3) = q';        % cov with r
q = Zr*(Ss+Sn)*Zr'; S(D2+1:D3,D1+1:D2) = q; S(D1+1:D2,D2+1:D3) = q';   % with b

% 3) augment with three sets of angles
[m(1:D6), S(1:D6,1:D6)] = trigaug(m(1:D3), S(1:D3,1:D3), ...
                                                       [angi D1+angi D2+angi]);

% 4) compute distribution of unsquashed control signal
i = [D2+poli+(D5-D3)*(poli>D0)]; j = [1:D2 D3+1:D5];
[m(D6+1:D7), S(D6+1:D7,D6+1:D7), C] = policy.fcn(policy, m(i), S(i,i));
q = S(j,i)*C; S(j,D6+1:D7) = q; S(D6+1:D7,j) = q'; 

% 5) squash control signal
[m(1:D8), S(1:D8,1:D8)] = trigaug(m(1:D7), S(1:D7,1:D7), D6+1:D7, plant.maxU);

% 6) compute new states
i = [dyni+(D3-D0)*(dyni>D0) D7+1:2:D8];
[m(D8+1:D9), S(D8+1:D9,D8+1:D9), C] = dynmodel.fcn(dynmodel, m(i), S(i,i));
q = S(1:D1,i)*C; S(1:D1,D8+1:D9) = q; S(D8+1:D9,1:D1) = q';

j = [D1+dyni+(D4-D2)*(dyni>D0) D7+1:2:D8];
[m(D9+1:Da), S(D9+1:Da,D9+1:Da), D] = dynmodel.fcn(dynmodel, m(j), S(j,j));
q = S(1:D1,j)*D; S(1:D1,D9+1:Da) = q; S(D9+1:Da,1:D1) = q';

q = C'*S(i,j)*D; S(D8+1:D9,D9+1:Da) = q; S(D9+1:Da,D8+1:D9) = q';

%m6 = m(Da-19:Da); S6 = [];
%return

% 7) compute the distribution of the next state
i = D8+1:Da; difi = [difi D0+difi]; j = setdiff(1:D1,difi);
m6 = m(i); m6(difi) = m6(difi) + m(difi);
S6 = S(i,i); 
S6(difi,difi) = S6(difi,difi) + S(difi,i(difi)) + S(i(difi),difi) + ...
                                                                  S(difi,difi);
S6(j,difi) = S6(j,difi) + S(i(j),difi); S6(difi,j) = S6(difi,j) + S(difi,i(j));
