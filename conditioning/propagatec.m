function [mj, Sj, dynmodel] = propagatec(mu, Sigma, plant, dynmodel, policy)

% Propagate the state distribution one time step forward.
%
% Copyright (C) 2008-2011 by Marc Deisenroth, Carl Edward Rasmussen, Henrik
% Ohlsson, Andrew McHuthon and Joe Hall, 2011-11-29

angi = plant.angi; poli = plant.poli; dyni = plant.dyni; difi = plant.difi;

Di = length(dyni); Do = length(plant.dyno);
D0 = length(mu);                          % size of joint distribution, Di + Do
D1 = D0 + 2*length(angi);          % length after mapping all angles to sin/cos
D2 = D1 + length(plant.maxU);   % length after computing unsquashed ctrl signal
D3 = D2 + length(plant.maxU);            % length after squashing ctrl signal
D4 = D3 + Do;                                         % length after predicting
m = zeros(D4,1); m(1:D0) = mu; S = zeros(D4); S(1:D0,1:D0) = Sigma; % init m, S
Dold = D0 - Do;                 % on first time step this = 0, otherwise = dyni
angi = Dold + angi; poli = Dold + poli; dyni = Dold + dyni; difi = Dold + difi;

% 1) augment the state distribution with trigonometric functions
noise = zeros(D0); noise(Dold+1:D0,Dold+1:D0) = diag(exp(2*dynmodel.hyp(end,:))/2);
[mm, SS] = trigaug(m(1:D0), S(1:D0,1:D0) + noise, angi);
[m(1:D1), S(1:D1,1:D1)] = trigaug(m(1:D0), S(1:D0,1:D0), angi);

% 2) compute the distribution of the unsquashed control signal
[m(D1+1:D2), S(D1+1:D2,D1+1:D2), C] = policy.fcn(policy, mm(poli), SS(poli,poli));
q = S(1:D1,poli)*C; S(1:D1,D1+1:D2) = q; S(D1+1:D2,1:D1) = q'; Cu = C;

% 3) squash the control signal
[m(D2+1:D3) S(D2+1:D3,D2+1:D3) C] = gSin(m(1:D2), S(1:D2,1:D2), D1+1:D2, plant.maxU);
q = S(1:D2,1:D2)*C; S(1:D2,D2+1:D3) = q; S(D2+1:D3,1:D2) = q'; Csu = C;

% 4) compute the distribution of the change in state
i = [dyni D2+1:D3];                                    % adds indices of ctrl
[m(D3+1:D4), S(D3+1:D4,D3+1:D4), C] = dynmodel.fcn(dynmodel, m(i), S(i,i));
q = S(1:D3,i)*C; S(1:D3,D3+1:D4) = q; S(D3+1:D4,1:D3) = q';

% Compute conditional term
[S dynmodel] = correctVar(m,S,dynmodel,plant,Cu,Csu,C);

% 5) compute the distribution of the next state
i = D3+1:D4; di = difi-Dold; nd = setdiff(1:Do,di); % di,nd relative to just current state
m6 = m(i); m6(di) = m6(di) + m(difi);
S6 = S(i,i); 
S6(di,di) = S6(di,di) + S(difi,i(di)) + S(i(di),difi) + S(difi,difi);
S6(nd,di) = S6(nd,di) + S(i(nd),difi); S6(di,nd) = S6(di,nd) + S(di,i(nd));

% Compute covariance between Sigma(dyni) and S6
i = [dyni D2+1:D3]; Css6 = zeros(length(i),Do);
Css6(:,nd) = S(i,D3+nd);                           % C(x, y), non-difi variables
Css6(:,di) = S(i,D3+di) + S(i,difi); % C([w; x], x+y) = [C(w,x); V(x)] + C([w; x],y)
mj = [m(i); m6]; Sj = [S(i,i) Css6; Css6' S6];

try chol(S6); catch; fprintf('S6 not pos. def.\n'); keyboard; end
if ~isreal(S6); fprintf('S6 not real\n'); keyboard; end
try chol(Sj); catch; fprintf('Sj not pos. def.\n'); keyboard; end

