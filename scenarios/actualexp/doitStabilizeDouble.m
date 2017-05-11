% Cart and pole swingup task.
%

%  1  1  ooou        the oldest value of u
%  2  2  oox         even older cart position
%  3  3  ootheta1    even older angle of the inner pendulum
%  4  4  ootheta2    even older angle of the outer pendulum
%  5  5  oou         even older value of u
%  6  6  ox          old cart position
%  7  7  otheta1     old angle of the inner pendulum
%  8  8  otheta2     old angle of the outer pendulum
%  9  9  ou          old value of u
%COMMENTED OUT:  10    dx          verlocity of cart
%COMMENTED OUT:  11    dtheta1     angular velocity of inner pendulum
%COMMENTED OUT:  12    dtheta2     angular velocity of outer pendulum
%  13 10 x           cart position
%  14 11 theta1      angle of the inner pendulum
%  14 12 theta2      angle of the outer pendulum
%  15    u           force applied to cart
%
% Copyright (C) by Carl Edward Rasmussen, Rowan McAllister 2015-02-17

clear
close all
format short; format compact;

basename = 'swingupDMK3Expf';
varNames = {'x','theta1','theta2'};

try
  rd = '../../';
  addpath([rd 'base'],[rd 'util'],[rd 'util/tprod'],[rd 'gp'],...
    [rd 'control'],[rd 'loss'],[rd 'direct'],[rd 'test']);
catch
end
rng(3);

D = 12;
E = 3;

mu0 = [0 0 pi pi 0 0 pi pi 0 0 pi pi]';                                 % initial state mean

S0 = diag([1e-9 0.125 0.0071 0.0071 1e-9 0.125 0.0071 0.0071 1e-9 0.125 0.0071 0.0071].^2);

%x position
S0(2,6)=(0.12499)^2; S0(6,2)=S0(2,6); %0.0671 on dx
S0(6,10)=(0.12499)^2; S0(10,6)=S0(6,10); 
S0(2,10) = S0(2,6)/S0(2,2) * S0(2,6);
S0(10,2) = S0(2,10);

%angle of inner pendulum
S0(3,7)=(0.0060)^2; S0(7,3)=S0(3,7); % 0.16 on dtheta
S0(7,11)=(0.0060)^2; S0(11,7)=S0(7,11);
S0(3,11) = S0(3,7)/S0(3,3) * S0(3,7);
S0(11,3) = S0(3,11);

%angle of outer pendulum
S0(4,8)=(0.0060)^2; S0(8,4)=S0(4,8); % 0.16 on dtheta
S0(8,12)=(0.0060)^2; S0(12,8)=S0(8,12);
S0(4,12) = S0(4,8)/S0(4,4) * S0(4,8);
S0(12,4) = S0(4,12);

s = struct('m',mu0,'s',S0);

plant.dyno = [10:12];
plant.odei = [10:14]; %fake variable

poli = [1 2 5 6 9 10 13:24];
angi = [3 4 7 8 11 12];
H=30;

N = 10; K = 1; J = 1; U = 1; maxU = 10;
cost = Cost(D);


dyn = gpa(D+U, E, angi, 'vfe'); % 8 inputs, 6 outputs, and var number 2, 3 are angles
dyn.induce = zeros(300, 0, E);                % use 300 shared inducing inputs
dyn.opt = struct('length',-300,'verbosity',3,'method','BFGS','fh',6); 

%policy.fcn = @(policy,m,s)conCat(@congp,@gSat,policy,m,s);
policy.maxU = maxU;                                 % max. amplitude of control
%nc = 50;
%mm = trigaug(s.m, zeros(length(s.m)), angi);
%policy.p.inputs = gaussian(mm(poli), eye(length(poli)), nc)';
%policy.p.target = 0.1*randn(nc, U);
%policy.p.hyp = log([ones(1,6) ones(1,12)*0.7 1 0.01])';

policy.opt = struct('length',-100,'MFEPLS',20,'method','BFGS', 'verbosity',3);
policy.p.w = 0*randn(U, numel(poli));
policy.p.b = 0*randn(U, 1);

policy.fcn = @(policy,m,s)conCat(@conlin,@gSat,policy,m,s);

ctrl = CtrlNF(D,E,policy,angi,poli);

j=0;
getRunTrialDMK3

for j = 1:169
  trainDirect(dyn, data, [1:12], [10:12], j<144);
  %disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
  %          ['observation noise|process noise std|inducing targets'], '%0.5f');
  ctrl.set_dynmodel(dyn);                    % for CtrlBF. No effect for CtrlNF
  learnPolicy;

  filename = [basename num2str(j) '_H' num2str(H)]; save(filename);
  setRunTrialDMK3(ctrl)
  getRunTrialDMK3


  disp(['controlled trial # ' num2str(j)]);
end


%trainDirectS(dyns, data, [1:12], [10:12], j<44);