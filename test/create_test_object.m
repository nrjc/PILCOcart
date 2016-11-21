function object = create_test_object(type,varargin)

% CREATE_TEST_OBJECT is a utility for all other test scripts to generate random
% objects they might require in order to run a particular test. This function
% is commonly used for a test script's default inputs.
%
% GENERAL CALL:
%   object = create_test_object(type,varargin)
%
% EXAMPLE CALLS:
%   plant  = create_test_object('plant' , 'easy'     )
%   dyn    = create_test_object('dyn'   , plant      )
%   ctrlnf = create_test_object('CtrlNF', plant      )
%   ctrlbf = create_test_object('CtrlBF', plant, dyn )
%   s      = create_test_object('state' , plant, ctrl)
%   cost   = create_test_object('cost'  , plant      )
%   expl   = create_test_object('expl')
%
% Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2016-05-10

% SEED = 1;
% rng(SEED);

switch type
  case 'plant'
    object = create_plant(varargin{:});
  case 'dyn'
    object = create_dyn(varargin{:});
  case {'CtrlNF','CtrlBF'}
    object = create_ctrl(type,varargin{:});
  case 'state'
    object = create_state(varargin{:});
  case 'cost'
    object = create_cost(varargin{:});
  case 'expl'
    object = create_expl();
  otherwise
    error([mfile, ': invalid input.'])
end


function plant = create_plant(complexity)

% 1st indices are for the dynamics models
% 2nd indices are for the policy

switch complexity
  case 'easiest'
    %D  1 1  x          position
    %Dd 2    u          action
    D = 1; E = 1; U = 1;     % state dim, dynmodel output dim, action dim
    angi = [];               % angle variables
    poli = 1;              % variables that serve as inputs to the policy
    
  case 'easy'
    %   1 1  old u      old action
    %D  2 2  x          position
    %Dd 3    u          action
    D = 2; E = 1; U = 1;     % state dim, dynmodel output dim, action dim
    angi = [];               % angle variables
    poli = [1 2];            % variables that serve as inputs to the policy
    
  case 'medium'
    %   1 1  old u      old action
    %D  2 2  theta      angle
    %Dd 3    u          action
    %   4 3  sin(theta)
    %d  5 4  cos(theta)
    D = 2; E = 1; U = 1;     % state dim, dynmodel output dim, action dim
    angi = 2;                % angle variables
    poli = [1   3 4];        % variables that serve as inputs to the policy
    
  case 'medium1'
    %   1 1  old u      old action
    %   2 2  x1
    %D  3 3  x2
    %Dd 4    u          action
    D = 3; E = 2; U = 1;     % state dim, dynmodel output dim, action dim
    angi = [];               % angle variables
    poli = [1 2 3];          % variables that serve as inputs to the policy
    
  case 'medium2'
    %   1 1  old u      old action
    %   2 2  x          position
    %D  3 3  theta      angle
    %Dd 4    u          action
    %   5 4  sin(theta)
    %d  6 5  cos(theta)
    D = 3; E = 2; U = 1;     % state dim, dynmodel output dim, action dim
    angi = 3;                % angle variables
    poli = [1 2   4 5];      % variables that serve as inputs to the policy
    
  case 'medium3'
    %   1 1  old u1     old action 1
    %   2 2  old u2     old action 2
    %D  3 3  x1         position 1
    %   4    u1         action 1
    %Dd 5    u2         action 2
    D = 3; E = 1; U = 2;     % state dim, dynmodel output dim, action dim
    angi = [];                % angle variables
    poli = [1 2 3];  % variables that serve as inputs to the policy
    
  case 'hard'
    %   1 1  old u1     old action 1
    %   2 2  old u2     old action 2
    %   3 3  x1         position 1
    %   4 4  theta      angle
    %D  5 5  x2         position 2
    %   6    u1         action 1
    %Dd 7    u2         action 2
    %   8 6  sin(theta)
    %d  9 7  cos(theta)
    D = 5; E = 3; U = 2;     % state dim, dynmodel output dim, action dim
    angi = 4;                % angle variables
    poli = [1 2 3   5 6 7];  % variables that serve as inputs to the policy
    
  case 'cartPole'
    %   1  1  oldu      old value of u
    %   2  2  x         cart position
    %   3  3  theta     angle of the pendulum
    %   4  4  v         cart velocity
    %D  5  5  dtheta    angular velocity
    %Dd 6     u         force applied to cart
    %   7  6  sin(theta)
    %d  8  7  cos(theta)
    D = 5; E = 4; U = 1;     % state dim, dynmodel output dim, action dim
    angi = 3;                % angle variables
    poli = [1 2   4 5 6 7];  % variables that serve as inputs to the policy
    
  case 'cartPoleMarkov'
    %  1  1  oou        even older value of u
    %  2  2  ox         old cart position
    %  3  3  otheta     old angle of the pendulum
    %  4  4  ou         old value of u
    %  5  5  x          cart position
    %  6  6  theta      angle of the pendulum
    %  7     v          cart velocity
    %  8     dtheta     angular velocity
    %  9     u          force applied to cart
    D = 6; E = 2; U = 1;           % state dim, dynmodel output dim, action dim
    angi = [3 6];                  % angle variables
    poli = [1 2   4 5   7 8 9 10]; % vars that serve as inputs to the policy
    
  case 'cartDoublePendulum'
    %  1   1  oldu        old value of u
    %  2   2  dx          Verlocity of cart
    %  3   3  dtheta1     angular velocity of inner pendulum
    %  4   4  dtheta2     angular velocity of outer pendulum
    %  5   5  x           Position of cart
    %  6   6  theta1      angle of inner pendulum
    %  7   7  theta2      angle of outer pendulum
    %  8      u           Force on Cart
    %  9   8  sin(theta1)
    % 10   9  cos(theta1)
    % 11  10  sin(theta2)
    % 12  11  cos(theta2)
    D = 7; E = 6; U = 1;           % state dim, dynmodel output dim, action dim
    angi = [6 7];                  % angle variables
    poli = [1 2 3 4 5  8 9 10 11]; % vars that serve as inputs to the policy
    
  case 'cartDoublePendulumMarkov'
    %  1   1  oou         even older value of u
    %  2   2  ox          old position of cart
    %  3   3  otheta1     old angle of inner pendulum
    %  4   4  otheta2     old angle of outer pendulum
    %  5   5  ou          old value of u
    %  6   6  x           position of cart
    %  7   7  theta1      angle of inner pendulum
    %  8   8  theta2      angle of outer pendulum
    %  9      dx          verlocity of cart
    %  10     dtheta1     angular velocity of inner pendulum
    %  11     dtheta2     angular velocity of outer pendulum
    %  12     u           force applied to cart
    D = 8; E = 3; U = 1;           % state dim, dynmodel output dim, action dim
    angi = [3 4 7 8];                  % angle variables
    poli = [1 2 5 6 9 10 11 12 13 14 15 16]; % vars that serve as inputs to the policy
    
  case 'unicycle'
    %  1   1  dtheta  roll angular velocity
    %  2   2  dphi    yaw angular velocity
    %  3   3  dpsiw   wheel angular velocity
    %  4   4  dpsif   pitch angular velocity
    %  5   5  dpsit   turn table angular velocity
    %  6   6  xc      x position of origin (self centered coordinates)
    %  7   7  yc      y position of origin (self centered coordinates)
    %  8   8  theta   roll angle
    %  9   9  phi     yaw angle
    % 10  10  psif    pitch angle
    % 11      dx      x velocity
    % 12      dy      y velocity
    % 13      dxc     x velocity of origin (self centered coordinates)
    % 14      dyc     y velocity of origin (self centered coordinates)
    % 15      x       x position
    % 16      y       y position
    % 17      psiw    wheel angle
    % 18      psit    turn table angle
    % 19      ct      control torque for turn table
    % 20      cw      control torque for wheel
    D = 10; E = 10; U = 2;         % state dim, dynmodel output dim, action dim
    angi = [];                     % angle variables
    poli = 1:D;                    % vars that serve as inputs to the policy
  otherwise
    error([mfilename,': invalid experiment name.'])
end

pE = E;
di = D + U;                 % dynmodel input dim (without angles)
dia = di + 2*length(angi);  % dynmodel input dim (with angles)

plant.type = complexity;
plant.angi = angi;
plant.D    = D;
plant.di   = di;
plant.dia  = dia;
plant.E    = E;
plant.pE   = pE;
plant.poli = poli;
plant.U    = U;


%% DYN ------------------------------------------------------------------------
function dyn = create_dyn(plant)

angi = plant.angi;
di   = plant.di;
dia  = plant.dia;
E    = plant.E;
pE   = plant.pE;

N = 10;                   % number training points
x = 1+2*randn(N,dia,pE); hm = 0.5*randn(1,dia,E); hb = 3*randn(1,E);
sf2 = 0.8+1*rand(1,1,E); ell = 0.5*rand(1,1,E,dia); sn = 0.05+0.3*rand(E,1);
my = bsxfun(@plus,sum(bsxfun(@times,x,hm),2),hb); % mean function of inputs
dyn = gpa(di,E,angi);
dyn.inputs = x; dyn.target = nan(N,E); dyn.beta = nan(N,E);
for e=1:E;
  hyp.l = log(unwrap(ell(1,1,e,:)));
  hyp.s = log(sqrt(sf2(1,1,e)));
  hyp.n = log(sn(e));
  hyp.m = unwrap(hm(1,:,e));
  hyp.b = hb(e);
  hyps(e) = hyp; %#ok<AGROW>
  
  z = bsxfun(@times, x(:,:,min(e,pE)), exp(-hyp.l')); % scale inputs
  K = exp(2*hyp.s-maha(z,z)/2) + exp(2*hyp.n)*eye(N);
  L = chol(K)';
  
  dyn.target(:,e)= L*randn(N,1) + my(:,e) + sn(e)*randn(N,1);
  dyn.W(:,:,e) = L'\(L\eye(N));
  dyn.beta(:,e) = dyn.W(:,:,e) * (dyn.target(:,e) - my(:,e));
end
dyn.hyp = hyps;
dyn.on = [dyn.hyp.n]' - log(2)/2; % log(rand(E,1)/10);
dyn.pn = [dyn.hyp.n]' - log(10); % log(rand(E,1)/10);


%% CTRL -----------------------------------------------------------------------
function ctrl = create_ctrl(ctrl_class, plant, dyn)

angi = plant.angi;
D    = plant.D;
E    = plant.E;
poli = plant.poli;
U    = plant.U;

policy.type = 'conlin';
policy.maxU = 20*ones(1,U);
policy.p.w = 1e-2*randn(U, length(poli));
policy.p.b = 1e-2*randn(U, 1);
policy.fcn = @(policy,m,s)conCat(@conlin,@gSat7,policy,m,s);
ctrl = feval(ctrl_class,D,E,policy,angi,poli);

if exist('dyn','var')
  ctrl.set_dynmodel(dyn);
  ctrl.set_on(diag(exp(2*dyn.on)));
else
  log_on = log(rand(E,1)/10);
  ctrl.set_on(diag(exp(2*log_on)));
end


%% STATE ----------------------------------------------------------------------
function s = create_state(plant, ctrl)

D    = plant.D;
s.m = 0.1*randn(D,1);
s.s = 0.1*randn(D);
s.s = s.s'*s.s;

if exist('ctrl','var')
  s = ctrl.reset_filter(s);
  assert(ctrl.F == length(s.m));
  if isa(ctrl,'CtrlBF')
    s.v = 2.0*ctrl.on;
    s.reset = false;
  end
end

assert(all(eig(s.s) >= -1e-12));


%% COST -----------------------------------------------------------------------
function cost = create_cost(plant)
D = plant.D;
cost = Cost(D);


%% EXPL -----------------------------------------------------------------------
function expl = create_expl()
expl.method = 'uncertaintyReductionGivenMarginalSimulation';
expl.fcn = @exploreUCB;
expl.beta = 0.5;
expl.ccs_cov = false;
