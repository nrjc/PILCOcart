function dz = dynamics_stiffness(t,z,musc)
% DYNAMICS Simulate the movement of the arm
%   State is given by [theta, phi, dtheta, dphi]
%   
%   theta: angle of upper arm
%   phi:   angle of lower arm

global odeA odeFIELD odeF2 odeF3 fieldCentre;

% arm model parameters
%======================================================================
m(1)= 1.93;     % mass of upper arm [kg]
m(2)= 1.52;     % mass of forearm [kg]
l(1)= 0.29;     % length of upper arm [m]
l(2)= 0.34;     % length of forearm [m]
cl(1)=0.165;    % position of center of mass of upper arm [m]
cl(2)=0.19;     % position of center of mass of forearm [m]
I(1)=0.0141;    % Mass moment of inertia for upper arm [kgm^2]
I(2)=0.0188;    % Mass moment of inertia for forearm [kgm^2]
d1=0.03; d2=0.03; d3=0.021; d4=0.021; d5=0.044; d6=0.044; d7=0.0338; d8=0.0338; % moment arms
Jm=[d1 0; -d2 0; 0 d3; 0 -d4; d5 d7; -d6 -d8];

% PFM
%====================================================
PFMmass=[1.516 0;0 1.404];  
PFMdamp=[10.247 0;0 7.592];
statfric=[0.102;0.356];

% PILCO setup
musc = musc(t)+20;
ang = [1 2];
angv = [3 4];

Ja=jacobian(l,z(ang));        % Jacobian relating joint and Cartesian space
Jadot=jacobiandot(l,z(ang),z(angv));           % Derivative of the Jacobian
xv=Ja*z(angv);

% Components in F=ma calculation
Hm= mass(m,l,cl,I,z(ang));
C= coriolis(m,l,cl,z(ang),z(angv));

% Generally, max of mv will be ~0.15 = 0.021 * 7
mv = (Jm * z(angv))';                                   % muscle velocities
musc = musc - musc .* (odeA * abs(mv));               % modified muscle forces
JT = Jm'*musc';                                             % Joint torques

% Interaction with the environment
%======================================================================
if odeFIELD==1 %NF
    EXT=[0;0];
elseif odeFIELD==2         % Curl field
    [x,y] = cartSpace(l, z(ang)');
    x = x - fieldCentre(1); y = y - fieldCentre(2); % change of coords to centre of field
    dist = [-y; x];
    force = odeF2 * dist;
    EXT = Ja' * force;
elseif odeFIELD==3
    [x,~] = cartSpace(l, z(ang)');
    x = x - fieldCentre(1);                         % change of coords to centre of field
    force = odeF3 * [x; 0];
    EXT = Ja' * force;
end

%PFM dynamics
%===========================================================================
PFMdynX=PFMdamp*xv+tanh(diag(xv)*200)*statfric;
PFMdyn=Ja'*PFMdynX;
op=Ja'*PFMmass*Jadot*z(angv);

% motion integration
%==================================================================
qa=(Hm + Ja'*PFMmass*Ja)\(JT + EXT - C - PFMdyn - op);
% qv=z(angv)+qa*dt;
% q=z(ang)+z(angv)*dt+0.5*qa*dt*dt;
dz = [z(3) z(4) qa(1) qa(2)]';
end

