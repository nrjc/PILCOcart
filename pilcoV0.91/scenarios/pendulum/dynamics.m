function dz = dynamics(t,z,u)

% Simulate the pendulum
%
% state is given by [dtheta,theta]
%
% dtheta:   [rad/s] angular velocity
% theta:    [rad]   angle
% u:        [Nm]    motor torque

l = 1;  % [m]      length of pendulum
m = 1;  % [kg]     mass of pendulum
g = 9.82; % [m/s^2]  acceleration of gravity

dz = zeros(2,1);
dz(1) = ( u(t) - m*g*l*sin(z(2))/2 ) / (m*l^2/3);
dz(2) = z(1);