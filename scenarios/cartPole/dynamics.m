function dz = dynamics(t, z, f)

% Ordinary differential equantion (ode) describing the dynamics of the "cart
% and pole" system. Returns the time derivatives of the state variables, or
% if only two input arguments are given, the total mechanical energy.
%
% t       [s]      time
% z                current state vector, where
%   z(1)  [m]      horizontal position of the cart
%   z(2)  [rad]    angle of the pendulum
%   z(3)  [m/s]    horizontal velocity of cart
%   z(4)  [rad/s]  angular velocity of the pendulum
% f       [N]      function of time, the horizontal force applied to the cart
%
% Copyright (C) 2008-2013 by Carl Edward Rasmussen, 2013-12-12.

l = 0.5;  % [m]      length of pendulum
m2 = 0.5;  % [kg]     mass of pendulum
m1 = 0.5;  % [kg]     mass of cart
b = 0.1;  % [Ns/m]   coefficient of friction between cart and ground
g = 9.82; % [m/s^2]  acceleration of gravity

if nargin==3
  dz = zeros(4,1);
  dz(1) = z(3);
  dz(2) = z(4);
  dz(3) = ( -2*m2*l*z(4)^2*sin(z(2)) + 3*m2*g*sin(z(2))*cos(z(2)) ...
                        + 4*f(t) - 4*b*z(3) )/( 4*(m1+m2)-3*m2*cos(z(2))^2 );
  dz(4) = (-3*m2*l*z(4)^2*sin(z(2))*cos(z(2)) + 6*(m1+m2)*g*sin(z(2)) ...
            + 6*(f(t)-b*z(3))*cos(z(2)) )/( 4*l*(m2+m1)-3*m2*l*cos(z(2))^2 );
else
  dz = (m1+m2)*z(3)^2/2 + 1/6*m2*l^2*z(4)^2 + m2*l*(g-z(3)*z(4))*cos(z(2))/2;
end
