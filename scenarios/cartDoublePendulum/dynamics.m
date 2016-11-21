function [dz, dfdx, dfdu] = dynamics(t,z,f)

% Simulate the cart and double pendulum dynamics. Either compute state
% derivatives, or if no action is given, report the total mechanical energy.
% For more then one output argument the linearisation around the
% upright position ist return dfdx and dfdu
%
% z                current state vector, where
%   z(1)  [m/s]    velocity of the cart
%   z(2)  [rad/s]  angular velocity of inner pendulum
%   z(3)  [rad/s]  angular velocity of outer pendulum
%   z(4)  [m]      horizontal position of the cart
%   z(5)  [rad]    angle of inner pendulum
%   z(6)  [rad]    angle of outer pendulum
%
% Copyright (C) 2008-2015 by Carl Edward Rasmussen and Marc Deisenroth and
%                                                        Jonas Umlauft 2015-03-23

m1 = 0.5;  % [kg]     mass of cart
m2 = 0.5;  % [kg]     mass of 1st pendulum
m3 = 0.5;  % [kg]     mass of 2nd pendulum
r  = 0.1;  % [N/m/s]  coefficient of friction between cart and ground
l2 = 0.6;  % [m]      length of 1st pendulum
l3 = 0.6;  % [m]      length of 2nd pendulum
g  = 9.82; % [m/s^2]  acceleration of gravity

if nargin == 3
  A = [2*(m1+m2+m3) -(m2+2*m3)*l2*cos(z(5)) -m3*l3*cos(z(6))
       -(3*m2+6*m3)*cos(z(5)) (2*m2+6*m3)*l2 3*m3*l3*cos(z(5)-z(6))
       -3*cos(z(6)) 3*l2*cos(z(5)-z(6)) 2*l3];
  b = [2*f(t)-2*r*z(1)-(m2+2*m3)*l2*z(2)^2*sin(z(5))-m3*l3*z(3)^2*sin(z(6))
       (3*m2+6*m3)*g*sin(z(5))-3*m3*l3*z(3)^2*sin(z(5)-z(6))
       3*l2*z(2)^2*sin(z(5)-z(6))+3*g*sin(z(6))];
  x = A\b;

  dz = zeros(6,1);
  dz(1) = x(1);
  dz(2) = x(2);
  dz(3) = x(3);
  dz(4) = z(1);
  dz(5) = z(2);
  dz(6) = z(3);
else
  dz = (m1+m2+m3)*z(5)^2/2+(m2/6+m3/2)*l2^2*z(6)^2+m3*l3^2*z(1)^2/6 ...
        -(m2/2+m3)*l2*z(5)*z(6)*cos(z(2))-m3*l3*z(5)*z(1)*cos(z(3))/2 ...
        +m3*l2*l3*z(6)*z(1) *cos(z(2)-z(3))/2+(m2/2+m3)*l2*g*cos(z(2)) ...
        +m3*l3*g*cos(z(3))/2;        
    
  % I2 = m2*l2^2/12;  % moment of inertia around pendulum midpoint (1st link)
  % I3 = m3*l3^2/12;  % moment of inertia around pendulum midpoint (2nd link)
  %
  % dz = m1*z(2)^2/2 + m2/2*(z(2)^2-l2*z(2)*z(3)*cos(z(5))) ...
  %     + m3/2*(z(2)^2 - 2*l2*z(2)*z(3)*cos(z(5)) - l3*z(2)*z(4)*cos(z(6))) ...
  %     + m2*l2^2*z(3)^2/8 + I2*z(3)^2/2 ...
  %     + m3/2*(l2^2*z(3)^2 + l3^2*z(4)^2/4 + l2*l3*z(3)*z(4)*cos(z(5)-z(6))) ...
  %     + I3*z(4)^2/2 ...
  %     + m2*g*l2*cos(z(5))/2 + m3*g*(l2*cos(z(5))+l3*cos(z(6))/2);
end

if nargout == 3
  dfdx = zeros(6);
  dfdu = zeros(6,1);
    
  q = 2*l2*l3*(m1*(4*m2+3*m3)+m2*(m2+m3));
    
  dfdx(1,5) = 3*l2*l3*(m2+2*m3)*(2*m2+m3)*g;
  dfdx(2,5) = 3*l3*(m2+2*m3)*(4*m1+4*m2+m3)*g;
  dfdx(3,5) = -9*l2*(m2+2*m3)*(2*m1+m2)*g;    
    
  dfdx(1,6) = -3*l2*l3*m2*m3*g;
  dfdx(2,6) = -9*l3*m3*(2*m1+m2)*g;
  dfdx(3,6) = 3*l2*(4*m1*(m2+3*m3)+m2*(m2-4*m3))*g;    
    
  dfdx(1,1) = -2*l2*l3*(4*m2+3*m3)*r;
  dfdx(2,1) = -6*l3*(2*m2+m3)*r;
  dfdx(3,1) = 6*l2*m2*r;
    
  dfdx = dfdx./q;
  dfdx(4:6,1:3) = eye(3);
    
  dfdu(1) = 2*l2*l3*(4*m2+3*m3);
  dfdu(2) = 6*l3*(2*m2+m3);
  dfdu(3) = -6*l2*m2;
  dfdu = dfdu./q;
end