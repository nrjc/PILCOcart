function dz = dynamics(t,z,f)

% Simulate the pendubot dynamics (where the input torque is applied to the
% base link). Either compute state derivatives, or if no action is given,
% report the total mechanical energy.
%
% Copyright (C) 2008 by Carl Edward Rasmussen and Marc Deisenroth,
% 2009-01-30.
%
% The state is given by [dtheta1, dtheta2, theta1, theta2] (angles are the
% two last coordinates)

m1 = 0.5;  % [kg]     mass of 1st link
m2 = 0.5;  % [kg]     mass of 2nd link
b1 = 0.0;  % [N/m/s]  coefficient of friction (1st joint)
b2 = 0.0;  % [N/m/s]  coefficient of friction (2nd joint)
l1 = 0.5;  % [m]      length of 1st pendulum
l2 = 0.5;  % [m]      length of 2nd pendulum
g  = 9.82; % [m/s^2]  acceleration of gravity
I1 = m1*l1^2/12;  % moment of inertia around pendulum midpoint (1st link)
I2 = m2*l2^2/12;  % moment of inertia around pendulum midpoint (2nd link)

if nargin == 3

  A = [l1^2*(0.25*m1+m2) + I1,      0.5*m2*l1*l2*cos(z(3)-z(4));
       0.5*m2*l1*l2*cos(z(3)-z(4)), l2^2*0.25*m2 + I2          ];
  b = [g*l1*sin(z(3))*(0.5*m1+m2) - 0.5*m2*l1*l2*z(2)^2*sin(z(3)-z(4))...
                                                        + f(t) - b1*z(1);
       0.5*m2*l2*( l1*z(1)^2*sin(z(3)-z(4)) + g*sin(z(4)) )    - b2*z(2)];
  x = A\b;

  dz = zeros(4,1);
  dz(1) = x(1);
  dz(2) = x(2);
  dz(3) = z(1);
  dz(4) = z(2);

else
  dz = m1*l1^2*z(1)^2/8 + I1*z(1)^2/2 + m2/2*(l1^2*z(1)^2 ...
    + l2^2*z(2)^2/4 + l1*l2*z(1)*z(2)*cos(z(3)-z(4))) + I2*z(2)^2/2 ...
    + m1*g*l1*cos(z(3))/2 + m2*g*(l1*cos(z(3))+l2*cos(z(4))/2);
end