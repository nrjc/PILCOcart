function r = augment(s)

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

rw = 0.225;                               % wheel radius in meters
s = s(:)';

dphi = s(2);
dpsiw = s(3);
phi = s(9);
xy = s(15:16)';

dx = rw*cos(phi)*dpsiw;
dy = rw*sin(phi)*dpsiw;
A = -[cos(phi) sin(phi); -sin(phi) cos(phi)];
dA = -dphi*[-sin(phi) cos(phi); -cos(phi) -sin(phi)];
dxcdyc = A*[dx;dy] + dA*xy;
xcyc = A*xy;

r = [dx;dy;dxcdyc;xcyc];
