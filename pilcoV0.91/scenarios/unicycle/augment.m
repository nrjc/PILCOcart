function r = augment(s)
rw = 0.225;                               % wheel radius in meters
s = s(:)';
r(1) = rw*cos(s(15))*s(7); r(2) = rw*sin(s(15))*s(7);
A = -[cos(s(15)) sin(s(15)); -sin(s(15)) cos(s(15))];
dA = -s(6)*[-sin(s(15)) cos(s(15)); -cos(s(15)) -sin(s(15))];
r(3:4) = A*r(1:2)' + dA*s(10:11)'; r(5:6) = A*s(10:11)'; 

