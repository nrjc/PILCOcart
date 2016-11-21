function [x, y] = cartSpace(l, angles)
%CARTSPACE Convert shoulder and elbow angles to cartesian coords of hand
%   Origin is at shoulder
x = l(1) * cos(angles(:,1)) + l(2) * cos(angles(:,1) + angles(:,2));
y = l(1) * sin(angles(:,1)) + l(2) * sin(angles(:,1) + angles(:,2));

end