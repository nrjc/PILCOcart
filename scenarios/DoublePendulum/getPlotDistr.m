function [M1, S1, M2, S2] = getPlotDistr(m, s, ell1, ell2)

% augment input distribution
[m1 s1] = trigaug(m, s, [3 4], [ell1, ell2]);

% mean
M1 = [-m1(5)-m1(7); m1(6) + m1(8)];
M2 = [-m1(5); m1(6)];

% put covariance matrix together (outer pendulum)
s11 = s1(5,5) + s1(7,7) + s1(5,7) + s1(7,5); % ell2 sin(th2) + ell3 sin(th3)
s22 = s1(6,6) + s1(8,8) + s1(6,8) + s1(8,6); % ell2 cos(th2) + ell3 cos(th3)
s12 = -(s1(5,6)+s1(5,8)+s1(7,6)+s1(7,8)); 
S1 = [s11 s12; s12' s22];
%   cov(-ell2\sin(th2), ell2\cos(th2))
% + cov(-ell2\sin(th2), ell3\cos(th3))
% + cov(-ell3\sin(th3), ell2\cos(th2))
% + cov(-ell3\sin(th3), ell3\cos(th3))

% put covariance matrix together (inner pendulum)
s11 = s1(5,5);
s12 = s1(5,6);
s22 = s1(6,6);
S2 = [s11 s12; s12' s22];

try
  chol(S1);
catch
  disp('matrix S1 not pos.def. (getPlotDistr)');
end


try
  chol(S2);
catch
  disp('matrix S2 not pos.def. (getPlotDistr)');
end