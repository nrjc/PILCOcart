function [M, S] = getPlotDistr(m, s, ell1, ell2)

% augment input distribution
[m1 s1] = trigaug(m, s, [3 4], [ell1, ell2]);

% mean
M = [-m1(5)-m1(7); m1(6) + m1(8)];

% put covariance matrix together
s11 = s1(5,5) + s1(7,7) + s1(5,7) + s1(7,5); % ell2 sin(th2) + ell3 sin(th3)
s22 = s1(6,6) + s1(8,8) + s1(6,8) + s1(8,6); % ell2 cos(th2) + ell3 cos(th3)

%   cov(-ell2\sin(th2), ell2\cos(th2))
% + cov(-ell2\sin(th2), ell3\cos(th3))
% + cov(-ell3\sin(th3), ell2\cos(th2))
% + cov(-ell3\sin(th3), ell3\cos(th3))


s12 = -(s1(5,6)+s1(5,8)+s1(7,6)+s1(7,8)); 

S = [s11 s12; s12' s22];
try
  chol(S);
catch
  disp('matrix S not pos.def. (getPlotDistr)');
end