function [d dy dh] = checkgrad(f, X, e, varargin)

% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the difference divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad('f', X, e, other, ...)
%
% where X is the argument and e is the small perturbation used for the finite
% differences, and "other, ..." are optional additional parameters which get
% passed to f. The function f should be of the type 
%
% [fX, dfX] = f(X, other, ...)
%
% where fX is the function value and dfX are the partial derivatives in the
% same format as X.
%
% Carl Edward Rasmussen and Andrew McHutchon, 2012-10-18

Z = unwrap(X); NZ = length(Z);                      % number of input variables
[y dy] = feval(f, X, varargin{:});             % get the partial derivatives dy
[D E] = size(y); y = y(:); Ny = length(y);         % number of output variables
if iscell(dy) || isstruct(dy); dy = unwrap(dy); end;
dy = reshape(dy,Ny,NZ);

dh = zeros(Ny,NZ);
for j = 1:NZ
  dx = zeros(length(Z),1);
  dx(j) = dx(j) + e;                               % perturb a single dimension
  y2 = feval(f, rewrap(X,Z+dx), varargin{:});                       %#ok<PFBNS>
  y1 = feval(f, rewrap(X,Z-dx), varargin{:});
  dh(:,j) = (y2(:) - y1(:))/(2*e);
end

d = sqrt(sum((dh-dy).^2,2)./sum((dh+dy).^2,2)); % norm of diff divided by norm of sum
small = max(abs([dy dh]),[],2) < 1e-5;                       % small derivatives 
d(d > 1e-3 & small) = NaN;              % are poorly tested by finite differences
d = reshape(d,D,E);

disp('Analytic(dy)  Numerical(dh)')
for i=1:Ny;
    disp([dy(i,:)' dh(i,:)']);                           % print the two vectors
    %fprintf('d = %e\n\n',d(i))
end 

if Ny > 1; disp('For all outputs, d = '); disp(d); end; fprintf('\n');
