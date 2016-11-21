function [A B] = covSEard1D(hyp, x, z)

if nargin == 0; A = 'D*2'; return; end;      % report number of hypers

D = size(x,2); A = 0;
    
if nargin == 2                      % Calculate training covariance matrix
    for i = 1:D
        hypi = [hyp(i); hyp(D+i)];
        Ai = covSEard(hypi, x(:,i));
        A = A + Ai;
    end
    
elseif nargout == 2                 % Calculate test covariance matrices
    B = 0;
    for i = 1:D
        hypi = [hyp(i); hyp(D+i)];
        [Ai Bi] = covSEard(hypi, x(:,i), z(:,i));
        A = A + Ai; B = B + Bi;
    end
        
elseif nargin == 3                  % Derivatives of K w.r.t. hypers
    i = rem(z-1,D) + 1;             % 1 -> D
    hypi = [hyp(i); hyp(D+i)];
    clear covSEard;
    A = covSEard(hypi, x(:,i), ceil(z/D));
  
end
    
        


