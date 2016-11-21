function [dyn, ddyn] = preComp(dyn)
% function to calculate a bunch or precomputable matrices and their derivatives
%
% matricies calculated are:
%   Kclean  N x N x E  pseudo inputs covariance matrix
%   K       N x N x E  K clean + process noise on the diagonal
%   R       N x N x E  cholesky of K
%   iK      N x N x E  inverse of noisy covariance matrix
%   XmXh2    N^2 x D   [(X - X')/2]^2
%   XpXh     N^2 x D   (X + X')/2
%
% derivatives calculated are:
%   Kdll            derivative of K w.r.t. log lengthscales
%   iKdll           derivative of iK w.r.t. log lengthscales
%   iKdlpn          derivative of iK w.r.t. log process noise std dev
%   iKdlsf          derivative of iK w.r.t. log signal std dev

x = dyn.inputs; [N, D] = size(x); E = size(dyn.hyp,2);
h = dyn.hyp; iell = exp(-2*[h.l]); 

% Initialisations
iK = zeros(N,N,E); dyn.K = iK; dyn.R = iK; iKdlpn = zeros(N^2,E); 
iKdlsf = zeros(N^2,E); 

% calculate the noise-free training covariance matrix
XmX = bsxfun(@minus,permute(x,[1,3,2]),permute(x,[3,1,2]));       % N-by-N-by-D
XmX2 = reshape(XmX.^2,N^2,D);
K = exp(bsxfun(@minus,2*[h.s],0.5*XmX2*iell));
K = reshape(K,N,N,E); dyn.Kclean = K;

for i=1:E;                                       % calculate noisy K, R, and iK
  K(:,:,i) = K(:,:,i) + (exp(2*h(i).n)+1e-9)*eye(N);
  if isfield(dyn,'nigp'); K(:,:,i) = K(:,:,i) + dyn.nigp(:,:,i); end
  dyn.R(:,:,i) = chol(K(:,:,i));
  iK(:,:,i) = K(:,:,i)\eye(N);
end
dyn.K = K; dyn.iK = iK; 

if nargout > 1                                        % derivative calculations
  XmX2iell = bsxfun(@times,XmX2,permute(iell,[3,1,2]));         % N^2-by-D-by-E
  Kdll = bsxfun(@times,reshape(XmX2iell,N,N,D,E),...         % N-by-N-by-D-by-E
                                                     permute(dyn.K,[1,2,4,3]));  

  iKdll = ...                                                % N-by-N-by-D-by-E
           -etprod('1234',etprod('1234',iK,'154',Kdll,'5234'),'1534',iK,'524');
  iKdll = reshape(iKdll,N^2,D,E);

  for i=1:E;
    iKdlpn(:,i) = -2*reshape(exp(2*h(i).n)*iK(:,:,i)*iK(:,:,i),N^2,1);
    iKdlsf(:,i) = -2*reshape(iK(:,:,i)*dyn.Kclean(:,:,i)*iK(:,:,i),N^2,1);
  end

  ddyn.Kdll = Kdll; ddyn.iKdll = iKdll; ddyn.iKdlpn = iKdlpn;
  ddyn.iKdlsf = iKdlsf;
end