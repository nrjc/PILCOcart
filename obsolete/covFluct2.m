function [C dynmodel dCdm dCds] = covFluct2(dynmodel, m, s)

% covFluct2: compute the covariance of the fluctuations under a posterior
% Gaussian process at two inputs with joint N(m, s), and its derivatives.
%
% Andrew McHutchon and Carl Edward Rasmussen, 2012-05-29

%try chol(s); catch; 
%  fprintf(['covFluct: input covariance matrix not pos def\n']); keyboard;
%end

D = size(m,1)/2; E = size(dynmodel.target,2); N = size(dynmodel.inputs,1); 
C = zeros(E); sf2 = exp(2*dynmodel.hyp(end-1,:));                      % 1-by-E
if nargout > 2, dCdm = zeros(E,E,2*D); dCds = zeros(E,E,2*D,2*D); end 

if ~isfield(dynmodel,'iK')                              % calculate beta and iK
  if isfield(dynmodel,'induce') && size(dynmodel.induce,2)
    dynmodel = preCalcFitc(dynmodel); iK = dynmodel.iK2; 
  else
    iK = zeros(N,N,E); dynmodel = preCalcDyn(dynmodel); 
    for k = 1:E; iK(:,:,k) = solve_chol(dynmodel.R(:,:,k), eye(N)); end
  end
  dynmodel.iK = iK;
end

for k = 1:E                                              % loop over GP outputs
  if isfield(dynmodel,'induce') && size(dynmodel.induce,2)
    x = dynmodel.induce(:,:,min(k,size(dynmodel.induce,3)));
  else
    x = dynmodel.inputs;
  end
 
  iL = diag(exp(-2*dynmodel.hyp(1:D,k)));  
  t1 = sf2(k)*exp(-(m(1:D)-m(D+1:2*D))'*iL*(m(1:D)-m(D+1:2*D))/2);
  xm1 = bsxfun(@minus,x,m(1:D)'); xm2 = bsxfun(@minus,x,m(D+1:end)');      
  e1 = sf2(k)*exp(-sum(xm1*iL.*xm1,2)/2);
  e2 = sf2(k)*exp(-sum(xm2*iL.*xm2,2)/2);
  t2 = e1'*dynmodel.iK(:,:,k)*e2;
  C(k,k) = t1 - t2;
  
  if nargout > 2
    dCdm(k,k,1:D) = -t1*iL*(m(1:D)-m(D+1:2*D))-bsxfun(@times,e1,xm1*iL)'*dynmodel.iK(:,:,k)*e2;
    dCdm(k,k,D+1:2*D) = t1*iL*(m(1:D)-m(D+1:2*D))-bsxfun(@times,e2,xm2*iL)'*dynmodel.iK(:,:,k)*e1;
  end
  
end
