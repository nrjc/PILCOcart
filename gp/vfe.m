function [nlml, o2, o3] = vfe(p, inputs, target, style, test)

ridge = 1e-06;               % relative jitter to make matrix better conditioned
switch style, case 'fitc', fitc = 1; vfe = 0; case 'vfe', vfe = 1; fitc = 0; end
induce = p.induce; hyp = p.hyp;                             % shorthand notation
[N, D] = size(inputs); M = size(induce,1); nlml = 0;

for e = 1:length(hyp)
  y = target(:,e) - inputs*hyp(e).m - hyp(e).b;
  l = exp(hyp(e).l); s2 = exp(2*hyp(e).s); n2 = exp(2*hyp(e).n);
  u = bsxfun(@rdivide, induce, l');                     % scaled inducing inputs
  x = bsxfun(@rdivide, inputs, l');                     % scaled training inputs
  Kuu = s2*(exp(-maha(u,u)/2) + ridge*eye(M));
  Kuf = s2*exp(-maha(u,x)/2);
  L = chol(Kuu)';
  V = L\Kuf;
  r = s2 - sum(V.*V,1)';                % diagonal residual Kff - Kfu Kuu^-1 Kuf
  G = fitc*r + n2; iG = 1./G;
  A = eye(M) + V*bsxfun(@times,iG,V');
  J = chol(A)';
  B = J\V;
  z = iG.*y - (y'.*iG'*B'*B.*iG')';
  nlml = nlml + y'*z/2 + sum(log(diag(J))) + sum(log(G))/2 ...
                                              + vfe*sum(r)/n2/2 + N*log(2*pi)/2;
  R = L'\V;
  iKuu = inv(Kuu);
  q = bsxfun(@times,iG',R) - bsxfun(@times,bsxfun(@times, R, iG')*B'*B,iG');
  o2 = (R*z).^2./(diag(iKuu)-sum(R.*q,2))/2 ...
              + log(1-sum(R.*q,2)./diag(iKuu))/2 + sum(R.*R,2)./diag(iKuu)/n2/2;
  if nargin == 5                                              % make predictions
    Ktu = s2*exp(-maha(bsxfun(@rdivide,test,l'), u)/2);
    Ktf = s2*exp(-maha(bsxfun(@rdivide,test,l'), x)/2);
    R = Ktu*R-Ktf;
    q = bsxfun(@times,iG',R) - bsxfun(@times,bsxfun(@times, R, iG')*B'*B,iG');
    c = s2 - sum((Ktu/L').^2,2);
    o3 = (R*z).^2./(c+sum(R.*q,2))/2 ...
                                - log(1+sum(R.*q,2)./c)/2 + sum(R.*R,2)./c/n2/2;
  end
end
