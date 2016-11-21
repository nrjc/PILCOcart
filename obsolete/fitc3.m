function [nlml, dnlml] = fitc3(args, gp, style)

induce = args.induce;
n2s = exp(2*args.n);
switch style, case 'fitc', fitc = 1; vfe = 0; case 'vfe', vfe = 1; fitc = 0; end
[N, D] = size(gp.inputs); E = length(gp.hyp);
[M, uD, uE] = size(induce); 
if uD ~= D || (uE~=1 && uE ~= E); error('Wrong size of inducing inputs'); end

ridge = 1e-06;                        % jitter to make matrix better conditioned
nlml = 0; dnlml.induce = zeros(M,D,E);               % zero and allocate outputs
dnlml.n = zeros(E,1);
for e = 1:E
  y = gp.target(:,e) - gp.inputs*gp.hyp(e).m - gp.hyp(e).b;
  l = exp(gp.hyp(e).l); s2 = exp(2*gp.hyp(e).s); n2 = n2s(e); % n2 = exp(2*gp.hyp(e).n);
  u = bsxfun(@rdivide, induce(:,:,1+(uE>1)*(e-1)), l'); % scaled inducing inputs
  x = bsxfun(@rdivide, gp.inputs, l');                  % scaled training inputs
  Kuu = s2*exp(-maha(u,u)/2) + ridge*eye(M);
  Kuf = s2*exp(-maha(u,x)/2);
  L = chol(Kuu)';
  V = L\Kuf;
  G = fitc*(s2 - sum(V.*V,1)') + n2;
  iG = 1./G;
  A = eye(M) + V*bsxfun(@times,iG,V');
  J = chol(A)';
  B = J\V;
  z = iG.*y - (y'.*iG'*B'*B.*iG')';
  T = vfe*(N*s2 - sum(sum(V.*V)));
  nlml = nlml + y'*z/2 + sum(log(diag(J))) + sum(log(G))/2 ...
                                                       + T/n2/2 + N*log(2*pi)/2;
                                    
  if nargout == 2                % ... and if requested, its partial derivatives
    R = L'\V;
    RiG = bsxfun(@times,R,iG');
    RdlqdQ = -R*z*z'+RiG-bsxfun(@times,RiG*B'*B,iG');
    dlqdG = z.^2-iG+iG.^2.*sum(B.*B,1)';
    for d = 1:D
      P = bsxfun(@minus,u(:,d),u(:,d)').*Kuu*R ...
                                           - bsxfun(@minus,u(:,d),x(:,d)').*Kuf;
      dnlml.induce(:,d,e) = ...
                 (sum(P.*RdlqdQ,2) + P.*R*dlqdG*fitc - vfe*sum(P.*R,2)/n2)/l(d);
    end
    
    trace_iQG = sum(iG) - sum((B.*B)*(iG.*iG));                % trace(inv(Q+G))
    dnlml.n(e) = trace_iQG*n2 - z'*z*n2 -T/n2;
  end
end
if 1 == uE; dnlml.induce = sum(dnlml.induce,3); end  % combine derivatives if sharing inducing