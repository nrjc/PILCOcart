function [nlml, dnlml] = vfe(induce, gp)

% Titsias code adapted from Michalis K. Titsias variational sparse
% pseudo-input GP:
% "Variational Learning of Inducing Variables in Sparse Gaussian Processes"
%
% nlml   1 x 1      negative log marginal likelihood
% dnlml  M x F x E  derivatives of nlml w.r.t. induce
%
% See also <a href="gpa.pdf">gpa.pdf</a>
%
% Rowan McAllister, 2014-12-17

ridge = 1e-6;                      % jitter to make matrix better conditioned
[N, F] = size(gp.inputs); [M, uF, uE] = size(induce);                           % TODO: augment induce angles +  chain derivatives?
if uF ~= F || (uE~=1 && uE ~= gp.E); error(['Wrong inducing inputs']); end
nlml = 0; dnlml = zeros(M, F, gp.E);     % allocate outputs

y = gp.target - bsxfun(@plus,gp.inputs*[gp.hyp.m],[gp.hyp.b]);
for e = 1:gp.E                                    % loop over target dimensions
  if uE > 1; u = induce(:,:,e); else u = induce; end
  b = exp(gp.hyp(e).l');
  c = 2*gp.hyp(e).s;                                               % log signal
  n = exp(2.*gp.hyp(e).n);                                     % noise variance
  
  xb = bsxfun(@rdivide,u,b);                          % divide by length-scales
  x = bsxfun(@rdivide,gp.inputs,b);
  
  Kmm = exp(c-maha(xb,xb)/2) + ridge*eye(M);
  Kmn = exp(c-maha(xb,x)/2);
  
  % see gpa.pdf for an explanation of the following code:
  try
    L = chol(Kmm);
  catch
    nlml = Inf; dnlml = zeros(size(params)); return;
  end
  V = L'\Kmn;
  C = chol(n*eye(M)+V*V');
  U = C'\V;
  isigQ = (eye(N) - U'*U)/n;                   % isigQ: inv(sig*I + Qnn), Eq3
  
  nlml = nlml + 0.5*( ...                                           % Eq9 paper
    y(:,e)'*isigQ*y(:,e) + ...                % exp term
    (N-M)*log(n) + 2*sum(log(diag(C))) + ...  % det term
    N*log(2*pi) + ...                         % const term
    (N*exp(c) - sum(sum(V.*V)))/n);           % trace term
  
  if nargout == 2               % ... and if requested, its partial derivatives
    
    A = n*Kmm + Kmn*Kmn';
    B = Kmn'/A;
    k = Kmn'/Kmm;
    iA = eye(M)/A;
    iKmm = eye(M)/Kmm;
    
    % Note: it might be possible to get rid of these following (slow) loops, to
    % make the code run faster.
    for m=1:M
      
      xbxb = bsxfun(@minus,xb(m,:),xb);
      xbx = bsxfun(@minus,xb(m,:),x);
      dKmm = bsxfun(@rdivide,bsxfun(@times,-xbxb,Kmm(:,m)), b);
      dKmn = bsxfun(@rdivide,bsxfun(@times,-xbx,Kmn(m,:)'),b);
      hdA = n*dKmm + Kmn*dKmn; % half dA
      dL1 = 2*n*(iA(m,:)*hdA - iKmm(m,:)*dKmm);
      
      for f=1:F
        
        dABKmn = bsxfun(@times,hdA(:,f),B(:,m)');
        dABKmn(m,:) = dABKmn(m,:) + hdA(:,f)'*B' - 2*dKmn(:,f)';
        dL2 = y(:,e)'*B*dABKmn*y(:,e);
        
        kdKmm2 = bsxfun(@times,k(:,m),dKmm(:,f)');
        kdKmm2(:,m) = k*dKmm(:,f);
        dL3 = sum(sum((kdKmm2.*k))) - 2*dKmn(:,f)'*k(:,m);
        
        dnlml(m,f,e) = (dL1(f) + dL2 + dL3)/(2*n);
      end
    end
    
  end                                              % end derivative computation
end

% end loop over targets
if 1 == uE; dnlml = sum(dnlml,3); end     % add derivatives if sharing inducing
end