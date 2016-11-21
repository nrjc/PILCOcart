classdef gpa < handle
  % gpa - Gaussian Process object which handles Angles.
  %
  % Copyright (C) 2014, Carl Edward Rasmussen, Rowan McAllister 2014-11-11
  
  properties(SetAccess = immutable)
    D        %          number of inputs to the GP
    E        %          number of targets
    angi     %          indicies of angular variables
  end
  
  properties
    hyp      % 1 x E    struct array of hyperparameters
    inputs   % n x F    inputs augmented by trig functions, F=D+2*numel(angi)
    target   % n x E    targets
    induce   % MxF(xE)  inducing inputs, shared if no third dim
    iK       %          inverse covariance matrix
    beta
    approxS
    fixLin   %
    on       % E x 1    log std dev observation noise
    pn       % E x 1    log std dev process noise
    opt      % .        struct containing options for minimize
  end
  
  properties (Access = private)          % some internal book-keeping variables
    nota     %          indices of original inputs which are not angles
    notA     %          indices of all inputs which are not angles
    hypA     %
  end
  
  methods
    
    function self = gpa(D, E, angi)      % constructor, set some default values
      if nargin > 2, self.angi = angi; else self.angi = []; end
      self.D = D; self.E = E; A = numel(self.angi);
      self.nota = setdiff(1:D,self.angi);
      self.notA = [self.nota D+1:D+2*A];
      self.hypA = [self.nota reshape(repmat(self.angi,2,1),1,2*A)];
      [self.hyp(1:E).l] = deal(zeros(D,1));        % initialize hyperparameters
      [self.hyp(:).s] = deal(0);
      [self.hyp(:).n] = deal(-1);
      [self.hyp(:).m] = deal(zeros(D,1));     % initialize linear model to zero
      [self.hyp(:).b] = deal(0);
      self.opt = struct('length',-100,'MFEPLS',20,'verbosity',3);
    end
    
    function train(self, data, dyni, dyno)
      % the train function does a sequence of things: 1) copies data to the gpa
      % object, 2) re-initializes hyperparameters, 3) trains hypers by
      % separately, training each target dimension and 4) if necessary, trains
      % fitc sparse gps.
      x = cell2mat(arrayfun(@(Y)Y.state(1:end-1,:),data,'uniformoutput',0)');
      y = cell2mat(arrayfun(@(Y)Y.state(2:end,:),data,'uniformoutput',0)');
      u = cell2mat(arrayfun(@(U)U.action,data,'uniformoutput',0)');
      self.inputs = [x(:,dyni) u];                % inputs are state and action
      for i = 1:numel(self.angi)         % augment with trigonometric functions
        self.inputs(:,end+1) = sin(self.inputs(:,self.angi(i)));
        self.inputs(:,end+1) = cos(self.inputs(:,self.angi(i)));
      end
      self.target = y(:,dyno);                                    % set targets
      
      t = log(std(self.inputs(:,1:self.D)))';      % initialize hyperparaneters
      t(self.angi) = 0; [self.hyp.l] = deal(t);     % length scales to std devs
      t = log(std(self.target));
      s = num2cell(t); [self.hyp.s] = deal(s{:});          % log signal std dev
      s = num2cell(t-log(10)); [self.hyp.n] = deal(s{:});   % log noise std dev
      
      for i=1:self.E              % train gps, each target dimension separately
        [self.hyp(i), v] = minimize(self.hyp(i), @solo, -300, self.inputs, ...
          self.target(:,i), self);
        nlml(i) = v(end);           % save the negative log marginal likelihood
      end
      self.on = [self.hyp.n]' - log(2)/2;                 % split noise equally
      self.pn = [self.hyp.n]' - log(2)/2;     % between process and observation
      
      if size(self.induce,1) ~= 0        % are we using a sparse approximation?
        n = size(self.target,1); [M, d, e] = size(self.induce);
        if M < n             % only call FITC if we have enough training points
          if d == 0                               % initialize inducing inputs?
            self.induce = zeros(M, size(self.inputs,2), e);    % allocate space
            for i = 1:e
              j = randperm(n);                                   % random order
              self.induce(:,:,i) = self.inputs(j(1:M),:);       % random subset
            end
          end
          [self.induce, nlml2] = minimize(self.induce, @fitc, -300, self);
          fprintf('GP NLML, full: %e, sparse: %e, diff: %e\n', ...
            sum(nlml), nlml2(end), nlml2(end)-sum(nlml));
        end
      end
      self.pre();                                % do possible pre-computations
    end                                                             % end train
    
    function [M, S, W, dMdm, dSdm, dVdm, dMds, dSds, dVds] = pred(self, m, s)
      % Base function for computing GP predictions with uncertain inputs. This
      % function acts as a landing pad for gp calls, directing the call on to
      % the appropriate function.
      %
      % gp        .        Gaussian process model struct
      %   hyp     1 x E    struct array of GP hyper-parameters
      %     l     D x 1    log lengthscales
      %     s     1 x 1    log signal standard deviation
      %     n     1 x 1    log noise standard deviation
      %     m     D x 1    optional linear weights for the GP mean
      %     b     1 x 1    optional biase for the GP mean
      %   inputs  n x F    training inputs, F = D+2*length(angi)
      %   target  n x E    training targets
      %   induce  PxF(xE)  optional inducing inputs, shared or separate per output
      %   iK      nxnxE    inverse covariance matrix
      %   beta    n x E    iK*target
      %   approxS
      %   angi
      % m         D x 1    mean of the test distribution
      % s         D x D    covariance matrix of the test distribution
      % M         E x 1    mean of predictive distribution
      % S         E x E    covariance of predictive distribution
      % W         D x E    inv(s) times covariance between input and output
      % dMdm      E x D    deriv of output mean w.r.t. input mean
      % dSdm      E*ExD    deriv of output covariance w.r.t input mean
      % dWdm      D*ExD    deriv of input-output cov w.r.t. input mean
      % dMds      ExD*D    deriv of ouput mean w.r.t input covariance
      % dSds      E*ExD*D  deriv of output cov w.r.t input covariance
      % dWds      D*ExD*D  deriv of inv(s)*input-output covar w.r.t input cov
      
      D = self.D; DD = D*D; A = numel(self.angi); F = D+2*A;
      E = length(self.hyp);
      i = [1:D]; j = [D+1:F]; k = self.nota;
      nAA = sub2ind2(F,self.notA,self.notA);
      nAE = sub2ind2(F,self.notA,1:E);
      
      [ma, sa, ca, dmadm, dsadm, dcadm, dmads, dsads, dcads] = ...
        gTrigN(m, s, self.angi);
      switch (nargout > 3) + 2*(1-isempty(self.approxS))
        case 0                                         % no derivatives, full S
          [M, S, VV(self.notA,:)] = ...
            gpp0(self, ma(self.notA), sa(self.notA,self.notA));
        case 1                                            % derivatives, full S
          [M, S, VV(self.notA,:), dMdma(:,self.notA), dSdma(:,self.notA), ...
            dVdma(nAE,self.notA), dMdsa(:,nAA), dSdsa(:,nAA), dVdsa(nAE,nAA)] = ...
            gpp0d(self, ma(self.notA), sa(self.notA,self.notA));
        case 2                                       % no derivatives, approx S
          [M, S, V] = gpas(gp, m, s);
        case 3                                          % derivatives, approx S
          [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpasd(gp, m, s);
      end
      
      hypm = [self.hyp.m];
      M = M + hypm'*m(1:D) + [self.hyp.b]';          % add linear contributions
      V = ca*VV;
      S = S + V'*s(1:D,1:D)*hypm + hypm'*s(1:D,1:D)*V + hypm'*s(1:D,1:D)*hypm;
      W = V + [self.hyp.m];
      
      if nargout > 3
        Vdm = dVdma*dmadm + dVdsa*dsadm;
        caVdm = prodd(ca,Vdm) + prodd([],dcadm,VV);
        dMdm = dMdma*dmadm + dMdsa*dsadm + hypm';
        dSdm = dSdma*dmadm + dSdsa*dsadm + ...
          prodd([],reshape(permute(reshape(caVdm,[D E D]),[2 1 3]),[D*E D]),s(1:D,1:D)*hypm) + ...
          prodd(hypm'*s(1:D,1:D),caVdm);
        
        dVdm = caVdm;
        Vds = dVdma*dmads + dVdsa*dsads;
        caVds = prodd(ca,Vds) + prodd([],dcads,VV);
        dMds = dMdma*dmads + dMdsa*dsads;
        dSds = dSdma*dmads + dSdsa*dsads + ...
          prodd([],reshape(permute(reshape(caVds,[D E DD]),[2 1 3]),[D*E DD]),s(1:D,1:D)*hypm) + ...
          prodd(hypm'*s(1:D,1:D),caVds) + kron(hypm',V') + kron(V',hypm') + kron(hypm',hypm');
        
        dVds = caVds;
      end
      
    end                                                         % function pred
    
    function pre(self)
      % pre - function to precompute the inverse covariance matrix iK and beta
      % vector which don't depend on the test inputs. The function checks to
      % see if sparse approximations are being used, and acts accordingly.
      y = self.target - ...             % y is targets less linear contribution
        bsxfun(@plus,self.inputs(:,1:self.D)*[self.hyp.m], [self.hyp.b]);
      if numel(self.induce) > 0                          % FITC initialisation?
        ridge = 1e-6;                % jitter to make matrix better conditioned
        [np, ~, pE] = size(self.induce);
        self.iK = zeros(np, np, self.E); self.beta = zeros(np, self.E);
        for i=1:self.E
          h = self.hyp(i);
          z = bsxfun(@times, self.induce(:,self.notA,min(i,pE)), ...
            exp(-h.l(self.hypA)'));
          x = bsxfun(@times, self.inputs(:,self.notA), exp(-h.l(self.hypA)'));
          Kmm = exp(2*h.s-maha(z,z)/2) + ridge*eye(np);             % add ridge
          Kmn = exp(2*h.s-maha(z,x)/2);
          L = chol(Kmm)';
          V = L\Kmn;                                       % inv(sqrt(Kmm))*Kmn
          G = exp(2*h.s)-sum(V.^2);
          if isprop(self,'nigp'); G = G + self.nigp(:,i)'; end
          G = sqrt(1+G/exp(2*h.n));
          V = bsxfun(@rdivide,V,G);
          Am = chol(exp(2*h.n)*eye(np) + V*V')';
          At = L*Am;                              % chol(sig*B) [thesis, p. 40]
          iAt = At\eye(np);
          iK = ((Am\(bsxfun(@rdivide,V,G)))'*iAt)';
          self.beta(:,i) = iK*y(:,i);
          iB = iAt'*iAt.*exp(2*h.n);             % inv(B), [Ed's thesis, p. 40]
          self.iK(:,:,i) = Kmm\eye(np) - iB;    % cov matrix for predictive var
        end
      else                                             % Full GP initialisation
        n = size(self.target,1); K = zeros(n,n);
        self.iK = zeros(n,n,self.E); self.beta = zeros(n,self.E);
        for i=1:self.E                                   % compute K and inv(K)
          z = bsxfun(@times, self.inputs(:,self.notA), ...       % scale inputs
            exp(-self.hyp(i).l(self.hypA)'));
          K = exp(2*self.hyp(i).s-maha(z,z)/2);
          if isprop(self,'nigp')
            L = chol(K + exp(2*self.hyp(i).n)*eye(n) + diag(self.nigp(:,i)))';
          else
            L = chol(K + exp(2*self.hyp(i).n)*eye(n))';
          end
          self.iK(:,:,i) = L'\(L\eye(n));
          self.beta(:,i) = L'\(L\y(:,i));
        end
      end
    end                                                               % end pre
    
  end                                                             % end methods
  
end                                                                 % end class


function [nlml, dnlml] = solo(h, x, y, self)
% solo - function for training a sinlge gp with a scalar output.
n = numel(y); D = self.D; A = numel(self.angi);
z = bsxfun(@times, x(:,self.notA), exp(-h.l(self.hypA)'));
y = y - x(:,1:D)*h.m - h.b;                  % targets less linear contribution

K = exp(2*h.s-maha(z,z)/2);                      % noise-free covariance matrix
L = chol(K + exp(2*h.n)*eye(n))';     % cholesky of the noisy covariance matrix
alpha = solve_chol(L', y);
nlml = y'*alpha/2 + sum(log(diag(L))) + n*log(2*pi)/2;       % neg log marg lik

if nargout > 1                                       % derivative computations?
  if self.fixLin                          % derivative of linear model desired?
    dnlml.m = zeros(D,1); dnlml.b = 0;                        % zero derivative
  else
    dnlml.m = -x(:,1:D)'*alpha; dnlml.b = -sum(alpha);
  end
  W = L'\(L\eye(n))-alpha*alpha';                  % precompute for convenience
  t = sq_dist(z',[],K.*W)/2;
  dnlml.l(self.nota) = t(1:D-A);                        % regular length scales
  dnlml.l(self.angi) = sum(reshape(t(D-A+1:end),2,A),1);            % trig sums
  dnlml.s = K(:)'*W(:);
  dnlml.n = exp(2*h.n)*trace(W);
end
end

function [nlml, dnlml] = fitc(induce, gp)
% fitc code adapted from Ed Snelson's thesis.
%
% nlml      negative log marginal likelihood
ridge = 1e-6;                        % jitter to make matrix better conditioned
[N, F] = size(gp.inputs); [M, uF, uE] = size(induce); A = numel(gp.angi);
if uF ~= F || (uE~=1 && uE ~= gp.E); error(['Wrong inducing inputs']); end
nlml = 0; dfxb = zeros(M, F); dnlml = zeros(M, F, gp.E);     % allocate outputs

y = gp.target - bsxfun(@plus,gp.inputs(:,1:gp.D)*[gp.hyp.m],[gp.hyp.b]);
for j = 1:gp.E                                    % loop over target dimensions
  if uE > 1; u = induce(:,:,j); else u = induce; end
  b = exp(gp.hyp(j).l(gp.hypA)');
  c = gp.hyp(j).s;                                         % log signal std dev
  sig = exp(2.*gp.hyp(j).n);                                   % noise variance
  
  xb = bsxfun(@rdivide,u(:,gp.notA),b);               % divide by length-scales
  x = bsxfun(@rdivide,gp.inputs(:,gp.notA),b);
  
  Kmm = exp(2*c-maha(xb,xb)/2) + ridge*eye(M);
  Kmn = exp(2*c-maha(xb,x)/2);
  
  try
    L = chol(Kmm)';
  catch
    nlml = Inf; dnlml = zeros(size(params)); return;
  end
  V = L\Kmn;                                               % inv(sqrt(Kmm))*Kmn
  
  Gamma = 1 + (exp(2*c)-sum(V.^2)')/sig;        % Gamma = diag(Knn-Qnn)/sig + I
  
  V = bsxfun(@rdivide,V,sqrt(Gamma)');    % inv(sqrt(Kmm))*Kmn*inv(sqrt(Gamma))
  y(:,j) = y(:,j)./sqrt(Gamma);
  Am = chol(sig*eye(M) + V*V')';        % chol(inv(sqrt(Kmm))*A*inv(sqrt(Kmm)))
  % V*V' = inv(chol(Kmm)')*K*inv(diag(Gamma))*K'*inv(chol(Kmm)')'
  Vy = V*y(:,j);
  beta = Am\Vy;
  
  nlml = nlml + sum(log(diag(Am))) + (N-M)/2*log(sig) + sum(log(Gamma))/2 ...
    + (y(:,j)'*y(:,j) - beta'*beta)/2/sig + 0.5*N*log(2*pi);
  
  if nargout == 2               % ... and if requested, its partial derivatives
    At = L*Am; iAt = At\eye(M);                   % chol(sig*B) [thesis, p. 40]
    iA = iAt'*iAt;                                                 % inv(sig*B)
    
    % C = iAt*Kmn2;
    % iK = diag(1./(sig*Gamma)) - C'*C;
    % B = At*At'./sig; % B matrix [thesis, p. 40]
    
    iAmV = Am\V;                                                    % inv(Am)*V
    B1 = At'\(iAmV);
    b1 = At'\beta;                                                  % b1 = B1*y
    
    iLV = L'\V;                                 % inv(Kmm)*Kmn*inv(sqrt(Gamma))
    iL = L\eye(M);
    iKmm = iL'*iL;
    
    mu = ((Am'\beta)'*V)';
    bs = y(:,j).*(beta'*iAmV)'/sig - sum(iAmV.*iAmV)'/2 - ...
      (y(:,j).^2+mu.^2)/2/sig + 0.5;
    TT = iLV*(bsxfun(@times,iLV',bs));
    Kmn = bsxfun(@rdivide,Kmn,sqrt(Gamma)');                    % overwrite Kmn
    
    for i = 1:numel(gp.notA)                  % derivatives wrt inducing inputs
      dsq_mm = bsxfun(@minus,xb(:,i),xb(:,i)').*Kmm;
      dsq_mn = bsxfun(@minus,-xb(:,i),-x(:,i)').*Kmn;
      dGamma = -2/sig*dsq_mn.*iLV;
      
      dfxb(:,gp.notA(i)) = -b1.*(dsq_mn*(y(:,j)-mu)/sig + dsq_mm*b1) + ...
        dGamma*bs + sum((iKmm - iA*sig).*dsq_mm,2) - 2/sig*sum(dsq_mm.*TT,2);
      dfxb(:,gp.notA(i)) = dfxb(:,gp.notA(i)) + sum(dsq_mn.*B1,2);
      dfxb(:,gp.notA(i)) = dfxb(:,gp.notA(i))/b(i);
    end
    
    dnlml(:,:,j) = dfxb;
  end                                              % end derivative computation
end                                                     % end loop over targets
if 1 == uE; dnlml = sum(dnlml,3); end     % add derivatives if sharing inducing
end

function [nlml, dnlml] = titsias(induce, gp)
% titsias code adapted from Michalis K. Titsias variational sparse
% pseudo-input GP:
% "Variational Learning of Inducing Variables in Sparse Gaussian Processes"
%
% nlml      negative log marginal likelihood
% dnlml     derivatives of nlml w.r.t. induce
%
% See also <a href="gpa.pdf">gpa.pdf</a>
ridge = 1e-6;                      % jitter to make matrix better conditioned
[N, F] = size(gp.inputs); [M, uF, uE] = size(induce); A = numel(gp.angi);
if uF ~= F || (uE~=1 && uE ~= gp.E); error(['Wrong inducing inputs']); end
nlml = 0; dfxb = zeros(M, F); dnlml = zeros(M, F, gp.E);     % allocate outputs

y = gp.target - bsxfun(@plus,gp.inputs(:,1:gp.D)*[gp.hyp.m],[gp.hyp.b]);
for j = 1:gp.E                                    % loop over target dimensions
  if uE > 1; u = induce(:,:,j); else u = induce; end
  b = exp(gp.hyp(j).l(gp.hypA)');
  c = gp.hyp(j).s;                                         % log signal std dev
  sig = exp(2.*gp.hyp(j).n);                                   % noise variance
  
  xb = bsxfun(@rdivide,u(:,gp.notA),b);               % divide by length-scales
  x = bsxfun(@rdivide,gp.inputs(:,gp.notA),b);
  
  Kmm = exp(2*c-maha(xb,xb)/2) + ridge*eye(M);
  Kmn = exp(2*c-maha(xb,x)/2);
  
  % see gpa.pdf for an explanation of the following code:
  try
    L = chol(Kmm);
  catch
    nlml = Inf; dnlml = zeros(size(params)); return;
  end
  V = L'\Kmn;
  C = chol(sig*eye(M)+V*V');
  U = C'\V;
  isigQ = (eye(N) - U'*U)/sig;                   % isigQ: inv(sig*I + Qnn), Eq3
  
  nlml = nlml + 0.5*( ...                                           % Eq9 paper
    maha(y(:,j),-y(:,j),isigQ) + ...            % exp term
    (N-M)*log(sig) + 2*sum(log(diag(C))) + ...  % det term
    N*log(2*pi) + ...                           % const term
    (N*exp(2*c) - sumsqr(V))/sig);              % trace term
  
  if nargout == 2               % ... and if requested, its partial derivatives
    % TODO                                                                      % TODO
    
    for i = 1:numel(gp.notA)                  % derivatives wrt inducing inputs
      % TODO                                                                    % TODO
    end
    
    % TODO
    % dnlml(:,:,j) = dfxb;
  end                                              % end derivative computation
end                                                     % end loop over targets
if 1 == uE; dnlml = sum(dnlml,3); end     % add derivatives if sharing inducing
end

function [M, V, C] = gpp0(gp, m, s)
% Compute joint predictions for multiple GPs with uncertain inputs. Predictive
% variances contain uncertainty about the function, but no noise.
%
% gp               Gaussian process model struct
%   hyp     1 x E  struct array of GP hyper-parameters
%     l     D x 1  log lengthscales
%     s     1 x 1  log signal standard deviation
%     n     1 x 1  log noise standard deviation
%   inputs  n x D  matrix of training inputs
%   target  n x E  matrix of training targets
%   iK      nxnxE  inverse covariance matrix
%   beta    n x E  iK*(targets - mean function of inputs)
% m         D x 1  mean of the test distribution
% s         D x D  covariance matrix of the test distribution
% M         E x 1  mean of pred. distribution
% V         E x E  covariance of the pred. distribution
% C         D x E  inv(s) times covariance between input and output

if numel(gp.induce) > 0; x = gp.induce; else x = gp.inputs; end

[n, ~, pE] = size(x); D = numel(gp.notA); E = size(gp.beta,2);
h = gp.hyp; iK = gp.iK; beta = gp.beta;
M = zeros(E,1); C = zeros(D,E); V = zeros(E); k = zeros(n,E); a = zeros(D,E);

inp = bsxfun(@minus,x(:,gp.notA),m');     % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  il = diag(exp(-h(i).l(gp.hypA)));                               % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;                             % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D);                      % Lambda^-1/2 * V * *Lambda^-1/2 + I
  t = in/B;                                      % in.*t = (X-m) (V+L)^-1 (X-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il;
  c = exp(2*h(i).s)/sqrt(det(B));                   % = sf2/sqrt(det(V*iL + I))
  M(i) = sum(lb)*c;                                            % predicted mean
  C(:,i) = tL'*lb*c;                     % inv(s) times input-output covariance
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
  liBl = il*(B\il); xm = x(:,gp.notA,min(i,pE))'*lb*c;
  a(:,i) = diag(exp(2*h(i).l(gp.hypA)))*liBl*m*M(i) + s*liBl*xm;
end

hl = [h.l];
iL = exp(-2*hl(gp.hypA,:)); xiL = bsxfun(@times,inp,permute(iL,[3,1,2]));
for i=1:E                  % compute predictive covariance, non-central moments
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); t = 1/sqrt(det(R));
    L = exp(bsxfun(@plus,k(:,i),k(:,j)') + maha(xiL(:,:,i),-xiL(:,:,j),R\s/2));
    V(i,j) = beta(:,i)'*L*beta(:,j)*t;                   % variance of the mean
    V(j,i) = V(i,j);
  end
  V(i,i) = V(i,i) + exp(2*h(i).s) - t*sum(sum(iK(:,:,i).*L)); % last L has i==j
end
V = V - M*M';                                              % centralize moments
end

function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = gpp0d(gp, m, s)
% Compute joint predictions and derivatives for multiple GPs with uncertain
% inputs. This version uses a linear + bias mean function and also
% computes derivatives of the output moments w.r.t the input moments.
% Predictive variances contain uncertainty about the function, but no noise.
%
% gp               Gaussian process model struct
%   hyp(i)  1-by-E struct array of GP hyper-parameters
%     .l    D-by-1 log lengthscales
%     .s    1-by-1 log signal standard deviation
%     .n    1-by-1 log noise standard deviation
%     .m    D-by-1 linear weights for the GP mean
%     .b    1-by-1 biases for the GP mean
%   inputs  n by D matrix of training inputs
%   target  n by E matrix of training targets
%   iK      n-by-n-by-E, inverse covariance matrix
%   beta    n-by-E, iK*(targets - mean function of inputs)
% m         D by 1 vector, mean of the test distribution
% s         D by D covariance matrix of the test distribution
%
% M         E-by-1 vector, mean of pred. distribution
% S         E-by-E matrix, covariance of the pred. distribution
% V         D-by-E inv(s) times covariance between input and prediction
% dMdm      E-by-D, deriv of output mean w.r.t. input mean
% dSdm      E^2-by-D, deriv of output covariance w.r.t input mean
% dVdm      D*E-by-D, deriv of input-output cov w.r.t. input mean
% dMds      E-by-D^2, deriv of ouput mean w.r.t input covariance
% dSds      E^2-by-D^2, deriv of output cov w.r.t input covariance
% dVds      D*E-by-D^2, deriv of inv(s)*input-output covariance w.r.t input cov

if numel(gp.induce) > 0, x = gp.induce; else x = gp.inputs; end

[n, ~, pE] = size(x); D = numel(gp.notA); E = size(gp.beta,2);
h = gp.hyp; iK = gp.iK; beta = gp.beta;

k = zeros(n,E); M = zeros(E,1); V = zeros(D,E); S = zeros(E);      % initialize
dMds = zeros(E,D,D); dSdm = zeros(E,E,D); a = zeros(D,E); T = zeros(D);
dSds = zeros(E,E,D,D); dVds = zeros(D,E,D,D); dadm = zeros(D,D,E);
dads = zeros(D,E,D,D);

inp = bsxfun(@minus,x(:,gp.notA),m');     % x - m, either n-by-D or n-by-D-by-E

for i=1:E     % compute predicted mean and inv(s) times input-output covariance
  xi = x(:,gp.notA,min(i,pE));
  il = diag(exp(-h(i).l(gp.hypA)));                               % Lambda^-1/2
  in = inp(:,:,min(i,pE))*il;                             % (X - m)*Lambda^-1/2
  B = il*s*il+eye(D); liBl = il/B*il;                  % liBl = (Lambda + S)^-1
  t = in/B;                                      % in.*t = (x-m) (S+L)^-1 (x-m)
  l = exp(-sum(in.*t,2)/2); lb = l.*beta(:,i);
  tL = t*il; tlb = bsxfun(@times,tL,lb);
  c = exp(2*h(i).s)/sqrt(det(B));                     % sf2/sqrt(det(S*iL + I))
  M(i) = sum(lb)*c;
  V(:,i) = tL'*lb*c;                     % inv(s) times input-output covariance
  dMds(i,:,:) = c*tL'*tlb/2 - liBl*M(i)/2;
  for d = 1:D
    dVds(d,i,:,:) = c*bsxfun(@times,tL,tL(:,d))'*tlb/2 - liBl*V(d,i)/2 - V(:,i)*liBl(d,:);
  end
  k(:,i) = 2*h(i).s-sum(in.*in,2)/2;
  xm = xi'*lb*c; L2iBL = diag(exp(2*h(i).l(gp.hypA)))*liBl;
  LiBLxm = liBl*xm; sLx = s*liBl*xi';
  a(:,i) = L2iBL*m*M(i) + s*LiBLxm;
  dadm(:,:,i) = L2iBL*(M(i)*eye(D) + m*V(:,i)') + s*liBl*xi'*bsxfun(@times,tL,lb)*c;
  dads(:,i,:,:) = -bsxfun(@times,L2iBL,permute(liBl*m,[2,3,1]))*M(i) ...
    + bsxfun(@times,L2iBL*m,dMds(i,:,:));
  for d=1:D; dads(d,i,d,:) = dads(d,i,d,:) + permute(LiBLxm,[2,3,4,1]); end
  tLtlb = bsxfun(@times,tL,permute(tlb,[1,3,2]));
  Lbc = bsxfun(@times,lb,permute(liBl*c,[3,1,2]));
  dads(:,i,:,:) = squeeze(dads(:,i,:,:)) - bsxfun(@times,s*liBl,permute(LiBLxm,[2,3,1]))...
    + reshape(sLx*reshape(c*tLtlb-Lbc,n,D*D),D,D,D)/2;
end
dMdm = V'; dVdm = 2*permute(dMds,[2 1 3]);                  % derivatives wrt m

hl = [h.l];
iL = exp(-2*hl(gp.hypA,:)); inpil = bsxfun(@times,inp,permute(iL,[3,1,2]));
for i=1:E           % compute predictive covariance matrix, non-central moments
  ii = inpil(:,:,i);
  
  for j=1:i
    R = s*diag(iL(:,i)+iL(:,j))+eye(D); t = 1/sqrt(det(R)); ij = inpil(:,:,j);
    L = exp(bsxfun(@plus,k(:,i),k(:,j)')+maha(ii,-ij,R\s/2));
    if i==j
      iKL = iK(:,:,i).*L; s1iKL = sum(iKL,1); s2iKL = sum(iKL,2);
      S(i,j) = t*(beta(:,i)'*L*beta(:,i) - sum(s1iKL));
      
      zi = ii/R;
      bibLi = L'*beta(:,i).*beta(:,i); cbLi = L'*bsxfun(@times, beta(:,i), zi);
      r = (bibLi'*zi*2 - (s2iKL' + s1iKL)*zi)*t;
      for d = 1:D
        T(d,1:d) = 2*(zi(:,1:d)'*(zi(:,d).*bibLi) + ...
          cbLi(:,1:d)'*(zi(:,d).*beta(:,i)) - zi(:,1:d)'*(zi(:,d).*s2iKL) ...
          - zi(:,1:d)'*(iKL*zi(:,d)));
        T(1:d,d) = T(d,1:d)';
      end
    else
      zi = ii/R; zj = ij/R;
      S(i,j) = beta(:,i)'*L*beta(:,j)*t;
      
      bibLj = L*beta(:,j).*beta(:,i); bjbLi = L'*beta(:,i).*beta(:,j);
      cbLi = L'*bsxfun(@times, beta(:,i), zi);
      cbLj = L*bsxfun(@times, beta(:,j), zj);
      
      r = (bibLj'*zi+bjbLi'*zj)*t;
      for d = 1:D
        T(d,1:d) = zi(:,1:d)'*(zi(:,d).*bibLj) + ...
          cbLi(:,1:d)'*(zj(:,d).*beta(:,j)) + zj(:,1:d)'*(zj(:,d).*bjbLi) + ...
          cbLj(:,1:d)'*(zi(:,d).*beta(:,i));
        T(1:d,d) = T(d,1:d)';
      end
    end
    S1 = S(i,j);
    S(i,j) = S1;
    S(j,i) = S(i,j);
    
    dSdm(i,j,:) = r - M(i)*dMdm(j,:)-M(j)*dMdm(i,:);
    dSdm(j,i,:) = dSdm(i,j,:);
    T = (t*T-S1*diag(iL(:,i)+iL(:,j))/R)/2;
    T = T - reshape(M(i)*dMds(j,:,:) + M(j)*dMds(i,:,:),D,D);
    dSds(i,j,:,:) = T;
    dSds(j,i,:,:) = permute(dSds(i,j,:,:),[1,2,4,3]);
  end
  
  S(i,i) = S(i,i) + exp(2*h(i).s);            % add signal variance to diagonal
end

S = S - M*M';                                              % centralize moments

dMds = reshape(dMds,[E D*D]);                           % vectorise derivatives
dSds = reshape(dSds,[E*E D*D]); dSdm = reshape(dSdm,[E*E D]);
dVds = reshape(dVds,[D*E D*D]); dVdm = reshape(dVdm,[D*E D]);
end