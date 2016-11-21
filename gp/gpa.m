classdef gpa < handle
  % gpa - Gaussian Process object which handles Angles.
  %
  % See also <a href="gpd.pdf">gpd.pdf</a>.
  % Copyright (C) 2015, Carl Edward Rasmussen, Rowan McAllister 2015-07-23
  
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
    idx      %          index into parameters structure
    iKdl
    K
    R
    Kclean
    beta     % M x E    cached prediction variable (see gpd.pdf)
    W        % MxMxE    cached prediction variable (see gpd.pdf) inverse covariance matrix
    approxS
    fixLin   %
    on       % E x 1    log std dev observation noise
    pn       % E x 1    log std dev process noise
    opt      % .        struct containing options for minimize
    opt_sparse %        struct containing options for minimize
    inf_method % Inference method. Either 'full', 'fitc', or 'vfe'.
  end
  
  methods
    
    function self = gpa(D, E, angi, inf_method, opt, opt_sparse) % constructor
      self.D = D;
      self.E = E;
      if exist('angi','var'); self.angi = angi; else self.angi = []; end
      
      % inference method
      if exist('inf_method','var'); self.inf_method = lower(inf_method);
      else self.inf_method = 'full'; end
      assert(any(strcmp(self.inf_method,{'full', 'fitc', 'vfe'})),...
        'error: inf_method must be full, fitc or vfe.');
      if ~strcmp(self.inf_method,'full'); self.induce= zeros(300,0,self.E); end
      
      % optimisation methods
      if exist('opt','var'); self.opt = opt;
      else self.opt = struct(...
          'length',-300,'method','BFGS','MFEPLS',20,'verbosity',3,'fh',6);
      end
      if exist('opt_sparse','var'); self.opt_sparse = opt_sparse;
      else self.opt_sparse = self.opt; end
    end
    
    function train(self, data, dyni, dyno)
      % TRAIN does a sequence of things:
      % 1) copies data to the gpa object,
      % 2) re-initializes hyperparameters,
      % 3) trains hypers by separately, training each target dimension and
      % 4) if necessary, trains sparse gps.
      x = cell2mat(arrayfun(@(Y)Y.state(1:end-1,:),data,'uniformoutput',0)');
      y = cell2mat(arrayfun(@(Y)Y.state(2:end,:),data,'uniformoutput',0)');
      u = cell2mat(arrayfun(@(U)U.action,data,'uniformoutput',0)');
      self.inputs = [x(:,dyni) u];                % inputs are state and action
      for i = 1:numel(self.angi)         % augment with trigonometric functions
        self.inputs(:,end+1) = sin(self.inputs(:,self.angi(i)));
        self.inputs(:,end+1) = cos(self.inputs(:,self.angi(i)));
      end
      self.target = y(:,dyno);                                    % set targets
      
      hyp = self.inithyp(); nlml = nan(self.E,1);
      for i=1:self.E            % train gps, each target dimension separately
        [hyp(i),v] = minimize(hyp(i), @self.gp, self.opt, ...
          self.inputs, self.target(:,i));
        nlml(i) = v(end);         % save the negative log marginal likelihood
      end
      if strcmp(self.inf_method,'full') || numel(self.induce) == 0
        self.hyp = hyp;
      end
      
      if ~strcmp(self.inf_method,'full') % are we using a sparse approximation?
        n = size(self.target,1); [M, d, e] = size(self.induce);
        if M < n    % only call sparse method if we have enough training points
          if d == 0                               % initialize inducing inputs?
            self.induce = zeros(M, size(self.inputs,2), e);    % allocate space
            for i = 1:e
              j = randperm(n);                                   % random order
              self.induce(:,:,i) = self.inputs(j(1:M),:);       % random subset
            end
          end
          nlml2 = nan(self.E,1);
          for i=1:e % loop each E inducing points (faster), since independent
            args.induce = self.induce(:,:,i);
            args.hyp = self.hyp(i);
            [args2, v] = minimize(args, @sgp, self.opt_sparse, ...
                               self.inputs, self.target(:,i), self.inf_method);
            self.induce(:,:,i) = args2.induce;
            self.hyp(i) = args2.hyp;
            nlml2(i) = v(end);
          end
          fprintf('GP NLML, full: %e, sparse: %e, diff: %e\n', ...
                                  sum(nlml), sum(nlml2), sum(nlml2)-sum(nlml));
        end
      end
      self.on = [self.hyp.n]' - log(2)/2;
      self.pn = [self.hyp.n]' - log(10);
      d = size(self.inputs,2);
      if ~isfield(self.hyp,'m'); [self.hyp.m] = deal(zeros(d,1)); end
      if ~isfield(self.hyp,'b'); [self.hyp.b] = deal(0); end
      
      self.pre();                                % do possible pre-computations
    end                                                             % end train
    
    function [nlml, dnlml] = gp(self, h, x, y)
      % GP computes the negative log maginal likelihood and optionally
      % derivatives for a gp with possibly angular inputs. The function is a
      % wrapper which deals with the angular variables and calls generic gp
      % code to do the actual work. Signal to noise ratios over a certain limit
      % are penalized.
      
      [nlml, dnlml] = gp(h, x, y, self.fixLin);
      dnlml.l(self.angi) = 0;
      i = self.D+(1:2*length(self.angi));
      dnlml.l(i) = repmat(mean(reshape(dnlml.l(i),2,[])),2,1);
      if isfield(h,'m'), dnlml.m(i) = 0; end
      pwr = 30; maxsnr = 500; r = (exp(h.s-h.n)/maxsnr)^pwr;  % enforce max snr
      nlml = nlml + r; dnlml.n = dnlml.n - pwr*r; dnlml.s = dnlml.s + pwr*r;
    end
    
    function hyp = inithyp(self)
      % INITHYP initialises the hyperparmeters prior to minimise.
      l = log(std(self.inputs))';
      i = self.D+(1:2*length(self.angi));
      l(i) = repmat(mean(reshape(l(i),2,[])),2,1);   % averages for trig inputs
      l(self.angi) = l(self.angi) + 20;     % very long length scale for angles
      s = log(std(self.target));
      n = s - log(10);
      for i = 1:self.E
        hyp(i).l = l;                               % length scales to std devs           
        hyp(i).s = s(i);                                   % log signal std dev
        hyp(i).n = n(i);                                    % log noise std dev
        hyp(i).m = zeros(size(self.inputs,2),1);
        hyp(i).b = 0;
      end
    end
    
    function pre(self)
      % PRE - function to precompute the inverse covariance matrix iK and beta
      % vector which don't depend on the test inputs.
      if strcmp(self.inf_method,'full') || numel(self.induce) == 0
        pre_full(self);  % full GP initialisation
      else
        pre_sparse(self); % otherwise sparse initialisation
      end
    end                                                               % end pre
    
    function pre_full(self)
      % PRE_FULL - function to precompute the W (the inverse covariance matrix
      % iK) and beta vector which don't depend on the test inputs.
      y = self.target - ...             % y is targets less linear contribution
        bsxfun(@plus,self.inputs*[self.hyp.m], [self.hyp.b]);
      n = size(self.target,1);
      self.W = zeros(n,n,self.E); self.beta = zeros(n,self.E);
      for i=1:self.E                                   % compute K and inv(K)
        z = bsxfun(@times, self.inputs, exp(-self.hyp(i).l')); % scale inputs
        K = exp(2*self.hyp(i).s-maha(z,z)/2);
        if isprop(self,'nigp')
          L = chol(K + exp(2*self.hyp(i).n)*eye(n) + diag(self.nigp(:,i)))';
        else
          L = chol(K + exp(2*self.hyp(i).n)*eye(n))';
        end
        self.W(:,:,i) = L'\(L\eye(n));
        self.beta(:,i) = L'\(L\y(:,i));
      end
    end
    
    function pre_sparse(self)
      % PRE_SPARSE - function to precompute the W and beta vector according to 
      % whichever sparse method is being used (see gpd.pdf for details).
      % Implementation follows Edward Lloyd Snelson's PhD thesis (page 40):
      % `Flexible and efficient Gaussian process models for machine learning'
      
      y = self.target - ...               % y is targets less linear contribution
        bsxfun(@plus,self.inputs*[self.hyp.m], [self.hyp.b]);
      ridge = 1e-6;                  % jitter to make matrix better conditioned
      n = size(self.inputs,1);
      [M, ~, pE] = size(self.induce);
      self.W = zeros(M, M, self.E); self.beta = zeros(M, self.E);
      for i=1:self.E
        h = self.hyp(i);
        u = bsxfun(@times, self.induce(:,:,min(i,pE)), exp(-h.l'));
        x = bsxfun(@times, self.inputs, exp(-h.l'));
        Kmm = exp(2*h.s-maha(u,u)/2) + ridge*eye(M);                % add ridge
        Kmn = exp(2*h.s-maha(u,x)/2);
        sqrtKmm = chol(Kmm)';
        sqrtQ = sqrtKmm\Kmn;
        if strcmp(self.inf_method,'fitc')
          Lambda = exp(2*h.s)-sum(sqrtQ.^2,1);     % Lambda's diagonal elements
          if isprop(gp,'nigp'); Lambda = Lambda + self.nigp(:,i)'; end
          Gamma = 1+Lambda/exp(2*h.n); % Gamma
          sqrtGamma = sqrt(Gamma);
        elseif strcmp(self.inf_method,'vfe')
          sqrtGamma = ones(1,n);          % Lambda = zeros(1,n) in 'vfe' method
        else
          error('bad inf_method value')
        end
        V = bsxfun(@rdivide,sqrtQ,sqrtGamma);                   % sqrt(Q*Gamma)
        Am = chol(exp(2*h.n)*eye(M) + V*V')'; % sqrt(noise*eye + sqrtQ/Gamma*sqrtQ')
        At = sqrtKmm*Am;      % sqrt(noise*Kmm + Kmn/Gamma*Knm) = sqrt(noise*B)
        iAt = At\eye(M);                                 % inv(sqrt(noise * B))
        self.beta(:,i) = ((Am\(bsxfun(@rdivide,V,sqrtGamma)))'*iAt)'*y(:,i);
        iB = iAt'*iAt.*exp(2*h.n);                                     % inv(B)
        self.W(:,:,i) = Kmm\eye(M) - iB;          % cov matrix for predictive var
      end
    end
    
    function pre2(self)
      % alternate precomputation which converts sparse model to simple GP,
      % should be called after sparse-method and pre.
      mu0 = 0*self.target;     % mu0 is posterior mean, excluding mean function
      for i = 1:size(self.inputs,1)
        mu0(i,:) = self.pred(self.inputs(i,1:self.D)', zeros(self.D))' - ...
                                  self.inputs(i,:)*[self.hyp.m] - [self.hyp.b];
      end
      ridge = 1e-6;
      [np, ~, pE] = size(self.induce);
      for i=1:self.E 
        h = self.hyp(i);
        z = bsxfun(@times, self.induce(:,:,min(i,pE)), exp(-h.l'));
        x = bsxfun(@times, self.inputs, exp(-h.l'));
        Kmm = exp(2*h.s-maha(z,z)/2) + exp(2*h.n)*eye(np);          % add noise
        Kmn = exp(2*h.s-maha(z,x)/2);
        self.beta(:,i) = (Kmn*Kmn'+ridge*eye(np))\(Kmn*mu0(:,i));
        L = chol(Kmm);
        self.W(:,:,i) = L'\L\(eye(np));
      end
      self.inputs = self.induce;
      self.induce = zeros(size(self.induce,1),0,1);
    end
    
    function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds] = pred(self, m, s)
      % PRED computes GP predictions and optionally derivatives with uncertain,
      % possibly angular inputs. The function is a wrapper which deals with the
      % angular variables and then calls generic gp code to do the actual work.
      %
      % self
      %   angi             vector of indices of inputs which are angular
      %   approxS          binary switch; if it exists, approximate S in calc
      % m          D x 1   mean of the input distribution
      % s          D x D   covariance matrix of the input distribution
      % M          E x 1   mean of predictive distribution
      % S          E x E   covariance of predictive distribution
      % C          D x E   inv(s) times covariance between input and output
      % dMdm       E x D   deriv of output mean w.r.t. input mean
      % dSdm      E*Ex D   deriv of output covariance w.r.t input mean
      % dCdm      D*Ex D   deriv of input-output cov w.r.t. input mean
      % dMds       E xD*D  deriv of ouput mean w.r.t input covariance
      % dSds      E*ExD*D  deriv of output cov w.r.t input covariance
      % dCds      D*ExD*D  deriv of inv(s)*input-output cov w.r.t input cov
       
      if nargout < 4
        [q, r, u] = gTrigN(m, s, self.angi);
      else
        [q, r, u, dqdm, drdm, dudm, dqds, drds, duds] = gTrigN(m, s, self.angi);
      end
      
      switch (nargout > 3) + 2*(1-isempty(self.approxS))
        case 0                                         % no derivatives, full S
          [M, S, V] = gpp(self, q, r);
        case 1                                            % derivatives, full S
          [M, S, V, dMdq, dSdq, dCdq, dMdr, dSdr, dCdr] = gpd(self, q, r);
        case 2                                       % no derivatives, approx S
          [M, S, V] = gpas(gp, q, r);
        case 3                                          % derivatives, approx S
          [M, S, V, dMdq, dSdq, dCdq, dMdr, dSdr, dCdr] = gpasd(gp, q, r);
      end
      
      C = u*V;                           % multiply together covariance factors
      
      if nargout > 3              % if derivatives desired, chain them together
        dMdm = dMdq*dqdm + dMdr*drdm;
        dSdm = dSdq*dqdm + dSdr*drdm;
        dCdm = prodd(u, dCdq*dqdm + dCdr*drdm) + prodd([], dudm, V);
        dMds = dMdq*dqds + dMdr*drds;
        dSds = dSdq*dqds + dSdr*drds;
        dCds = prodd(u, dCdq*dqds + dCdr*drds) + prodd([], duds, V);
      end
    end                                                         % function pred

    function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, ...
                                                       dCdp] = preD(self, m, s)
      % gpD computes GP predictions with uncertain, possibly angular inputs and
      % optionally derivatives. The function is a wrapper which deals with the
      % angular variables and then calls generic gp code to do the actual work.
      %
      % self
      %   angi             vector of indices of inputs which are angular
      %   idx              index into parameters
      % m          D x 1   mean of the input distribution
      % s          D x D   covariance matrix of the input distribution
      % M          E x 1   mean of predictive distribution
      % S          E x E   covariance of predictive distribution
      % C          D x E   inv(s) times covariance between input and output
      % dMdm       E x D   deriv of output mean w.r.t. input mean
      % dSdm      E*Ex D   deriv of output covariance w.r.t input mean
      % dCdm      D*Ex D   deriv of input-output cov w.r.t. input mean
      % dMds       E xD*D  deriv of ouput mean w.r.t input covariance
      % dSds      E*ExD*D  deriv of output cov w.r.t input covariance
      % dCds      D*ExD*D  deriv of inv(s)*input-output cov w.r.t input cov
      
      if nargout < 4
        [q, r, u] = gTrigN(m, s, self.angi);
        [M, S, V] = gpD(self, q, r);
      else
        [q, r, u, dqdm, drdm, dudm, dqds, drds, duds] = gTrigN(m, s, self.angi);
        [M, S, V, dMdq, dSdq, dVdq, dMdr, dSdr, dVdr, dMdp, dSdp, dVdp] = ...
                                                                gpD(self, q, r);
      end
      
      C = u*V;                           % multiply together covariance factors
      
      if nargout > 3              % if derivatives desired, chain them together
        dMdm = dMdq*dqdm + dMdr*drdm;
        dSdm = dSdq*dqdm + dSdr*drdm;
        dCdm = prodd(u, dVdq*dqdm + dVdr*drdm) + prodd([], dudm, V);
        dMds = dMdq*dqds + dMdr*drds;
        dSds = dSdq*dqds + dSdr*drds;
        dCds = prodd(u, dVdq*dqds + dVdr*drds) + prodd([], duds, V);
        dCdp = prodd(u, dVdp);
        i = [self.idx.m]; i = i(self.D+(1:2*length(self.angi)),:);
        dMdp(:,i) = 0; dSdp(:,i) = 0; dCdp(:,i) = 0;  % zero linear trig derivs
      end
    end                                                         % function pred
        
    function [M, S, C, V, dMdm, dSdm, dCdm, dVdm, dMds, dSds, dCds, dVds, ...
        dMdv, dSdv, dCdv, dVdv] = predh(self, m, s, v, combineSV, approxSV)
      % PREDH computes GP predictions with Hierarchically-uncertain inputs.
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
      % m         D x 1    mean       of mean of test input
      % s         D x D    covariance of mean of test input
      % v         D x D    covariance of         test input
      % combineSV bool     output S (not V) where S is instead S+V
      % approxSV  bool     compute an inexpensive-and-approximate S and V
      % M         E x 1    mean     or mean of mean of prediction
      % S         E x E    covariance variance of mean of prediction
      % C         D x E    inv(s) times input output covariance (of mean)
      % V         E x E    mean of covariance of prediction
      % dMdm      E x D    deriv of output mean wrt input mean-of-mean
      % dSdm    E*E x D    deriv of output var-of-mean wrt input mean-of-mean
      % dCdm    D*E x D    deriv of inv(s) input-output cov wrt input mean-of-mean
      % dVdm    E*E x D    deriv of output mean-of-var wrt input mean-of-mean
      % dMds      E x D*D  deriv of output mean wrt input var-of-mean
      % dSds    E*E x D*D  deriv of output var-of-mean wrt input var-of-mean
      % dCds    D*E x D*D  deriv of inv(s) input-output cov wrt input var-of-mean
      % dVds    E*E x D*D  deriv of output mean-of-var wrt input var-of-mean
      % dMdv      E x D*D  deriv of output mean wrt input var
      % dSdv    E*E x D*D  deriv of output var-of-mean wrt input var
      % dCdv    D*E x D*D  deriv of inv(s) input-output cov wrt input var
      % dVdv    E*E x D*D  deriv of output mean-of-var wrt input var
      
      if nargin < 3; s = zeros(numel(m)); end
      if nargin < 4; v = zeros(numel(m)); end
      if nargin < 5; combineSV = false; end
      if nargin < 6; approxSV = false; end
      
      if nargout <= 4
        
        [ma, sa, ca, va] = gTrighN(m, s, v, self.angi);
        [M, S, CC, V] = gph(self, ma, sa, va, combineSV, approxSV);
        C = ca*CC;
        
      else
        
        [ma, sa, ca, va, dmadm, dsadm, dcadm, dvadm, dmads, dsads, dcads, ...
          dvads, dmadv, dsadv, dcadv, dvadv] = gTrighN(m, s, v, self.angi);
        
        [M, S, CC, V, dMdma, dSdma, dCdma, dVdma, dMdsa, dSdsa, dCdsa, ...
          dVdsa, dMdva, dSdva, dCdva, dVdva] = ...
          gphd(self, ma, sa, va, combineSV, approxSV);
        
        C = ca*CC;
        
        dMdm = dMdma*dmadm + dMdsa*dsadm + dMdva*dvadm;
        dSdm = dSdma*dmadm + dSdsa*dsadm + dSdva*dvadm;
        dVdm = dVdma*dmadm + dVdsa*dsadm + dVdva*dvadm;
        Cdm  = dCdma*dmadm + dCdsa*dsadm + dCdva*dvadm;
        dCdm = prodd(ca,Cdm) + prodd([],dcadm,CC);
        
        dMds = dMdma*dmads + dMdsa*dsads + dMdva*dvads;
        dSds = dSdma*dmads + dSdsa*dsads + dSdva*dvads;
        dVds = dVdma*dmads + dVdsa*dsads + dVdva*dvads;
        Cds  = dCdma*dmads + dCdsa*dsads + dCdva*dvads;
        dCds = prodd(ca,Cds) + prodd([],dcads,CC);
        
        dMdv = dMdma*dmadv + dMdsa*dsadv + dMdva*dvadv;
        dSdv = dSdma*dmadv + dSdsa*dsadv + dSdva*dvadv;
        dVdv = dVdma*dmadv + dVdsa*dsadv + dVdva*dvadv;
        Cdv  = dCdma*dmadv + dCdsa*dsadv + dCdva*dvadv;
        dCdv = prodd(ca,Cdv) + prodd([],dcadv,CC);
      end
      
    end
    
    function gpa_copy = deepcopy(self)
      p = properties(self);
      gpa_copy = gpa(self.D, self.E, self.angi);
      for i = 1:numel(p)
        if any(strcmp(p{i},{'D', 'E', 'angi'})); continue; end % skip read-onlys
        gpa_copy.(p{i}) = self.(p{i});
      end
    end
    
  end                                                             % end methods
  
  methods (Static)
    
    function [gp,i] = combine(gp1,gp2,i,c)
      % COMBINE outputs one GP which is a combination of two inputted GPs.
      % Both input dimensionalities must agree.
      % i : 2 x D indices that calling fn uses to index both gp1 and gp2 inputs
      % c :       (optional) common indices of variables that gp1 and gp2 share
      %           note: c cannot yet handle angi variables.
      
      D = gp1.D;  %#ok<*PROP> % size of GP input (no trig)
      assert(D==gp2.D);  assert(D==size(i,2)); 
      Dt = size(gp1.inputs,2); % size of GP input (inc. trig)
      assert(Dt == size(gp2.inputs,2), 'GPs inconsisent input dimensionality.')
      u = setdiff(1:D,c); % unique variables (non-common)
      t = D+1:Dt; % trig variables
      lt = length(t);
      lu = length(u);
      if nargin < 4
        c = [];
        warning('gpa: combining two gpa models of with NO common variables.');
      end
      
      % 1. hyps:
      h1 = gp1.hyp; h2 = gp2.hyp;
      zu = zeros(lu,1); iu = inf(lu,1);
      zt = zeros(lt,1); it = inf(lt,1);
      for e=1:gp1.E
        h1(e).l=[h1(e).l(u) ; iu ; h1(e).l([c,t]); it];
        h1(e).m=[h1(e).m(u) ; zu ; h1(e).m([c,t]); zt];
      end
      for e=1:gp2.E
        h2(e).l=[iu ; h2(e).l([u,c]); it ; h2(e).l(t)];
        h2(e).m=[zu ; h2(e).m([u,c]); zt ; h2(e).m(t)];
      end
      
      % 2. inputs:                                                              % TODO handle nx ~= nz. Perhaps by zeroing corresponding beta values.
      x1 = gp1.inputs; [N1,~,pE1] = size(x1);
      x1 = cat(2,x1(:,u,:),zeros(N1,lu,pE1),x1(:,[c,t],:),zeros(N1,lt,pE1));
      x1 = repmat(x1,[1,1,gp1.E/pE1]);
      x2 = gp2.inputs; [N2,~,pE2] = size(x2);
      x2 = cat(2,zeros(N2,lu,pE2),x2(:,[u,c],:),zeros(N2,lt,pE2),x2(:,t,:));
      x2 = repmat(x2,[1,1,gp2.E/pE2]);
      
      % new combined GP object
      D  = 2*D - numel(c); % input dim of new GP
      E = gp1.E + gp2.E;
      a = zeros(1,D); a(gp1.angi) = 1; a(c) = []; angi1 = find(a);
      a = zeros(1,D); a(gp2.angi) = 1; a(c) = []; angi2 = find(a);
      angi = [angi1, lu+angi2];
      gp = gpa(D, E, angi);
      gp.hyp    = [h1, h2];                  % 1 x (E1+E2)
      gp.inputs = cat(3,x1,x2);              % nxDx(E1+E2)                      % TODO 'induce' points too?
      gp.W      = cat(3,gp1.W,gp2.W);        % nxnx(E1+E2)
      gp.beta   = cat(2,gp1.beta,gp2.beta);  % n x (E1+E2)
      
      i = [i(1,u),i(2,u),i(1,c)]; % only need one set of common variables
    end
    
  end                                                      % end static methods
  
end                                                                 % end class
