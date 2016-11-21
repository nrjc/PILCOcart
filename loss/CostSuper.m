classdef CostSuper < handle
  %COSTSUPER, cost superclass, which scenario-specific cost subclass inherits.
  % The cost function is the saturating function: 1 - exp(-(x-z)'*W*(x-z)/2).
  % The role of each subclass is to define the mean z, lengthscales W, and
  % augment and trigonometric variables within x (note the augment and trig
  % variables are specific to the cost function, and not necessarily the same
  % as the plant.augi and plant.angi variables). The role of this superclass is
  % to compute the cost distribution given x ~ N(m,S).
  %
  % CostSuper Properties:
  %   cangi     - indicies of Current (non-old) vars as angles (sin/cos rep)
  %   D         - dimensionality of state
  %   Da        - dimensionality of state + augmented variables
  %   Dt        - dimensionality of state + augmented + trig variables
  %   MAX_COST  - the maximum instantaneous cost
  %   zW        - lossSat inputs computed by pre()
  %
  % CostSuper Methods:
  %   augment   - a default function (which only does bookkeeping), to be
  %               overwritten by cost-subclass if augment variable(s) exist
  %   augmenth  - a hiehierarchical augment function
  %   CostSuper - constructor
  %   cov       - computes the covariance of costs between two random states
  %   fcn       - computes the cost mean and variance of a random state
  %   fcnh      - computes the cost of a hierarchically random state
  %   fillIn    - computes cross S terms, and performs derivative chain rule
  %
  % Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2016-03-15
  
  properties (SetAccess = protected)
    cangi
    D
    Da
    Dt
    MAX_COST = 1.0
    zW = {}
  end
  
  methods
    
    % Constructor
    function self = CostSuper(D)
      self.D = D;
    end
    
    function [L, dL] = fcn(self, s)
      % COSTSUPER.FCN, the main function which returns a cost-mean and
      % cost-variance in range [0,1] (inc. derivatives w.r.t. input state)
      %
      % The function first takes the input state, augments the input state,
      % then computes any trig terms of the original and/or augmented variables
      % before inputting to the 'lossSat' function. In addition, the 'zW' self
      % variables are also passed into 'lossSat' which define the mean and width
      % of the quadratic term in the lossSat function.
      %
      % [L, dLds] = fcn(self, s)
      %
      % s        .    state structure
      %   m      Fx1  mean of state distribution
      %   s      FxF  covariance matrix for the state distribution
      % L        .    cost structure
      %   m      1x1  cost mean scalar
      %   s      1x1  cost variance scalar
      %   c      Dx1  cost inverse-input-variance times input-output covariance
      % dL       .    derivative of cost structure L w.r.t. state structure
      %   m      1x(D+DD) derivative of L.m w.r.t. s.m and s.s
      %   s      1x(D+DD) derivative of L.s w.r.t. s.m and s.s
      %   c      Dx(D+DD) derivative of L.c w.r.t. s.m and s.s
      
      D = self.D;  %#ok<*PROPLC>
      Da = self.Da;
      Dt = self.Dt;
      s.m = s.m(1:D);         % extract the real (non-filter) part of the state
      if isfield(s,'s'); ss = s.s(1:D,1:D); else ss = zeros(D); end
      derivatives_requested = nargout > 1;
      
      % augment variables
      if ~derivatives_requested
        [M, S, Ca] = self.augment(s.m, ss);
      else
        [M, S, Ca, Mdm, Mds, Sdm, Sds, Cadm, Cads] = self.augment(s.m, ss);
      end
      
      % trigonometric variables
      i = 1:Da; k = Da+1:Dt;
      if ~derivatives_requested
        [M(k), S(k,k), Cg] = gTrig(M(i), S(i,i), self.cangi);
        S = CostSuper.fillIn(i,k,Dt,S,Cg);
      else
        [M(k), S(k,k), Cg, mdm, sdm, Cgdm, mds, sds, Cgds] = gTrig(M(i),S(i,i),self.cangi);
        [S, Mdm, Sdm, Mds, Sds] = ...
          CostSuper.fillIn(i,k,Dt,S,Cg,mdm,sdm,Cgdm,mds,sds,Cgds,Mdm,Sdm,Mds,Sds);
        ii = sub2ind2(Dt,i,i); assert(numel(M) == Dt && numel(S) == Dt*Dt);
        dCgdm = Cgdm*Mdm(i,:) + Cgds*Sdm(ii,:); dCgds = Cgdm*Mds(i,:) + Cgds*Sds(ii,:);
      end
      
      % compute lossSat
      Lm = 0; dLmdm = zeros(1,D); dLmds = zeros(1,D*D);
      Ls = 0; dLsdm = zeros(1,D); dLsds = zeros(1,D*D);
      Lc = zeros(D,1); dLcdm = zeros(D,D); dLcds = zeros(D,D*D);
      I = eye(Da);
      dIdm = zeros(Da*Da,D);                                                    % TODO: fix, this is wasteful
      dIds = zeros(Da*Da,D*D);                                                  % TODO: fix, this is wasteful
      n = length(self.width);
      for j = 1:n                            % scale mixture of immediate costs
        [r, rdM, rdS, s2, s2dM, s2dS, Cl, dCldM, dCldS] = lossSat(self.zW{j},M,S);
        
        Lm = Lm + r;
        Ls = Ls + s2;
        Lc = Lc + Ca*[I,Cg]*Cl;
        if derivatives_requested
          dLmdm = dLmdm + rdM*Mdm + rdS(:)'*Sdm;
          dLmds = dLmds + rdM*Mds + rdS(:)'*Sds;
          dLsdm = dLsdm + s2dM*Mdm + s2dS(:)'*Sdm;
          dLsds = dLsds + s2dM*Mds + s2dS(:)'*Sds;
          dCldm = dCldM*Mdm + dCldS*Sdm;
          dClds = dCldM*Mds + dCldS*Sds;
          dLcdm = dLcdm + prodd([],Cadm,[I,Cg]*Cl) + ...
            prodd(Ca,catd(2,dIdm,dCgdm),Cl) + prodd(Ca*[I,Cg],dCldm);
          dLcds = dLcds + prodd([],Cads,[I,Cg]*Cl) + ...
            prodd(Ca,catd(2,dIds,dCgds),Cl) + prodd(Ca*[I,Cg],dClds);
        end
      end
      
      % normalize
      L.m = Lm/n;
      L.s = Ls/n;
      L.c = Lc/n;
      if derivatives_requested
        dL.m = [dLmdm, dLmds] / n;
        dL.s = [dLsdm, dLsds] / n;
        dL.c = [dLcdm, dLcds] / n;
      end
    end
    
    function [L, dL] = fcnh(self, s)                                      % TODO: finish writing + debugging.
      % COSTSUPER.FCNH, a hierarchical version of the main function which
      % returns a cost-mean-of-mean, a cost-variance in range [0,1] (inc. derivatives w.r.t. input state)
      %
      % The function first takes the input state, augments the input state,
      % then computes any trig terms of the original and/or augmented variables
      % before inputting to the 'lossSat' function. In addition, the 'zW' self
      % variables are also passed into 'lossSat' which define the mean and width
      % of the quadratic term in the lossSat function.
      %
      % [L, dLds] = fcnh(self, m, s, v)
      %
      % s        .    state structure
      %   m      Fx1  mean of state distribution
      %   s      FxF  covariance matrix for the state distribution
      %   v      FxF  mean-of-variance of the state distribution
      % L        .    cost structure
      %   m      1x1  cost mean-of-mean
      %   s      1x1  cost variance-of-mean
      %   c      Dx1  cost inverse-input-variance times input-output covariance
      %   v      DxD  cost mean-of-variance
      % dL       .    derivative of cost structure L w.r.t. state structure
      %   m      1x(D+2DD) derivative of L.m w.r.t. s.m and s.s and s.v
      %   s      1x(D+2DD) derivative of L.s w.r.t. s.m and s.s and s.v
      %   c      Dx(D+2DD) derivative of L.c w.r.t. s.m and s.s and s.v
      %   v      1x(D+2DD) derivative of L.v w.r.t. s.m and s.s and s.v
      
      D = self.D;  %#ok<*PROPLC>
      Da = self.Da;
      Dt = self.Dt;
      derivatives_requested = nargout > 1;
      s.m = s.m(1:D);         % extract the real (non-filter) part of the state
      if isfield(s,'s'); s.s = s.s(1:D,1:D); else s.s = zeros(D); end
      if isfield(s,'v'); s.v = s.v(1:D,1:D); else s.v = zeros(D); end
      
      % augment variables
      if ~derivatives_requested
        [M, S, Ca, V] = self.augmenth(s);
      else
        % setup some derivative matrix support:
        nx = length(unwrap(s));  % TODO: should Ctrl.m be doing this?           % TODO: required?
        % is = rewrap(s,1:nx);     % TODO: should Ctrl.m be doing this?           % TODO: required?
        % assert(nx == D + 2*D*D); % TODO: handle if false.
        
        % Derivative 'dx' means [dm, ds, dv].
        [M, S, Ca, V, Mdx, Sdx, Cadx, Vdx] = self.augmenth(s);
      end
      
      % trigonometric variables
      i = 1:Da; k = Da+1:Dt;
      if ~derivatives_requested
        [M(k), S(k,k), Cg, V(k,k)] = gTrigh(M(i), S(i,i), V(i,i), self.cangi);
        [S, V] = CostSuper.fillInh(i,k,Dt,S,Cg,V);
      else
        [M(k), S(k,k), Cg, V(k,k), mdm, sdm, Cgdm, vdm, mds, sds, Cgds, vds, mdv, sdv, Cgdv, vdv] = ...
          gTrigh(M(i), S(i,i), V(i,i), self.cangi);
        [S, V, Mdx, Sdx, Vdx] = CostSuper.fillInh(i,k,Dt,S,Cg,V, ...
          mdm,sdm,vdm,Cgdm,mds,sds,vds,Cgds,mdv,sdv,vdv,Cgdv,Mdx,Sdx,Vdx);
        
        ii = sub2ind2(Dt,i,i); assert(numel(M) == Dt && numel(S) == Dt*Dt);
        dCgdx = Cgdm*Mdx(i,:) + Cgds*Sdx(ii,:) + Cgdv*Vdx(ii,:);
      end
      
      % compute lossSat
      Lm = 0;
      Ls = 0;
      Lc = zeros(D,1);
      Lv = 0;
      I = eye(Da);
      if derivatives_requested
        dLmdx = zeros(1,nx);
        dLsdx = zeros(1,nx);
        dLcdx = zeros(D,nx);
        dLvdx = zeros(1,nx);
        dIdx = zeros(Da*Da,nx);                                                 % TODO: fix, this is wasteful
      end
      
      n = length(self.width);
      for j = 1:n                            % scale mixture of immediate costs
        [Lmj, LmjdM, LmjdS, LmjdV, Lsj, LsjdM, LsjdS, LsjdV, Cl, dCldM, ...
          dCldS, dCldV, Lvj, LvjdM, LvjdS, LvjdV] = lossSath(self.zW{j},M,S,V);
        
        Lm = Lm + Lmj;
        Ls = Ls + Lsj;
        Lc = Lc + Ca*[I,Cg]*Cl;                                                 % TODO: unsure if correct in this hierarchical case?
        Lv = Lv + Lvj;
        if derivatives_requested
          dLmdx = dLmdx + LmjdM*Mdx + LmjdS*Sdx + LmjdV*Vdx;
          dLsdx = dLsdx + LsjdM*Mdx + LsjdS*Sdx + LsjdV*Vdx;
          dCldx = dCldM*Mdx + dCldS*Sdx + dCldV*Vdx;
          dLcdx = dLcdx + prodd([],Cadx,[I,Cg]*Cl) + ...
            prodd(Ca,catd(2,dIdx,dCgdx),Cl) + prodd(Ca*[I,Cg],dCldx);
          dLvdx = dLvdx + LvjdM*Mdx + LvjdS*Sdx + LvjdV*Vdx;
        end
      end
      
      % normalize
      L.m = Lm/n;
      L.s = Ls/n;
      L.c = Lc/n;
      L.v = Lv/n;
      if derivatives_requested
        dL.m = dLmdx / n;
        dL.s = dLsdx / n;
        dL.c = dLcdx / n;
        dL.v = dLvdx / n;
      end
      
    end
    
    function [M, S, C, Mdm, Mds, Sdm, Sds, Cdm, Cds] = augment(self, m, s)
      % COSTSUPER.AUGMENT, implements a 'default' augment function which does
      %  not augment any variables, but only does bookkeeping. This function is
      %  only used if no augment variables exits (augment variables are linear
      %  combinations of state variables), in which case the subclass will not
      %  have overridden this function. If the subclass does have augment
      %  variables (i.e. if self.Da > self.D), then the subclass will override
      %  this augment function with its own augment function. The bookkeeping
      %  involved here allocates space for joint mean and variance matrices
      %  ready for the trig variables (which exist if self.Dt > self.Da).
      %
      %  [M, S, C, Mdm, Mds, Sdm, Sds, Cdm, Cds] = augment(self, m, s)
      %
      %  m       Dx1    mean vector of state distribution
      %  s       DxD    covariance matrix for the state distribution
      %  M      Dtx1    augmented mean (and extra space for trig vars)
      %  S      DtxDt   augmented variance
      %  C       DxDt   inverse-input-variance times intput-output covariance
      %  Mdm    DtxD    derivative of M w.r.t. m
      %  Mds    DtxDD   derivative of M w.r.t. s
      %  Sdm  DtDtxD    derivative of S w.r.t. m
      %  Sds  DtDtxDD   derivative of S w.r.t. s
      %  Cdm   DDtxD    derivative of C w.r.t. m
      %  Cds   DDtxDD   derivative of C w.r.t. s
      
      assert(self.Da == self.D); % dummy fcn only called if no augment vars
      
      D = self.D; %#ok<*PROP>
      Dt = self.Dt;
      
      M = [m; nan(Dt-D,1)];
      S = blkdiag(s,nan(Dt-D));
      C = eye(D);
      if nargout > 3
        Mdm = eye(Dt,D);
        Mds = zeros(Dt,D*D);
        Sdm = zeros(Dt*Dt,D);
        Sds = symmetrised(kron(Mdm,Mdm),1);
        Cdm = zeros(D*D,D);
        Cds = zeros(D*D,D*D);
      end
    end
    
    function [M, S, C, V, Mdx, Sdx, Cdx, Vdx] = augmenth(self, s)
      % A hierarchical version of augment, based off the augment() function.
      % This function assumes:
      %   1) augment() is a linear function,
      %   2) derivatives are in order: m, s, v.
      
      [M, S, C] = augment(self, s.m, s.s);
      
      if nargout < 5 % no derivatives required
        [m, V, c] = augment(self, s.m, s.v);
      else
        [m, V, c, Mdm, Mds, Sdm, Sds, Cdm, Cds] = augment(self, s.m, s.v);
        
        % Derivative 'dx' means [dm, ds, dv].
        Mdx = [Mdm, Mds, Mds];
        Cdx = [Cdm, Cds, Cds];
        Sdx = [Sdm, Sds, 0*Sds];
        Vdx = [Sdm, 0*Sds, Sds];
      end
      
      % test linearity
      max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
      assert(max_diff({M,C},{m,c}) < 1e-10, ...
        'augmenth() only valid for linear augment()');
      if nargout > 4 % derivatives required
        assert(max_diff({Mds,Sdm},{0*Mds,0*Sdm}) < 1e-10, ...
          'augmenth() only valid for linear augment()');
      end
    end
    
    function [L, dLdsa, dLdsb, dLdc] = cov(self, sa, sb, c)
      % COSTSUPER.COV, implements the cost-covariance between two states,
      %  i.e.: Cov[cost(sa), cost(sb)], oweing to the covariance between random
      %  state sa and state sb.
      %
      % [L, dLdsa, dLdsb, dLdc] = cov(self, sa, sb, c)
      %
      % sa    .     state A structure
      %   m   Fx1   mean vector of state A distribution
      %   s   FxF   covariance matrix for the state A distribution
      % sb    .     state B structure
      %   m   Fx1   mean vector of state B distribution
      %   s   FxF   covariance matrix for the state B distribution
      % c     DxD   covariance matrix between state A and state B real vars
      % L     1x1   cost covariance scalar
      % dLdsa 1x(D+DD)   derivative of cost covariance w.r.t. [sa.m, sa.s]
      % dLdsb 1x(D+DD)   derivative of cost covariance w.r.t. [sb.m, sb.s]
      % dLdc  1xDD       derivative of cost covariance w.r.t. covariance c
      
      D = self.D; %#ok<*PROP>
      Da = self.Da;
      Dt = self.Dt;
      sa.m = sa.m(1:D);
      sb.m = sb.m(1:D);
      sa.s = sa.s(1:D,1:D);
      sb.s = sb.s(1:D,1:D);
      
      if ~isfield(sa,'s'), sa.s = zeros(D); end
      if ~isfield(sb,'s'), sb.s = zeros(D); end
      derivatives_requested = nargout > 1;
      
      M = [sa.m; sb.m];
      S = [sa.s , c ; c', sb.s];
      
      % augmented variables
      if ismethod(self,'augmentcov')
        if ~derivatives_requested
          [M, S] = self.augmentcov(M, S);
        else
          [M, S, Mdm, Mds, Sdm, Sds] = self.augmentcov(M, S);
        end
      end
      
      % trigonometric variables
      cangi2 = [self.cangi, Da+self.cangi]; nA = 2*length(self.cangi);
      a = 1:D; b = D+1:2*D;
      a0 = 1:Da; b0 = Da+1:2*Da;
      A = [a0,2*Da+(1:nA)]; B = [b0,2*Da+nA+(1:nA)]; N = numel([A,B]);
      aa = sub2ind2(2*D,a,a); ab = sub2ind2(2*D,a,b); bb = sub2ind2(2*D,b,b);
      AA = sub2ind2(N,A,A); AB = sub2ind2(N,A,B); BB = sub2ind2(N,B,B);
      if ismethod(self,'augmentcov')
        i = 1:2*Da; k = 2*Da+1:2*Dt;
        if ~derivatives_requested
          [M(k), S(k,k), C] = gTrig(M(i), S(i,i), cangi2);
          S = CostSuper.fillIn(i,k,2*Dt,S,C);
        else
          [M(k), S(k,k), C, mdm, sdm, cdm, mds, sds, cds] = ...
            gTrig(M(i),S(i,i),cangi2);
          [S, Mdm, Sdm, Mds, Sds] = CostSuper.fillIn(i,k,2*Dt,S,C,mdm,sdm,...
            cdm,mds,sds,cds,Mdm,Sdm,Mds,Sds);
        end
      else
        if ~derivatives_requested
          [M, S] = gTrigN(M, S, cangi2);
        else
          [M, S, ~, Mdm, Sdm, ~, Mds, Sds] = gTrigN(M, S, cangi2);
        end
      end
      
      % cov cost
      L = 0;
      if derivatives_requested;
        Sds = symmetrised(Sds,1);
        dLdc = zeros(1,D*D);
        dLdma = zeros(1,D); dLdSa = zeros(1,D*D);
        dLdmb = zeros(1,D); dLdSb = zeros(1,D*D);
      end
      
      n = length(self.zW);
      for j = 1:n                            % scale mixture of immediate costs
        if ~derivatives_requested
          f = costcovSat(self.zW{j},M(A),S(A,A),M(B),S(B,B),S(A,B));
        else
          [f, dfdmA, dfdsA, dfdmB, dfdsB, dfdc] = ...
            costcovSat(self.zW{j},M(A),S(A,A),M(B),S(B,B),S(A,B));
          
          dLdma = dLdma + dfdmA*Mdm(A,a)  + dfdsA*Sdm(AA,a)  + dfdc*Sdm(AB,a);
          dLdSa = dLdSa + dfdmA*Mds(A,aa) + dfdsA*Sds(AA,aa) + dfdc*Sds(AB,aa);
          dLdmb = dLdmb + dfdmB*Mdm(B,b)  + dfdsB*Sdm(BB,b)  + dfdc*Sdm(AB,b);
          dLdSb = dLdSb + dfdmB*Mds(B,bb) + dfdsB*Sds(BB,bb) + dfdc*Sds(AB,bb);
          dLdc = dLdc + dfdc*Sds(AB,ab);
        end
        L = L + f;
      end
      
      L = L/n;                                                      % normalize
      if derivatives_requested
        dLdsa = [dLdma,dLdSa]/n;
        dLdsb = [dLdmb,dLdSb]/n;
        dLdc = 2*dLdc/n; % 2 cross-covariances
      end
    end
    
  end
  
  methods (Static)
    
    % Fill in covariance matrix...and derivatives ----------------------------
    function [S, Mdm, Sdm, Mds, Sds] = ...                                      % TODO: add as a standalone file in /util.
        fillIn(i,k,D,S,C,mdm,sdm,cdm,mds,sds,cds,Mdm,Sdm,Mds,Sds)
      S(i,k) = S(i,i)*C; S(k,i) = S(i,k)';                       % off-diagonal
      if nargout < 2; return; end
      
      X = reshape(1:D*D,[D D]); XT = X';                   % vectorised indices
      I=0*X; I(i,i)=1; ii=X(I==1)'; I=0*X; I(k,k)=1; kk=X(I==1)';
      I=0*X; I(i,k)=1; ik=X(I==1)'; ki=XT(I==1)';
      
      Mdm(k,:)  = mdm*Mdm(i,:) + mds*Sdm(ii,:);                     % chainrule
      Sdm(kk,:) = sdm*Mdm(i,:) + sds*Sdm(ii,:);
      Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:);
      Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:);
      dCdm      = cdm*Mdm(i,:) + cds*Sdm(ii,:);
      dCds      = cdm*Mds(i,:) + cds*Sds(ii,:);
      
      Sdm(ik,:) = prodd(S(i,i),dCdm) + prodd([],Sdm(ii,:),C);
      Sdm(ki,:) = Sdm(ik,:);
      Sds(ik,:) = prodd(S(i,i),dCds) + prodd([],Sds(ii,:),C);
      Sds(ki,:) = Sds(ik,:);
    end
    
    function [S, V, Mdx, Sdx, Vdx, Cdx] = fillInh(i,k,D,S,C,V, ...              % TODO: (modified from CtrlBF) add as a standalone file in /util.
        mdm,sdm,vdm,cdm,mds,sds,vds,cds,mdv,sdv,vdv,cdv, ...
        Mdx,Sdx,Vdx)
      
      S(i,k) = S(i,i)*C; S(k,i) = S(i,k)';                       % off-diagonal
      V(i,k) = V(i,i)*C; V(k,i) = V(i,k)';                       % off-diagonal
      if nargout <= 2; return; end
      
      if isempty(k); return; end    % if no trig, then inputs-outputs unchanged
      if isempty(vdm), vdm=0*sdm; end; if isempty(vds), vds=0*sds; end          % TODO: required?
      if isempty(mdv), mdv=0*mds; end; if isempty(sdv), sdv=0*sds; end
      if isempty(vdv), vdv=0*sds; end; if isempty(cdv), cdv=0*cds; end
      
      X = reshape(1:D*D,[D D]); XT = X';                   % vectorised indices
      I=0*X; I(i,i)=1; ii=X(I==1)'; I=0*X; I(k,k)=1; kk=X(I==1)';
      I=0*X; I(i,k)=1; ik=X(I==1)'; ki=XT(I==1)';
      
      Mdx(k,:)  = mdm*Mdx(i,:) + mds*Sdx(ii,:) + mdv*Vdx(ii,:);     % chainrule
      Sdx(kk,:) = sdm*Mdx(i,:) + sds*Sdx(ii,:) + sdv*Vdx(ii,:);
      Vdx(kk,:) = vdm*Mdx(i,:) + vds*Sdx(ii,:) + vdv*Vdx(ii,:);
      Cdx       = cdm*Mdx(i,:) + cds*Sdx(ii,:) + cdv*Vdx(ii,:);
      
      Sdx(ik,:) = prodd(S(i,i),Cdx) + prodd([],Sdx(ii,:),C);
      Sdx(ki,:) = Sdx(ik,:);
      Vdx(ik,:) = prodd(V(i,i),Cdx) + prodd([],Vdx(ii,:),C);
      Vdx(ki,:) = Vdx(ik,:);
    end
    
  end
  
end