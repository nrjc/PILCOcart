classdef Ctrl < handle
  %CTRL, controller superclass, which ctrlBF and ctrlNF inherit.
  %
  % Ctrl Properties:
  %   actuate       -  @      (optional) call this function with the action
  %   angi          -         indicies for vars treated as angles (sin/cos rep)
  %   D             -         number of physical state variables
  %   Dz            -         number of infomation state variables
  %   dyn           -  .      dynamics model object (only used by CTRLBF class)
  %   E             -         number of predicted state variables
  %   F             -         number of state variables (physical + infomation)
  %   is            -  .      state index structure
  %   mu0           -  D x 1  initial state mean
  %   np            -         number of policy parameters
  %   ns            -         state parameters (including filter)
  %   on            -  D x D  non-log observation noise
  %   onp           -  D x D  non-log obs. noise (only last E vars non zeros)
  %   poli          -         indicies for the inputs to the policy
  %   policy        -         policy struct
  %   S0            -  D x D  initial state variance
  %   U             -         number of control outputs
  %
  % Ctrl Methods:
  %   build_state_index  -   (private) builds 'is' field to index state-struct
  %   clear_filter       -   clears all filter variables (if they exist)
  %   Ctrl               -   constructor
  %   fcn                -   main function, computes control signal
  %   random_action      -   outputs a random control, independent of state
  %   reset_filter       -   resets state fields {zs,zc,v} to a broad prior
  %   set_dynmodel       -   sets the dynnamics model object
  %   set_on             -   sets the N-Markov observations noise
  %   set_policy_opt     -   sets the policy optimisation settings
  %   set_policy_p       -   sets the policy parameters
  %
  % See also CTRLBF.M, CTRLNF.M.
  % Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2016-01-19
  
  properties % (SetAccess = private)
    actuate
    angi
    D
    Dz
    dyn
    E
    F
    is
    mu0
    np
    ns
    on
    onp
    poli
    policy
    S0
    U
  end
  
  methods
    
    % Constructor
    function self = Ctrl(D, E, policy, angi, poli, mu0, S0, actuate)
      % CTRL is the super-class controller constructor
      %
      % ctrl = Ctrl(D, E, policy, angi, poli, mu0, S0, actuate)
      %
      % INPUTS:
      %   D                 number of physical state variables
      %                     (or a ctrl object to be copied)
      %   E                 number of predicted state variables
      %   policy     .      policy struct
      %     fcn      @      policy function
      %     maxU            maximum control output magnitudes
      %     opt      .      optimisation structure
      %       fh            figure handle for minimize() to display to
      %       length        how many optimisations steps
      %       method
      %       verbosity
      %   angi              indicies for variables treated as angles
      %   poli              indicies for the inputs to the policy
      %   mu0        D x 1  initial state mean
      %   S0         D x D  initial state variance
      %   actuate    @      function to actuate calculated action
      
      % Special case input:
      % Another controller object might be the first (and only) input.
      % This allows easy translations from one controller type to the next
      if ~isnumeric(D); assert(nargin == 1); ctrl = D;
        D = ctrl.D;
        E = ctrl.E;
        policy = ctrl.policy;
        angi = ctrl.angi;
        poli = ctrl.poli;
        mu0 = ctrl.mu0;
        S0 = ctrl.S0;
        actuate = ctrl.actuate;
        self.on = ctrl.on;
        self.onp = ctrl.onp;
        self.dyn = ctrl.dyn;
      end
      
      self.D = D;
      self.E = E;
      self.policy = policy;
      if ~isfield(policy,'opt'); self.policy.opt = ...
          struct('length',-1000,'method','BFGS','MFEPLS',20,'verbosity',3); end
      if ~isfield(self.policy.opt,'fh'); self.policy.opt.fh = 1; end
      self.U = length(policy.maxU);
      if exist('mu0','var'); self.mu0 = mu0; else self.mu0 = nan(D,1); end
      if exist('S0','var'); self.S0 = S0; else self.S0 = nan(D); end
      self.build_state_index(); % computes Dz, F, is, ns
      if ~isfield(policy,'type'); policy.type = ''; end
      if isfield(self.policy,'p');
        self.np = length(unwrap(self.policy.p));
      end
      
      if exist('angi','var'); self.angi = angi; else self.angi = []; end
      if exist('poli','var'); self.poli = poli; else self.poli = 1:self.D; end
      if exist('actuate','var') && ~isempty(actuate); self.actuate=actuate; end
    end
    
    function [uM,uS,uC,s] = fcn(self, s)
      % CTRL.FCN is the main function to output control. Sub-classes will
      % override this function.
      %
      % [uM, uS, uC, s, duMds, duSds, duCds, dsds, duMdp, duSdp, duCdp, ...
      %   dsdp] = CTRL.FCN(s,propdyn)
      %
      % self        .          controller structure
      %   actuate   @          function to actuate calculated action
      %   angi                 indices of angular variabels
      %   dyn       @          controller's dynamics model (for CtrlBF only)
      %   E                    number of predictive state variables
      %   on        D x D      observation noise
      %   onp       D x D      observation noise (only last E vars non-zero)
      %   poli                 indices of policy input
      %   policy    .          policy structure
      %     fcn     @          policy function
      %   U                    number of control outputs
      % s           .          state structure
      %   m         F x 1      state mean
      %   s         F x F      state variance
      %   v        Dz x Dz     filter variance
      % M      (U+Dz) x 1      control signal mean vector
      % S      (U+Dz) x (U+Dz) control signal variance matrix
      % C           F x (U+Dz) input-output covariance matrix
      % dMds   (U+Dz) x S      derivatives of outputs wrt input state struct
      % dSds (U+Dz)^2 x S
      % dCds F*(U+Dz) x S
      % dsds        S x S      ouput state derivative wrt input state
      % dMdp   (U+Dz) x P      P is the total number of parameters is the policy
      % dSdp (U+Dz)^2 x P
      % dCdp F*(U+Dz) x P
      % dsdp        S x P      ouput state derivative wrt policy parameters
      %
      % See also CTRLBF.FCN, CTRLNF.FCN.
      if strcmp(self.policy.type, 'random')
        uM = self.random_action();
        uS = zeros(self.U); uC = zeros(self.D,self.U);
      end
    end
    
    % Filter Clearer
    function s = clear_filter(self, s)
      % Clears all filter variables (if they exist).
      % s = CTRL.CLEAR_FILTER(s)
      %   s:  state struct
      i = 1:self.D; % only physical variables, no filter variable indices
      s.m = s.m(i);
      if isfield(s,'s'); s.s = s.s(i,i); end
      if isfield(s,'v'); s = rmfield(s,'v'); end
    end
    
    function u = random_action(self)
      % u = CTRL.RANDOM_ACTION(), outputs a random action
      %   u:   U x 1 control action
      u = self.policy.maxU.*(2*rand(1,self.U)-1);
    end
    
    function s = reset_filter(~, s)
      % s = CTRL.RESET_FILTER(s), does nothing unless overwritten
      %   s:  state struct
      % See also CTRLBF.RESET_FILTER
    end
    
    function set_dynmodel(self, dyn)
      % CTRL.SET_DYNMODEL(dyn), for updating controller's dynmodel
      %   dyn: dynamics model object
      % Note: dyn property only ever used by CTRLBF
      self.dyn = dyn;
    end
    
    function set_on(self, onE)
      % CTRL.SET_ON(onE), sets observation noise
      %   onE: E x E, observation noise (physical state-variables only)
      assert(numel(onE) == self.E^2);
      onD = blkdiag(zeros(self.U), onE);
      onD = repmat(onD,ceil(self.D/(self.U+self.E)));
      self.on = onD(end-self.D+1:end, end-self.D+1:end);
      p = self.D-self.E+1:self.D;
      self.onp = 1e-4*eye(self.D); self.onp(p,p) = onE;
    end
    
    function set_policy_p(self, p)
      % CTRL.SET_POLICY_P(p), updateds the policy parameters
      %    p:   policy parameter struct
      self.policy.p = p;
      self.np = length(unwrap(p));
    end
    
    function set_policy_opt(self, opt)
      % CTRL.SET_POLICY_OPT(p), updateds the policy optimisation settings
      %    opt:   policy optimisation settings
      self.policy.opt = opt;
    end
    
  end
  
  methods (Access = private)
    function build_state_index(self)
      % CTRL.BUILD_STATE_INDEX(), computes internal field 'is' - a state struct
      % whose members are a states members' indexes. Requires subclass to
      % implement function CTRL.RESET_FILTER
      s.m = nan(self.D,1);
      s = self.reset_filter(s); % generates possible filter variables
      s.s = nan(length(s.m)); % s.m is now length F 
      if isfield(s,'reset'); s = rmfield(s,'reset'); end                        % TODO: delete? Is 'reset' used still?
      self.ns = length(unwrap(s));
      self.is = rewrap(s,1:self.ns);
      self.F = length(s.m);
      self.Dz = self.F - self.D;
    end
  end
  
end
