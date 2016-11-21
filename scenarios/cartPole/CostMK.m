classdef CostMK < CostSuper
  % Cart-pole (with Markov states) immediate cost function. The cost is
  % 1 - exp(-0.5*a*d^2), where "a" is a (positive) constant and "d^2" is the
  % squared (Euclidean) distance between the tip of the pendulum and the
  % upright position.
  %
  % CostMK Properties:
  %   ell     - 1x1 length of the pendulum
  %   gamma   - discounting parameter
  %   width   - array of widths of the cost (summed together)
  %
  % CostMK Methods:
  %   Cost   - constructor
  %   pre    - precomputations that only need to happen once
  %
  % Copyright (C) 2008-2015 by Carl Edward Rasmussen and Rowan McAllister
  % 2015-07-17
  
  properties (SetAccess = private)
    % default values:
    ell = 0.5
    width = 0.25
    gamma = 1.0    % no discounting
  end
  
  methods
    
    % Constructor
    function self = CostMK(D, ell, width, gamma)
      % COST constructor
      %
      %   self = CostMK(ell, width, gamma)
      %
      self@CostSuper(D);
      if exist('ell','var'); self.ell = ell; end
      if exist('width','var') ;self.width = width; end
      if exist('gamma','var'); self.gamma = gamma; end
      
      self.pre();
    end
    
    function pre(self)
      % PRE, precomputations that only need to happen once
      
      D = self.D; %#ok<*PROP>
      self.cangi = D;
      self.D0 = D;
      self.D1 = D + 2*length(self.cangi);
      
      Ix = D-1;
      Isin = D+1;
      Icos = D+2;
      
      z(Icos,1) = 1;
      Q(Icos,Icos) = self.ell^2;
      Q([Ix,Isin],[Ix,Isin]) = [1 -self.ell]'*[1 -self.ell];
      for i = 1:length(self.width)                 % scale mixture of immediate costs
        W = Q/self.width(i)^2;
        self.zW{i} = struct('z',z,'W',W);
      end
      
    end
    
  end
  
end