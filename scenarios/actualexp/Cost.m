classdef Cost < CostSuper
  % CartDoublePole immediate cost function. The cost is 1 - exp(-0.5*a*d^2),
  % where "a" is a (positive) constant and "d^2" is the squared (Euclidean)
  % distance between the tip of the pendulum and the upright position.
  %
  % Cost Properties:
  %   ell     - 2x1 lengths of the two part of the pendulum
  %   gamma   - discounting parameter
  %   width   - array of widths of the cost (summed together)
  %
  % Cost Methods:
  %   Cost   - constructor
  %   pre    - precomputations that only need to happen once
  %
  % For documentation, see also <a href="task.pdf">task.pdf</a>
  % Copyright (C) 2008-2015 by Carl Edward Rasmussen and Rowan McAllister
  % 2015-07-17
  
  properties (SetAccess = private)
    % default values:
    ell = [60 142]/136.6
    %ell = [60 142]/60
    width = 0.8
    gamma = 1.0    % no discounting
  end
  
  methods
    
    % Constructor
    function self = Cost(D, ell, width, gamma)
      % COST constructor
      %
      %   self = Cost(D, ell, width, gamma)
      %
      self@CostSuper(D);
      if exist('ell','var'); self.ell = ell; end
      if exist('width','var'); self.width = width; end
      if exist('gamma','var'); self.gamma = gamma; end
      
      self.pre();
    end
    
    function pre(self)
      % PRE, precomputations that only need to happen once
      
      D = self.D; %#ok<*PROP>
      self.cangi = [3 4 7 8 11 12];
      self.Da = D;
      self.Dt = D + 2*length(self.cangi);
      
      z = zeros(2*length(self.cangi),1); z([22 24]) = 1;
      C = zeros(2,2*length(self.cangi)); C(1,10) = 1; C(1,[21 23]) = -self.ell;
      C(2,[22 24]) = self.ell; Q = C'*C;
      for i = 1:length(self.width)           % scale mixture of immediate costs
        W = Q/self.width(i)^2;
        self.zW{i} = struct('z',z,'W',W);
      end
      
    end
    
  end
  
end
