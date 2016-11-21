classdef Cost < CostSuper
  % Unicycle immediate cost function. The cost is 1 - exp(-0.5*a*d^2), where
  % "a"is a (positive) constant and "d^2" is the squared difference between
  % the current z-position of the top of the unicycle and the upright position.
  %
  % Cost Properties:
  %   p      - parameters: [radius of wheel, length of rod]
  %   width  - array of widths of the cost (summed together)
  %   gamma  - discount factor
  %
  % Cost Methods:
  %   augment    - function to augment variables
  %   augmentcov - function to handle the covariances of augment variables
  %   Cost       - constructor
  %   pre        - precomputations that only need to happen once
  %
  % Copyright (C) 2009-2014 Carl Edward Rasmussen, Marc Deisenroth,
  % Philipp Hennig, Joe Hall, Rowan McAllister 2016-03-04
  
  properties (SetAccess = private)
    % default values:
    p = [0.22 0.81]
    width = 1.0
    gamma = 1.0    % no discounting
  end
  
  methods
    
    % Constructor
    function self = Cost(D, p, width, gamma)
      % COST constructor
      %
      %   self = Cost(D, p, width, gamma)
      %
      self@CostSuper(D);
      if exist('p','var'); self.p = p; end
      if exist('width','var'); self.width = width; end
      if exist('gamma','var'); self.gamma = gamma; end
      
      self.pre();
    end
    
    function pre(self)
      % PRE, precomputations that only need to happen once
      
      D = self.D; %#ok<*PROP>
      Ixc = 6; Iyc = 7;                              % coordinates of xc and yc
      Itheta = 8;  Ipsi = 10;                    % coordinates of theta and psi
      self.cangi = [Itheta, D+1, D+2, Ipsi];
      self.Da = D + 2; % state dim (augmented with Itheta-Ipsi and Itheta+Ipsi)
      self.Dt = self.Da + 2*length(self.cangi);      % state dim (with sin/cos)
      
      cw = self.width; rw = self.p(1); r = self.p(2);
      
      % 2. Define static penalty as distance from target setpoint
      Q = zeros(D+10);
      C1 = [rw r/2 r/2];
      Q([D+4 D+6 D+8],[D+4 D+6 D+8]) = 8*(C1'*C1);                         % dz
      C2 = [1 -r];
      Q([Ixc D+9],[Ixc D+9]) = 0.5*(C2'*C2);                               % dx
      C3 = [1 -(r+rw)];
      Q([Iyc D+3],[Iyc D+3]) = 0.5*(C3'*C3);                               % dy
      Q(9,9) = (1/(4*pi))^2;                                   % yaw angle loss
      
      z = zeros(self.Dt,1); z([D+4 D+6 D+8 D+10]) = 1;        % target setpoint
      
      for i = 1:length(cw)                   % scale mixture of immediate costs
        W = Q/cw(i)^2;
        self.zW{i} = struct('z',z,'W',W);
      end
      
    end
    
    function [M,S,C,Mdm,Mds,Sdm,Sds,Cdm,Cds] = augment(self,m,s)
      
      D = self.D; %#ok<*PROP>
      Da = self.Da;    % state dim (augmented with Itheta-Ipsi and Itheta+Ipsi)
      Dt = self.Dt;
      
      Itheta = 8;  Ipsi = 10;                    % coordinates of theta and psi
      
      P = eye(Da,D); P(D+1:end,Itheta) = [1;1]; P(D+1:end,Ipsi) = [-1;1];
      
      M = [P*m; nan(Dt-Da,1)];
      S = blkdiag(P*s*P',nan(Dt-Da));
      C = P';
      if nargout > 3
        Mdm = [P; zeros(Dt-Da,D)];
        Mds = zeros(Dt,D*D);
        Sdm = zeros(Dt*Dt,D);
        Sds = symmetrised(kron(Mdm,Mdm),1);
        Cdm = zeros(D*Da,D);
        Cds = zeros(D*Da,D*D);
      end
      
    end
    
    function [M,S,Mdm,Mds,Sdm,Sds] = augmentcov(self,m,s)
      % m:  2*D x 1
      % s:  2*D x 2*D
      % M: 2*Da x 1
      % S: 2*Da x 2*Da
      
      D = self.D;
      Da = self.Da;
      Dd = 2*D;
      Dad = 2*Da;
      Dtd = 2*self.Dt;
      
      Itheta = 8;  Ipsi = 10;                    % coordinates of theta and psi
      
      P = eye(Da,D); P(D+1:end,Itheta) = [1;1]; P(D+1:end,Ipsi) = [-1;1];
      P = blkdiag(P,P);
      
      M = [P*m; nan(Dtd-Dad,1)];
      S = blkdiag(P*s*P',nan(Dtd-Dad));
      if nargout > 2
        Mdm = [P; zeros(Dtd-Dad,Dd)];
        Mds = zeros(Dtd,Dd*Dd);
        Sdm = zeros(Dtd*Dtd,Dd);
        Sds = symmetrised(kron(Mdm,Mdm),1);
      end
      
    end
    
  end
  
end
