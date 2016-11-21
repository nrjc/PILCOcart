classdef CtrlNF < Ctrl & handle
  
  % Controller with No Filter. The state is given either as a point s.m or as a
  % distribution N(s.m,s.s). First augment with trignometric functions if
  % necessary, then call the policy, and finally (optionally) call the actuate
  % function. There is no filter, so no updates to any state structure filter
  % fields are required.
  %
  % See also CTRL.M, CTRLNFT.M.
  % Copyright (C) 2015 by Carl Edward Rasmussen and Rowan McAllister 2015-06-02
  
  methods
    
    % Constructor
    function self = CtrlNF(varargin)
      % CTRLNF.CTRLNF is the sub-class constructor of CTRL.CTRL
      % For help, see also Ctrl.Ctrl
      self@Ctrl(varargin{:});                          % Super Ctrl constructor
    end
    
    % Filter Resetter (override Ctrl method)
    function s = reset_filter(self, s)
      % s = CTRLBF.RESET_FILTER(s)
      %   s:  state struct
      s = self.clear_filter(s);
    end
    
    % Main function
    function [uM, uS, uC, s, dMds, dSds, dCds, dsds, ...
        dMdp, dSdp, dCdp, dsdp] = fcn(self, s)
      % For help, see also CTRL.FCN
      
      % CtrlNF only operates on a noisy version of the state:
      D = self.D; DD = D*D;
      s = self.clear_filter(s);           % clear any filter variables in state
      assert(length(s.m) == D);
      if isfield(s,'s')
        sy = s.s + self.on;           % propagate mode (distribution of states)
      else
        sy = zeros(D);                 % rollout mode (point-mass state sample)
      end

      angi = self.angi; poli = self.poli; A = length(angi);
      derivativesRequested = nargout > 4;
      ns = self.ns; is = self.is;
      D1 = D + 2*A;
      i=1:D;
      M = zeros(D1,1); M(i) = s.m; S = zeros(D1); S(i,i) = sy;
      if derivativesRequested
        idx = @(i,j,I) bsxfun(@plus, I*(i'-1), j);
        Mds = zeros(D1,ns); Mds(i,is.m) = eye(D);
        Sds = zeros(D1*D1,ns); Sds(:,is.s) = kron(Mds(:,is.m),Mds(:,is.m));
        dsds = zeros(self.ns); dsdp = zeros(ns,self.np);
        dsds(is.m,is.m) = eye(D); dsds(is.s(:),is.s(:)) = eye(D*D);
        dsds(is.s,is.s) = symmetrised(dsds(is.s,is.s),[1,2]);
      end

      % augment state with trig functions
      i = 1:D; k = D+1:D1;
      if ~derivativesRequested
        [M(k), S(k,k), cg] = gTrig(M(i), S(i,i), angi);
      else
        kk = idx(k,k,D1); ik = idx(i,k,D1); ki = idx(k,i,D1);
        [M(k), S(k,k), cg, Mds(k,is.m), Sds(kk,is.m), cgdm, ...
          Mds(k,is.s), Sds(kk,is.s), cgds] = gTrig(M(i), S(i,i), angi);
        qdm = prodd(S(i,i),cgdm);
        Sds(ik,is.m) = qdm; Sds(ki',is.m) = qdm;
        qds = prodd(S(i,i),cgds) + prodd([],'eye',cg);
        Sds(ik,is.s) = qds; Sds(ki',is.s) = qds;
      end
      q = S(i,i)*cg; S(i,k) = q; S(k,i) = q';

      % compute control signal
      if ~derivativesRequested
        [uM, uS, uC] = self.policy.fcn(self.policy, M(poli), S(poli,poli));
      else
        [uM, uS, uC, mdm, sdm, cdm, mds, sds, cds, dMdp, dSdp, dCdp] = ...
          self.policy.fcn(self.policy, M(poli), S(poli,poli));
      end
      if isfield(self, 'actuate'), self.actuate(uM); end   % actuate controller

      ec = [eye(D) cg]; ecp = ec(:,poli);
      if derivativesRequested
        poli2 = idx(poli,poli,D1); ii = sub2ind2(D1,i,i);
        dMds = mdm*Mds(poli,:) + mds*Sds(poli2,:);
        dSds = sdm*Mds(poli,:) + sds*Sds(poli2,:);
        
        duC = cdm*Mds(poli,:) + cds*Sds(poli2,:);
        decp = [zeros(DD,ns); cgdm*Mds(i,:) + cgds*Sds(ii,:)];
        decp = decp(sub2ind2(D,1:D,poli),:);
        dCds = prodd(ecp,duC) + prodd([],decp,uC);
        
        dCdp = prodd(ecp,dCdp);
        
        dCds(:,is.s) = symmetrised(dCds(:,is.s),2);
        dMds(:,is.s) = symmetrised(dMds(:,is.s),2);
        dSds(:,is.s) = symmetrised(dSds(:,is.s),2);
      end
      uC = ecp*uC;
    end
    
  end
  
end
