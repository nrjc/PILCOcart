classdef CtrlBF < Ctrl & handle
  
  % Controller with Bayes Filter. The random state N(s.m,s.s) is composed of
  % two states. The real system state is given by N(s.rm,s.rs), and the
  % internal filter state N(N(s.zm,s.zs),s.v). The main function first calls
  % the policy, then augment with trignometric functions, and finally
  % (optionally) call the actuate function.
  % See ctrlBF.pdf for details.
  %
  % CtrlBF Properties:
  %   approxZC      -  (bool) use approximate world-filter covariance?
  %   broad_prior   -  width of prior filter
  %
  % Ctrl Methods:
  %   fcn           -  main function, computes control signal
  %   fillIn        -  (static) internal chain rule computations
  %   reset_filter  -  resets state fields {zs,zc,v} to a broad prior
  %   set_approxZC  -  sets the approx world-filter covariance flag
  %   symmetrise    -  (static) symmetrises variance matrices
  %
  % See also <a href="ctrlBF.pdf">ctrlBF.pdf</a>,
  % <a href="ctrlBF9.pdf">ctrlBF9.pdf</a> CTRL.M, CTRLBFT.M.
  % Copyright (C) 2014-2016 by 
  % Carl Edward Rasmussen and Rowan McAllister 2016-01-15
  
  properties (SetAccess = private)
    approxZC = false
    approxSV = false
    broad_prior = 10^3 % ideally >> 10, but causes inaccurate gradients
  end
  
  methods
    
    % Constructor
    function self = CtrlBF(varargin)
      % CTRLBF.CTRLBF is the sub-class constructor of CTRL.CTRL
      % For help, see also CTRL.CTRL
      self@Ctrl(varargin{:});                          % Super Ctrl constructor
    end
    
    % Filter Resetter (override Ctrl method)  
    function s = reset_filter(self, s)
      % s = CTRLBF.RESET_FILTER(s)
      %   s: state struct
      i = 1:self.D;
      s.m = [s.m(i) ; self.mu0]; % set filter prior mean to system initial mean
      if isfield(s,'s') % i.e. propagate_mode, not rollout mode.
        sx = s.s(i,i);
        s.s = zeros(self.F); s.s(i,i) = sx;
        assert(all(self.S0(:) == sx(:)));
      end
      s.v = self.S0; % set filter prior variance to system initial variance
    end

    % Filter Resetter (override Ctrl method)
    function s = reset_filter_broad(self, s)
      % s = CTRLBF.RESET_FILTER(s)
      %   s:  state struct
      i = 1:D;
      s.m = [s.m(i) ; zeros(self.D,1)];
      if isfield(s,'s') % i.e. propagate_mode, not rollout mode.
        sx = s.s(i,i);
        s.s = zeros(self.F); s.s(i,i) = sx;
      end
      s.v = self.broad_prior*eye(self.D);
    end
    
    function set_approxZC(self,new_approxZC)
      % CTRLBF.SET_APPROXZC(new_approxZC)
      self.approxZC = new_approxZC;
    end
    
    function set_approxSV(self,new_approxSV)
      % CTRLBF.SET_APPROXSV(new_approxSV)
      self.approxSV = new_approxSV;
    end
    
    % Main Function
    function [M, S, C, s, dMds, dSds, dCds, dsds, ...
        dMdp, dSdp, dCdp, dsdp] = fcn(self,s)
      % For help, see also CTRL.FCN
      
      D = self.D; Dz = self.Dz; F = self.F;
      if length(s.m) < F; s = self.reset_filter(s); end   % init filter if none
      assert(length(s.m) == self.F);
      rollout_mode = ~isfield(s,'s'); % else propagate mode
      if rollout_mode                                                           % TODO fix, s.s assumed to always exist
        sy = zeros(D);
        ss = zeros(F);
      else
        sy = s.s(1:D,1:D) + self.onp;            % sy is noisy version of state % TODO handle different types of noise models, i.e. if ctrl and propdyn differ
        ss = s.s;
      end
      % test variables:
      DEBUG_EIG=false; DEBUG_ROLLOUT=false; DEBUG_MODE=false;
      
      % 0) Initialisations ----------------------------------------------------
      angi = self.angi; poli = self.poli;
      U = self.U; A = 2*length(angi); E = self.E;
      I = eye(D); fillIn = @CtrlBF.fillIn;
      derivativesRequested = nargout > 4;
      
      % Indices of {M,S,V}
      ix = 1:D;                                       % indices of latent state
      iz = max(ix)      + (1:Dz);                % indices of input filter-mean
      iu = max(iz)      + (1:Dz);              % indices of updated filter-mean
      au = max(iu)      + (1:A); % ind. updated filter angles mapped to sin/cos
      j  = max([iu,au]) + (1:U);                    % indices of control signal
      k  = max(j)       + (1:E);             % indices of predicted filter-mean
      K  = max(k);
      
      % Input indices
      pi = [iu, au]; pi = pi(poli);                   % input indices of policy
      di = [iu, j];                                 % input indices of dynmodel
      oz = [iu, j, k]; oz = oz(end-Dz+1:end); % ind select next predicted state
      o  = [j, oz];            % controller outputs control and predicted state
      
      i = [ix,iz];
      M = nan(K,1); M(i) = s.m;
      S = nan(K); S(i,i) = ss;
      if DEBUG_MODE; disp(S(i,i)); end
      if DEBUG_EIG && any(eig(S(i,i))<-1e-8); keyboard; end
      Syz = ss; Syz(ix,ix) = sy;
      if DEBUG_EIG && any(eig(S(i,i))<-1e-8); keyboard; end
      V = zeros(K); V(iz,iz) = s.v;
      if derivativesRequested
        symmetrise = @CtrlBF.symmetrise;
        ns = self.ns; np = self.np; is = self.is;
        idx3 = @(D,i,j,k) (bsxfun(@plus, i(:), D*(j(:)'-1)) + D*D*(k-1));
        Mds = zeros(K,ns); Mds(i,is.m) = eye(F);
        Sds=zeros(K*K,ns); Sds(idx3(K,i,i,is.s)) = 1;
        Vds=zeros(K*K,ns); Vds(idx3(K,iz,iz,is.v)) = 1;
        dsds = zeros(ns); dsdp = zeros(ns,np);
        XS = unwrap({is.s,is.v}); XST = unwrap({is.s',is.v'}); DD = D*D;
      end
      
      % 1) Bayes-filter update step -------------------------------------------
      n = self.onp;
      v = s.v;
      wy = v/(v+n); wz = n/(v+n); w = [wy wz]; % Bayes update 'weights'
      
      M(iu) = w*M(i);
      Syzw = Syz*w'; S(iu,iu) = w*Syzw;
      S(iu,i) = w*S(i,i); S(i,iu) = S(iu,i)';
      V(iu,iu) = n/(v+n)*v; % only update the uncertain varibles
      if derivativesRequested
        iuiu = sub2ind2(K,iu,iu);
        iui  = sub2ind2(K,iu,i);
        Mds(iu,is.m) = w;
        Mds(iu,is.v) = kron((M(ix)-M(iz))'/(v+n),wz);
        Sds(iuiu,is.s) = prodd(w,'eye',w');
        Sds(iuiu,is.v) = 2*kron((Syzw(ix,:)-Syzw(iz,:))'/(v+n),wz);
        Sds(iui,is.s) = 2*prodd(w,'eye'); % 2 allocates both off-diags
        Sds(iui,is.v) = 2*kron((S(ix,i)-S(iz,i))'/(v+n),wz);
        Vds(iuiu,is.v) = kron(wz,wz);
        [Mds, Sds, Vds] = symmetrise(XS, XST, Mds, Sds, Vds);
      end
      if DEBUG_EIG && any(eig(V(iu,iu))<-1e-8); keyboard; end
      if DEBUG_EIG && any(eig(S([ix,iu],[ix,iu]))<-1e-8); keyboard; end
      
      % 2) Augment updated-filter ---------------------------------------------
      if ~derivativesRequested
        [M(au), S(au,au), Cg, V(au,au)] = gTrigh(M(iu),S(iu,iu),V(iu,iu),angi);
        [S,V] = fillIn(iu,iu,au,K,S,Cg,V);
      else
        [M(au), S(au,au), Cg, V(au,au), ...
          mdm, sdm, cgdm, vdm, mds, sds, cgds, vds, mdv, sdv, cgdv, vdv] = ...
          gTrigh(M(iu), S(iu,iu), V(iu,iu), angi);
        [S,V,Mds,Sds,Vds] = fillIn(iu,iu,au,K,S,Cg,V, ...
          mdm,sdm,vdm,cgdm,mds,sds,vds,cgds,mdv,sdv,vdv,cgdv, ...
          Mds,Sds,Vds,[],[],[],[]);
      end
      if DEBUG_EIG && any(eig(S([iu,au],[iu,au]))<-1e-8); keyboard; end
      
      % 3) Compute distribution of the control signal -------------------------
      if ~derivativesRequested
        [M(j), S(j,j), Cp] = self.policy.fcn(self.policy, M(pi), S(pi,pi));
        if rollout_mode; S(j,j) = 0; end                                        % TODO: is this correct? policy.fcn produces noise!
        if isfield(self, 'actuate'), self.actuate(M(j)); end     % actuate ctrl
        S = fillIn(pi,iu,j,K,S,Cp,[]); % no V fillin since C applies to only S here
        g = [I,Cg]; g = g(:,poli); wgp = w'*g*Cp;
        S(i,j) = S(i,i)*wgp; S(j,i) = S(i,j)';                                  % TODO fix, is this still needed?
      else
        pipi = sub2ind2(K,pi,pi);
        [M(j), S(j,j), Cp, mdm, sdm, cpdm, mds, sds, cpds, Mdp, Sdp, cpdp] = ...
          self.policy.fcn(self.policy, M(pi), S(pi,pi)); Vdp = 0*Sdp;
        if isfield(self, 'actuate'), self.actuate(M(j)); end     % actuate ctrl
        [S,~,Mds,Sds,~,~,Mdp,Sdp,Vdp] = fillIn(pi,iu,j,K,S,Cp,[], ...
          mdm,sdm,[],cpdm,mds,sds,[],cpds,[],[],[],[], ...
          Mds,Sds,Vds,Mdp,Sdp,Vdp,cpdp);
        g = [I,Cg]; g = g(:,poli); wgp = w'*g*Cp;
        dwgpdp = prodd(w'*g,cpdp);
        % most duCds chain terms:
        dgds = [zeros(DD,ns);cgdm*Mds(iu,:)+cgds*Sds(iuiu,:)+cgdv*Vds(iuiu,:)];
        dgds = dgds(sub2ind2(D,1:D,poli),:);
        dCpds = cpdm*Mds(pi,:)+cpds*Sds(pipi,:);
        dwgpds = prodd(w',dgds,Cp) + prodd(w'*g,dCpds);
        % extra duCds chain terms:
        dwtdv = kron(wz,[I;-I]/(v+n));
        dwtdv = symmetrised(dwtdv,2);
        dwgpds(:,is.v) = dwgpds(:,is.v) + prodd([],dwtdv,g*Cp);
        % iz-j covariance (for >1 Markov representations)
        S(i,j) = S(i,i)*wgp; S(j,i) = S(i,j)';
        ii = sub2ind2(K,i,i); ij = sub2ind2(K,i,j); ji = sub2ind2(K,j,i);
        Sds(ij,:) = prodd(S(i,i),dwgpds) + prodd([],Sds(ii,:),wgp);
        Sdp(ij,:) = prodd(S(i,i),dwgpdp);
        Sds(ji,:) = transposed(Sds(ij,:),numel(i));
        Sdp(ji,:) = transposed(Sdp(ij,:),numel(i));
        % TODO: the V derrivatives?
      end
      if DEBUG_EIG && any(eig(S([ix,iu,j],[ix,iu,j]))<-1e-8); keyboard; end
      
      % 4) Bayes-filter predict step ------------------------------------------
      if ~derivativesRequested
        [M(k),S(k,k),Cd,V(k,k)] = ...
          self.dyn.predh(M(di),S(di,di),V(di,di),false,self.approxSV);
        if rollout_mode; V(k,k)=V(k,k)+S(k,k); S(k,k)=0; end                    % TODO: correct?
        [S,V] = fillIn(di,j,k,K,S,Cd,V);
        % cross cov between z_{t+1} and {z_{t},x_{t}}, for >1 Markov state-rep:
        egp = [eye(D),g*Cp];
        egpd = egp*Cd;
        wegpd = w'*egpd;
        S(i,k) = S(i,i)*wegpd; S(k,i) = S(i,k)';
        % cross cov between z_{t+1} and {z_{tt}}, for >1 Markov state-rep:
        S(iu,k) = S(iu,iu)*egpd; S(k,iu) = S(iu,k)';
      else
        [M(k),S(k,k),Cd,V(k,k),mdm,sdm,cddm,vdm,mds,sds,cdds,vds,mdv,sdv,cddv, ...
          vdv] = self.dyn.predh(M(di),S(di,di),V(di,di),false,self.approxSV);
        [S,V,Mds,Sds,Vds,~,Mdp,Sdp,Vdp,cddp] = fillIn(di,j,k,K,S,Cd,V, ...
          mdm,sdm,vdm,cddm,mds,sds,vds,cdds,mdv,sdv,vdv,cddv, ...
          Mds,Sds,Vds,Mdp,Sdp,Vdp,[]);
        % cross cov between z_{t+1} and {z_{t},x_{t}}, for >1 Markov state-rep:
        egp = [eye(D),g*Cp];
        egpd = egp*Cd;
        wegpd = w'*egpd;
        S(i,k) = S(i,i)*wegpd; S(k,i) = S(i,k)';
        S(iu,k) = S(iu,iu)*egpd; S(k,iu) = S(iu,k)';
        % most chain terms:
        ik = sub2ind2(K,i,k); ki = sub2ind2(K,k,i);
        iuk = sub2ind2(K,iu,k); kiu = sub2ind2(K,k,iu);
        didi = sub2ind2(K,di,di);
        degpds = [zeros(DD,ns); prodd([],dgds,Cp) + prodd(g,dCpds)];
        degpdp = [zeros(DD,np); prodd(g,cpdp)];
        dCdds = cddm*Mds(di,:)+cdds*Sds(didi,:)+cddv*Vds(didi,:);
        degpdds = prodd([],degpds,Cd) + prodd(egp,dCdds);
        degpddp = prodd([],degpdp,Cd) + prodd(egp,cddp);
        dwegpdds = prodd(w',degpdds);
        dwegpddp = prodd(w',degpddp);
        % extra chain term:
        dwegpdds(:,is.v) = dwegpdds(:,is.v) + prodd([],dwtdv,egp*Cd);
        Sds(ik,:) = prodd([],Sds(ii,:),wegpd) + prodd(S(i,i),dwegpdds);
        Sdp(ik,:) = prodd(S(i,i),dwegpddp);
        Sds(ki,:) = transposed(Sds(ik,:),numel(i));
        Sdp(ki,:) = transposed(Sdp(ik,:),numel(i));
        % cross cov between z_{t+1} and {z_{tt}}, for >1 Markov state-rep:
        Sds(iuk,:) = prodd([],Sds(iuiu,:),egpd) + prodd(S(iu,iu),degpdds);
        Sdp(iuk,:) = prodd(S(iu,iu),degpddp);
        Sds(kiu,:) = transposed(Sds(iuk,:),numel(iu));
        Sdp(kiu,:) = transposed(Sdp(iuk,:),numel(iu));
      end
      if DEBUG_EIG && any(eig(S([j,k],[j,k]))<-1e-8); keyboard; end
      
      % 5) Select distribution of predicted-filter ----------------------------
      M = M(o);
      S = S(o,o);
      Cu = wgp;                               % cov[input, control-output] term
      Cz = [w', Cu, wegpd]; Cz = Cz(:,end-Dz+1:end); % cov[input,filter-output]
      C = [Cu, Cz];
      s.v = V(oz,oz);
      
      if derivativesRequested
        oo = sub2ind2(K,o,o);
        ozoz = sub2ind2(K,oz,oz);
        
        dCuds = dwgpds;
        dCudp = dwgpdp;
        dwtds = zeros(numel(w),ns); dwtds(:,is.v) = dwtdv;
        dwtdp = zeros(numel(w),np);
        dCzds = [dwtds; dCuds; dwegpdds]; dCzds = dCzds(end-(F*Dz)+1:end,:);
        dCzdp = [dwtdp; dCudp; dwegpddp]; dCzdp = dCzdp(end-(F*Dz)+1:end,:);
        dCds = [dCuds; dCzds];
        dCdp = [dCudp; dCzdp];
        
        [Mds,Sds,Vds,Sdp,Vdp,dCds]=symmetrise(XS,XST,Mds,Sds,Vds,Sdp,Vdp,dCds);
        dMds = Mds(o,:); dMdp = Mdp(o,:);
        dSds = Sds(oo,:); dSdp = Sdp(oo,:);
        
        dsds(is.m,is.m) = Mds(i,is.m);
        dsds(is.s,is.s) = Sds(sub2ind2(K,i,i),is.s);
        dsds(is.v ,:) = Vds(ozoz,:); dsdp(is.v ,:) = Vdp(ozoz,:);
      end
      
      if DEBUG_EIG && any(eig(S)<-1e-8); keyboard; end
      
      if DEBUG_ROLLOUT && rollout_mode % everything should be a point mass:
        assert(~any(S(:))); assert(~isfield(s,'s'));
      end
      
    end
    
  end
  
  methods (Static)
    
    % Apply chain rule and fill out cross covariance terms
    function [S, V, Mds, Sds, Vds, Cds, Mdp, Sdp, Vdp, Cdp] = ...
        fillIn(i,j,k,K,S,C,V, ...
        mdm,sdm,vdm,cdm,mds,sds,vds,cds,mdv,sdv,vdv,cdv, ...
        Mds,Sds,Vds,Mdp,Sdp,Vdp,Cdp)
      
      q = S(j,i)*C; S(j,k) = q; S(k,j) = q';                     % off-diagonal
      if ~isempty(V); q = V(j,i)*C; V(j,k) = q; V(k,j) = q'; end % off-diagonal
      if nargout <= 2, return, end;
      
      if isempty(k), return; end
      if isempty(vdm), vdm=0*sdm; end; if isempty(vds), vds=0*sds; end
      if isempty(mdv), mdv=0*mds; end; if isempty(sdv), sdv=0*sds; end
      if isempty(vdv), vdv=0*sds; end; if isempty(cdv), cdv=0*cds; end
      
      I = eye(length(k));
      ii = sub2ind2(K,i,i); kk = sub2ind2(K,k,k);          % vectorised indices
      ji = sub2ind2(K,j,i); jk = sub2ind2(K,j,k);
      kj = kron(k,ones(1,length(j))) + kron(ones(1,length(k)),(j-1)*K);
      
      Mds(k,:)  = mdm*Mds(i,:) + mds*Sds(ii,:) + mdv*Vds(ii,:);     % chainrule
      Sds(kk,:) = sdm*Mds(i,:) + sds*Sds(ii,:) + sdv*Vds(ii,:);
      Vds(kk,:) = vdm*Mds(i,:) + vds*Sds(ii,:) + vdv*Vds(ii,:);
      Cds       = cdm*Mds(i,:) + cds*Sds(ii,:) + cdv*Vds(ii,:);
      if isempty(Cdp) && nargout > 6
        Mdp(k,:)  = mdm*Mdp(i,:) + mds*Sdp(ii,:) + mdv*Vdp(ii,:);
        Sdp(kk,:) = sdm*Mdp(i,:) + sds*Sdp(ii,:) + sdv*Vdp(ii,:);
        Vdp(kk,:) = vdm*Mdp(i,:) + vds*Sdp(ii,:) + vdv*Vdp(ii,:);
        Cdp       = cdm*Mdp(i,:) + cds*Sdp(ii,:) + cdv*Vdp(ii,:);
      elseif nargout > 6
        mdp = zeros(K,size(Mdp,2)); sdp = zeros(K*K,size(Mdp,2)); vdp = sdp;
        mdp(k,:)  = Mdp; Mdp = mdp;
        sdp(kk,:) = Sdp; Sdp = sdp;
        vdp(kk,:) = Vdp; Vdp = vdp;
      end
      
      SS = kron(I,S(j,i)); CC = kron(C',eye(length(j)));
      Sds(jk,:) = SS*Cds + CC*Sds(ji,:); Sds(kj,:) = Sds(jk,:);
      if ~isempty(V);
        VV=kron(I,V(j,i));
        Vds(jk,:) = VV*Cds + CC*Vds(ji,:); Vds(kj,:) = Vds(jk,:);
      end
      if nargout > 6;
        Sdp(jk,:) = SS*Cdp + CC*Sdp(ji,:); Sdp(kj,:) = Sdp(jk,:);
        if ~isempty(V); Vdp(jk,:)=VV*Cdp+CC*Vdp(ji,:); Vdp(kj,:)=Vdp(jk,:); end
      end
    end
    
    % Symmetrise the numerator (row) and denomiator (column) cross covariances
    function [Mds, Sds, Vds, Sdp, Vdp, duCds] = ...
        symmetrise(XS, XST, Mds, Sds, Vds, Sdp, Vdp, duCds)
      Mds(:,XS) = (Mds(:,XS) + Mds(:,XST))/2;
      Sds = symmetrised(Sds,1); Sds(:,XS) = (Sds(:,XS) + Sds(:,XST))/2;
      Vds = symmetrised(Vds,1); Vds(:,XS) = (Vds(:,XS) + Vds(:,XST))/2;
      if nargout >= 4, Sdp = symmetrised(Sdp,1); end
      if nargout >= 5, Vdp = symmetrised(Vdp,1); end
      if nargout >= 6, duCds(:,XS) = (duCds(:,XS) + duCds(:,XST))/2; end
    end
    
  end
  
end
