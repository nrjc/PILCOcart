% Copyright (C) 2014, Carl Edward Rasmussen and Andrew McHutchon, 2014-11-27

function [M, S, nlp, nlpdp, Mdp, Sdp] = GPfstep(varargin)
  if nargout < 4
    [M, S, nlp] = GPfilt(varargin{:});
  else
    [M, S, nlp, nlpdp, Mdp, Sdp] = GPfiltd(varargin{:});
  end
end                                                                   % GPfstep

function [M, S, nlp] = GPfilt(p, dyn, m, s, y, u)
  % function to compute the filter distribution at the current time step
  % p(x_t | z_1:t) given a distribution over the previous state and a
  % measurement of the current state.

  E = length(m); Sp = exp(2*[p.pn]); So = diag(exp(2*[p.on]));

  if isempty(dyn)
    Mx = m; Sx = s;
  else
    Du = length(u);                                          % add the controls
    m = [m; u];
    s = [s zeros(E, Du); zeros(Du, E) 1e-9*eye(Du)];
    
    [Mx, Sx] = dyn.pred(m, s); % predictive distribution for the next time step

    Sx = Sx + diag(Sp);            % add process noise to latent state variance
  end

  % find the predicted distribution of observation at the next time step
  Mz = Mx; Sz = Sx + So; Cxz = Sx;               % same mean, expanded variance

  % calculate the negative log probability of the observation
  ismy = Sz\(Mz-y);
  nlp = E/2*log(2*pi) + log(det(Sz))/2 + (Mz-y)'*ismy/2;

  % condition on the actual observation. This is equivilant to multiplying
  % the predicted distribution with the observation distribution: N(y_t,S_n)
  isym = Sz\(y-Mz);
  M = Mx + Cxz*isym;
  S = So - So*(Sz\So);    % a more stable version of S = Sx - Cxz*(Sz\Cxz');
end
  
function [M, S, nlp, nlpdp, Mdp, Sdp] = GPfiltd(p, dyn, m, s, y, u, mdp, sdp)
  % function to compute the filter distribution at the current time step
  % p(x_t | z_1:t) given a distribution over the previous state and a 
  % measurement of the current state.

  E = length(m); Sp = diag(exp(2*[p.pn])); So = diag(exp(2*[p.on]));
  I = diag(true(1,size(Sp,1)));
  Np = length(unwrap(p)); idx = rewrap(p,1:Np); 
  if nargin < 8; mdp = zeros(E,Np); sdp = zeros(E^2,Np); end

  if isempty(dyn)
    Mx = m; Sx = s; Mxdp = mdp; Sxdp = sdp;
  else
    Du = length(u); 
    m = [m; u];                                              % add the controls
    s = [s zeros(E, Du); zeros(Du, E) 1e-9*eye(Du)];
    
    % find the predictive distribution for the next time step
    [Mx, Sx, ~, Mxdm, Sxdm, ~, Mxds, Sxds, ~, Mxdp, Sxdp, ~] = dyn.preD(m, s);
    i = E-dyn.E+1:E; ii = sub2ind2(E+Du,i,i);

    Mxdp = Mxdm(:,i)*mdp + Mxds(:,ii)*sdp + Mxdp;
    Sxdp = Sxdm(:,i)*mdp + Sxds(:,ii)*sdp + Sxdp;

    Sx = Sx + Sp;                  % add process noise to latent state variance
    Sxdp(I(:),[idx.pn]) = Sxdp(I(:),[idx.pn]) + 2*Sp;
  end

  % find the predicted distribution of observation at the next time step
  Mz = Mx; Sz = Sx + So; Cxz = Sx; % same mean, expanded variance
  Szdp = Sxdp; Szdp(I,[idx.on]) = Szdp(I,[idx.on]) + 2*So;

  E = size(Sp,1);
  % calculate the negative log probability of the observation & derivatives
  ismy = Sz\(Mz-y);
  nlp = E/2*log(2*pi) + log(det(Sz))/2 + (Mz-y)'*ismy/2;
  nlpdM = ismy; nlpdSz = Sz\eye(E)/2 - ismy*ismy'/2;
  nlpdp = nlpdM'*Mxdp + nlpdSz(:)'*Szdp;

  % condition on the actual observation. This is equivilant to multiplying
  % the predicted distribution with the observation distribution: N(y_t,S_n)
  isym = Sz\(y-Mz);
  M = Mx + Cxz*isym;
  S = So - So*(Sz\So);
  CiS = Cxz/Sz;
  sxdp1 = reshape(Sxdp,E,[]); sxdp2 = reshape(Sxdp',[],E);
  szdp2 = reshape(Szdp',[],E);
  Mdp = Mxdp + reshape(sxdp2*isym,[],E)' - CiS*reshape(szdp2*isym,[],E)' ...
                                                                    - CiS*Mxdp;
  Sdp = Sxdp - reshape(sxdp2*CiS',[],E^2)' + ...
               reshape(CiS*reshape(reshape(szdp2*CiS',[],E^2)',E,[]),E^2,[])...
                                                   - reshape(CiS*sxdp1,E^2,[]);
end
