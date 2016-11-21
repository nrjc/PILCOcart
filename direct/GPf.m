function [nlp, nlpdp] = GPf(p, dyn, data, dyni, dyno)
% Implement forward filtering using a GP transition model and
% an additive Gaussian noise observation model.
% Copyright (C) 2014, Carl Edward Rasmussen and Andrew McHutchon, 2015-02-12

useprior = 0;
if length(data) > 1
  error(['data is a struct array - call multiTrial or use indexing\n']);
end
y = data.state; u = data.action; U = size(u,2);

% Initialisations -------------------------------------------------------------
T = size(y,1); D = dyn.D-U; E = dyn.E; Np = length(unwrap(p)); 
if exist('dyni') && exist('dyno') 
    e1 = setdiff(dyni,dyno); e2 = dyno;
else
    e1 = 1:D-E; e2 = D-E+1:D;
end
Mf = zeros(E,T); Sf = zeros(E,E,T); nlp = zeros(1,T); nlpdp = zeros(Np,T);

if isfield(dyn,'priorm') % for t = 1 ------------------------------------------
  m = dyn.priorm; S = dyn.priorS;                        % Prior on first state
else
  m = zeros(E,1); S = eye(E);                 
end
  
if nargout < 2;
  [Mf(:,1), Sf(:,:,1), nlp(1)] = GPfstep(p, [], m, S, y(1,e2)');
else
  [Mf(:,1), Sf(:,:,1), nlp(1), nlpdp(:,1), Mdp, Sdp] = ...
                                                GPfstep(p, [], m, S, y(1,e2)');
end % -------------------------------------------------------------------------


for i=2:T % The forward filtering sweep for t = 2:T ---------------------------
  if nargout < 2
    [Mf(:,i), Sf(:,:,i), nlp(i)] = GPfstep(p, dyn, [y(i-1,e1) Mf(:,i-1)']', ...
                                 [1e-9*eye(length(e1)) zeros(length(e1),E); ...
                        zeros(E,length(e1)) Sf(:,:,i-1)], y(i,e2)', u(i-1,:)');
  else
    [Mf(:,i), Sf(:,:,i), nlp(i), nlpdp(:,i), Mdp, Sdp] = GPfstep(p, dyn, ...
        [y(i-1,e1) Mf(:,i-1)']', [1e-9*eye(length(e1)) zeros(length(e1),E); ...
              zeros(E,length(e1)) Sf(:,:,i-1)], y(i,e2)', u(i-1,:)', Mdp, Sdp);
  end
  if any(diag(Sf(:,:,i)) < -1e-9); keyboard; end
end % -------------------------------------------------------------------------

% Sum up nlp and add together derivatives
nlp = sum(nlp); if nargout == 2; nlpdp = sum(nlpdp, 2); end

% Rewrap to struct
if nargout == 2; 
  nlpdp = rewrap(p,nlpdp); 
  if dyn.fixLin
    [nlpdp.m] = deal(zeros(size(nlpdp(1).m)));
    [nlpdp.b] = deal(zeros(size(nlpdp(1).b)));
  end
end
