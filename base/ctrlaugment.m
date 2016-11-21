function [M, S, dynmodel, plant] = ctrlaugment(M, S, dynmodel, plant, nU)
% Augment dynmodel with sub-dynamics to keep a track of input-states. Used
% for first-order-hold and lagged control strategies. Note that dynamics
% affected by control scheme should be in dynmodel.sub{1}. Further note
% that it is a faff to sort out indices for dyni and poli because they may
% depend on angles...which get stuck on to the end of the state.
%
% Joe Hall 2012-07-04

Nf = length(dynmodel.sub); dt = plant.dt;
dyno=plant.dyno; dyni=plant.dyni; poli=plant.poli; odei=plant.odei;
if isfield(plant,'augi'), augi = plant.augi; else augi = []; end
if isfield(plant,'subi'), subi = plant.subi; else subi = []; end
D = max([odei subi augi]); Do = length(dyno); Di = length(dyni);

if isfield(plant,'delay'), delay = plant.delay; else delay = 0; end
if isfield(plant,'tau'), tau = plant.tau; else tau = dt; end

% Control Sub-Dynamics ---------------------------------------------------
con = functions(plant.ctrl); con = con.function; Nc = Nf+1;
if (strcmp(con,'zoh') && delay==0)                                   % ZOH
  Nc = Nf;
elseif strcmp(con,'zoh') || ...                              % delayed ZOH 
      (strcmp(con,'foh') && tau+delay<=dt) % ..or FOH with quick rise-time
  Du = nU;
  dynmodel.sub{Nf+1}.A = eye(nU);
  
elseif (strcmp(con,'lag') && delay==0)                               % LAG
  Du = nU; c = exp(-dt/tau);
  dynmodel.sub{Nf+1}.A = kron([1-c c],eye(nU));
  dynmodel.sub{Nf+1}.dyni = Di+(1:nU);
  
elseif strcmp(con,'foh')                                     % delayed FOH
  Du = 2*nU; c = (dt-delay)/(delay+tau-dt);
  dynmodel.sub{Nf+1}.A = kron([0 0 1; 1-c 0 c],eye(nU));
  dynmodel.sub{Nf+1}.dyni = Di+(1:Du);
    
elseif strcmp(con,'lag')                                     % delayed LAG
  Du = 2*nU; c1 = exp(-delay/tau); c2 = exp(-(dt-delay)/tau);
  dynmodel.sub{Nf+1}.A = kron([0 0 1;c1*(1-c2) (1-c1)*(1-c2) c2],eye(nU));
  dynmodel.sub{Nf+1}.dyni = Di+(1:Du);
  
end

if ~(strcmp(con,'zoh') && delay==0)   % no need to augment if standard ZOH
  dynmodel.sub{Nf+1}.fcn = @lin0d;
  dynmodel.sub{Nf+1}.dynu = 1:nU;
  dynmodel.sub{Nf+1}.dyno = Do+(1:Du);
  
  M = [M; zeros(Du,1)]; S = blkdiag(S,1e-8*eye(Du));
  plant.noise = blkdiag(plant.noise,1e-8*eye(Du));
  plant.odei = [odei D+(1:Du)];                         % sort out indices
  plant.dyno = [dyno D+(1:Du)];

  if ~isfield(plant,'angi')
    plant.dyni=[dyni Do+(1:Du)];
    plant.poli=[poli Do+(1:Du)];
    dynmodel.sub{1}.dyni = [dynmodel.sub{1}.dyni Di+(1:Du)];
  else                                            % deal with ruddy angles
    Di=sum(dyni<=Do); plant.dyni=[dyni(1:Di) Do+(1:Du) nU+dyni(Di+1:end)];
    Dp=sum(poli<=Do); plant.poli=[poli(1:Dp) Do+(1:Du) nU+poli(Dp+1:end)];
  
    dyni = dynmodel.sub{1}.dyni; Dii=sum(dyni<=Di);
    dynmodel.sub{1}.dyni = [dyni(1:Dii) Di+(1:Du) Du+dyni(Dii+1:end)];
    for i=2:Nf
      if isfield(dynmodel.sub{i},'dyni')
        dyni = dynmodel.sub{i}.dyni; Dii=sum(dyni<=Di);
        dynmodel.sub{i}.dyni = [dyni(1:Dii) Du+dyni(Dii+1:end)];
      end
    end
  end
end

% If dynmodel.sub{1} has Markov order 2 ----------------------------------
if isfield(plant,'markov') && plant.markov>1
  % deal with Markov-ness through multiple dynamics
end