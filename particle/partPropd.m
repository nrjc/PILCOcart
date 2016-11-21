function [X, dXdm, dXdp, dynmodel, minit, x, y, dudm, dudp, dXdoldy] = ...
            partPropd(m, s, plant, dynmodel, policy, oldx, oldy, doldudm, doldudp)
%
% Two cases: 1) m and s are column vectors - this provides an initial
% distribution from which to draw the samples. 2) m and s are D-by-Nsamp 
% matrices - samples have already been taken and need propagating one step
% further
%

% Dimensions
D = size(m,1); Du = length(plant.maxU); Nsamp = plant.Nsamp;
poli = plant.poli; dyni = plant.dyni; dyno = plant.dyno; difi = plant.difi;
if nargin  < 6; oldx = []; oldy = []; else t = size(oldx,2)+1; end
E = length(dyno); Np = numel(unwrap(policy)); dyniu = [dyni D+1:D+Du]; 
M = zeros(E,Nsamp); S = M; dM = zeros(E,Nsamp,t,D+Du); dS = dM; 
dMdoldy = zeros(E,Nsamp,t-1,E);

% Sample if a distribution is given
if ~isempty(s)
    samples = randn(plant.rStream,D,Nsamp); % D-by-Nsamp, standard normal distribution
    if size(s,2) > 1;
        m = bsxfun(@plus,m,chol(s)'*samples); % samples now distributed correctly
    else
        m = bsxfun(@plus,m,bsxfun(@times,sqrt(s),samples)); % samples now distributed correctly
    end
end
minit = m; % E-by-Nsamp

% Get control signal
u = zeros(Du,Nsamp); dudm = zeros(Du,Nsamp,D); dudp = zeros(Du,Nsamp,Np);
mn = m + bsxfun(@times,exp(dynmodel.hyp(end,:))'/sqrt(2),randn(plant.rStream,D,Nsamp)); % Add noise
parfor i=1:Nsamp;
    [u(:,i), ~, ~, dudm(:,i,poli),~,~,~,~,~,dudp(:,i,:)] =...
                        policy.fcn(policy,mn(poli,i),zeros(length(poli)));
end

% Squash the control signal
su = bsxfun(@times,plant.maxU',sin(u));                        % Du-by-Nsamp
dudm = bsxfun(@times,dudm,bsxfun(@times,plant.maxU',cos(u))); % Du-by-Nsamp-by-poli
dudp = bsxfun(@times,dudp,bsxfun(@times,plant.maxU',cos(u))); % Du-by-Nsamp-by-Np

x = [oldx permute([m(dyni,:);su],[1,3,2])];               % dyni+Du-by-t-by-Nsamp
dudm = [doldudm permute(dudm,[1,4,2,3])];               % Du-by-t-by-Nsamp-by-Np
dudp = [doldudp permute(dudp,[1,4,2,3])];               % Du-by-t-by-Nsamp-by-Np
samples = randn(plant.rStream,E,Nsamp);                             % E-by-Nsamp

% Propagate samples
if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce); dynmodel = preCalcFitc(dynmodel); 
else dynmodel = preCalc(dynmodel); end
if isempty(oldy);
    if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce);
        [M, S, ~, dm, ds] = fitcPred(dynmodel,squeeze(x(:,1,:))');
    else
        [M, S, ~, dm, ds] = gprPred(dynmodel,squeeze(x(:,1,:))');
    end
    M = M'; S = S';
    dM(:,:,1,dyniu) = permute(dm,[2,1,4,3]); dS(:,:,1,dyniu) = permute(ds,[2,1,4,3]);
else
    parfor i=1:Nsamp
        if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce)
            [M(:,i) S(:,i), ~, dM(:,i,:,dyniu), dS(:,i,:,dyniu), dMdoldy(:,i,:,:)] = ...
                fitcCond(dynmodel,x(:,:,i)',oldy(:,:,i)');
        else
            [M(:,i) S(:,i), ~, dM(:,i,:,dyniu), dS(:,i,:,dyniu), dMdoldy(:,i,:,:)] = ...
                gpCond(dynmodel,x(:,:,i)',oldy(:,:,i)');
        end
    end
end

dMdm = zeros(E,Nsamp,t,E);
dMdm(:,:,:,poli) = etprod('1234',dM(:,:,:,end-Du+1:end),'1235',dudm,'5324'); % E-by-Nsamp-by-t-by-D
dMdm(:,:,:,dyni) = dMdm(:,:,:,dyni) + dM(:,:,:,dyni); % add on direct derivatives
dSdm(:,:,:,poli) = etprod('1234',dS(:,:,:,end-Du+1:end),'1235',dudm,'5324'); % E-by-Nsamp-by-t-by-D
dSdm(:,:,:,dyni) = dSdm(:,:,:,dyni) + dS(:,:,:,dyni);
dMdp = etprod('123',dM(:,:,:,end-Du+1:end),'1245',dudp,'5423');  % E-by-Nsamp-by-Np
dSdp = etprod('123',dS(:,:,:,end-Du+1:end),'1245',dudp,'5423');  % E-by-Nsamp-by-Np

for i=1:Nsamp
    M(difi,i) = m(difi,i) + M(difi,i);     % we predict differences, E-by-Nsamp
    dMdm(difi,i,end,difi) = squeeze(dMdm(difi,i,end,difi)) + eye(length(difi));
end

% Sample
ssamp = sqrt(S).*samples;
X = M + ssamp; % samples now distributed correctly

dXdm = dMdm + bsxfun(@times,dSdm,ssamp./S)/2; % E-by-Nsamp-by-t-by-D
dXdp = dMdp + bsxfun(@times,dSdp,ssamp./S)/2;
if ~isempty(oldy); dXdoldy = dMdoldy; else dXdoldy = []; end

y = [oldy permute(X-m,[1,3,2])];                               % E-by-t-by-Nsamp

if any(~isreal(X(:))); keyboard; end
