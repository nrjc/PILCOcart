function [X, u, minit, dynmodel, x, y] = partProp(m, s, plant, dynmodel, policy, oldx, oldy)
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
E = length(dyno); maxU = plant.maxU;
M = zeros(E,Nsamp); S = M;

% Sample if a distribution is given
if ~isempty(s)
    samples = randn(plant.rStream,D,Nsamp); % D-by-Nsamp, standard normal distribution
    m = bsxfun(@plus,m,chol(s)'*samples); % samples now distributed correctly
    oldx = []; oldy = [];
end
minit = m; % E-by-Nsamp

% Get control signal
u = zeros(Du,Nsamp);
mn = m + bsxfun(@times,exp(dynmodel.hyp(end,:))'/sqrt(2),randn(plant.rStream,D,Nsamp)); % Add noise
f = policy.fcn;                    % Save passing policy to each parallel worker
parfor i=1:Nsamp;
    u(:,i) = maxU'.*sin(f(policy,mn(poli,i),zeros(length(poli)))); 
end

x = [oldx permute([m(dyni,:);u],[1,3,2])];              % dyni+Du-by-t-by-Nsamp
samples = randn(plant.rStream,E,Nsamp);                             % E-by-Nsamp

% Propagate samples
if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce); 
    dynmodel = preCalcFitc(dynmodel); 
else dynmodel = preCalc(dynmodel); end
if isempty(oldy);
    if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce);
        [M S] = fitcPred(dynmodel,squeeze(x(:,1,:))');
    else
        [M S] = gprPred(dynmodel,squeeze(x(:,1,:))');
    end
    M = M'; S = S';
else
    parfor i=1:Nsamp
        if isfield(dynmodel,'induce') && ~isempty(dynmodel.induce)
            [M(:,i) S(:,i)] = fitcCond(dynmodel,x(:,:,i)',oldy(:,:,i)');
        else
            [M(:,i) S(:,i)] = gpCond(dynmodel,x(:,:,i)',oldy(:,:,i)');
        end
    end
end
M(difi,:) = m(difi,:) + M(difi,:);          % we predict differences, E-by-Nsamp
                  
% Sample
X = M + sqrt(S).*samples;                    % samples now distributed correctly

y = X; y(difi,:) = y(difi,:) - m(difi,:);
y = [oldy permute(y,[1,3,2])];                               % E-by-t-by-Nsamp

if any(~isreal(X(:))); keyboard; end

