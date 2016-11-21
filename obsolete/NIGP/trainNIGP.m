function [dynmodel fx] = trainNIGP(dynmodel, plant, Nls, mode)
% [lhyp fx] = trainNIGPi(x,y,Nls,lhm,lsipn)
% Function to train the NIGP model by an iterative method. For each
% iteration the slope values used to refer the input noise to the output
% are fixed and the GP hyperparameters (including the input noise variance)
% are trained. The slope values are then recalculated and used for the next
% iteration. The number of linesearches in each iteration is a parameter
% which can be set and can influence the performance of the algorithm.
% The function keeps track of the best marginal likelihood found
% and returns the associated hyperparameters. If you notice the NLML is
% oscillating try reducing the number of linesearches in each iteration.
%
% Inputs:
%   dynmodel          struct of training data
%         .inputs     input training data matrix, N-by-D
%         .target     training targets matrix, N-by-E
%         .hyp        (optional) initial hyp settings
%   plant             struct of variable indexes
%   Nls               (optional) max total number of linesearches
%
% Outputs:
%   dynmodel    the trained hyperparameters returned in a struct:
%      .hyp     the trained log GP kernel hyperparameters
%      .nigp    the ip noise corrective diagonal terms to add to K
%
% For more information see the NIPS paper: Gaussian Process Training with
% Input Noise; McHutchon and Rasmussen, 2011. 
% http://mlg.eng.cam.ac.uk/?portfolio=andrew-mchutchon
%
% July 2012, Andrew McHutchon

format compact;
if nargin < 2; error('trainNIGP needs plant as well!\n'); end
if isfield(dynmodel,'dyni');              % handle multiple dynamics model setup
    plant.dyni = dynmodel.dyni; plant.dyno = plant.dyno(dynmodel.dyno); 
    plant.angi = dynmodel.angi;
end

x = dynmodel.inputs; y = dynmodel.target;
E = size(y,2); dimU = size(x,2)-length(plant.dyni);

% Set some defaults
if nargin < 3; Nls = 50; mode = 0; end      % total number of linesearches 
if length(Nls) == 1; Nls(2) = 50; end     % # of linesearches per iteration
if nargin < 4 || isempty(mode); mode = 0; end
options.verbosity = 1; fx = zeros(1,Nls(1));
if isfield(dynmodel,'hyp') && any(exp(dynmodel.hyp(end-1,:)-dynmodel.hyp(end,:))<0.1);
    dynmodel = rmfield(dynmodel,'hyp');  % If initial SNR is < 0.1, try new hypers
end

% Initial hyperparameters
if isfield(dynmodel,'hyp')
    lh.seard = dynmodel.hyp;
else
    lell = log(std(x,[],1))'; lsf = log(std(y,[],1)); lsn = lsf - log(10);
    lh.seard = [repmat(lell,1,E);lsf;lsn];                       % D+2-by-E
end
lh.lsipn = log(std(x(:,end-dimU+1:end),[],1)/10)';    % controls, dimU-by-1

fprintf('Started NIGP training: ');
bestfx = Inf; nigp.y = y; idx = 1; Lused = 0;
while Lused < abs(Nls(1))
    options.length = sign(Nls(2))*min(abs(Nls(1)) - Lused, abs(Nls(2)));
    
    % Update slopes
    fprintf('calculating slopes\r');
    if ~isfield(nigp,'df2')
        nigp = calcInitdf2(lh,x,plant,dynmodel,mode);          % First time
    else
        nigp = calcNewdf2(lh,x,plant,nigp,mode);         % Subsequent times
    end
        
    % Find next estimate of hyper-parameters
    [lh fX lused] = minimize(lh,'hypCurbNIGP',options,nigp.df2,x,y,plant);
    
    fx(idx) = fX(end); Lused = Lused + lused;
    if fx(idx) < bestfx; bestfx = fx(idx); hyp = lh; hyp.df2 = nigp.df2; end
    if idx > 1 && abs((fx(idx)-fx(idx-1))/fx(idx)) < 1e-6; break; end
    idx = idx + 1;
end
dynmodel.hyp = hyp.seard; hyp = opn2ipn(hyp,plant); fx = fx(1:idx-1);
dynmodel.nigp = sum(bsxfun(@times,hyp.df2,permute(exp(2*hyp.lsipn),[3,2,1])),3);
if any(dynmodel.hyp(end,:)<-10); % Occaisionally training can lead to way too small
    dynmodel = rmfield(dynmodel,'hyp'); % noise values. Usually reinitialising
    fprintf('Too low noise level, restarting training\n'); % solves this problem
    [dynmodel fx] = trainNIGP(dynmodel,plant,Nls,mode); return;
end

% FITC training
if isfield(dynmodel,'induce')            % are we using a sparse approximation?
  [N D] = size(dynmodel.inputs); [M uD uE] = size(dynmodel.induce);
  if M >= N; return; end     % if too few training examples, we don't need FITC
  if length(Nls) == 2; Nls(3) = -750; end % # of evals to train pseudo points
  fprintf('Starting FITC training\r');
  
  if uD == 0                               % we don't have inducing inputs yet?
    dynmodel.induce = zeros(M,D,uE);                           % allocate space
    for i = 1:uE 
      j = randperm(N);
      dynmodel.induce(:,:,i) = dynmodel.inputs(j(1:M),:);       % random subset
    end
  end
    
  [dynmodel.induce nlml2] = minimize(dynmodel.induce, 'fitc', Nls(3), dynmodel);
  fprintf('GP NLML, full: %e, sparse: %e, diff: %e\n', ...
                                  bestfx, nlml2(end), nlml2(end)-bestfx);
end


%%
function nigp = calcInitdf2(lh,x,plant,dynmodel,mode)
% Function calculate the slopes of the posterior at the training points for
% the first time, t_1. If there is no dynmodel.noise field, i.e. no previous
% estimate of the input noise and slopes, then the slopes for t_0 are set
% to zero, and the slopes at t_1 calculated based on this. If the
% dynmodel.noise field does exist then it is used as dipK_t0. As
% dynmodel.noise will be of a smaller size to the current training matrix
% (as it is based on data from a previous policy training step) the slopes
% at t_1 are based on predictions just using the previous training data.
% That is, if N = size(dynmodel.noise,1), then the slopes are based on
% predictions using {x(1:N,:), y(1:N,:)}.

if ~isfield(dynmodel,'noise');           % No initial ip noise information, 
    [N D] = size(x); nigp.y = dynmodel.target; % so use zero slopes to calc 
    nigp.df2 = zeros(N, size(nigp.y,2), D);      % inital fit from which we 
    nigp = calcNewdf2(lh,x,plant,nigp,mode);     % calculate initial slopes
    return;
end
ipK = dipK2ipK(dynmodel.noise);     % We have initial ip noise information
if 0==mode;                          % Function to calculate the derivative
    dffunc = @calcdf2u;   
elseif 1==mode
    dffunc = @calcdf2m;  
else
    dffunc = @calcdf2; 
end
lh = opn2ipn(lh,plant); N = size(ipK,1); nigp.y = dynmodel.target(1:N,:);
nigp = calcNigp(lh,[],x(1:N,:),nigp,ipK);    % R & alpha using pre-calc ipK
df2 = dffunc(lh, x(1:N,:), nigp, x);                        % Calculate df2
nigp.y = dynmodel.target;
nigp = calcNigp(lh,df2,x,nigp); % With df2 => update R & alpha with new slopes
%%
function nigp = calcNewdf2(lh,x,plant,nigp,mode)

if 0==mode;                          % Function to calculate the derivative
    dffunc = @calcdf2u;   
elseif 1==mode
    dffunc = @calcdf2m;  
else
    dffunc = @calcdf2; 
end 
lh = opn2ipn(lh,plant);
nigp = calcNigp(lh,[],x,nigp);    % With [] => update R & a with new hypers
df2 = dffunc(lh, x, nigp);                                  % Calculate df2
nigp = calcNigp(lh,df2,x,nigp);  % With df2 => update R & a with new slopes
%%
function nigp = calcNigp(lh,df2,x,nigp,ipK)
% Function to calculate precomputable variables from a change in
% hyperparameters, or df2
N = size(x,1);

% XmX
if ~isfield(nigp,'XmX') || size(nigp.XmX,1) ~= N;       % Only if necessary
    nigp.XmX = bsxfun(@minus,permute(x,[1,3,2]),permute(x,[3,1,2])); % N-by-N-by-D
end

% K
nigp.K = calcK(lh.seard, nigp.XmX.^2);

% ipK
if nargin < 5
    if isempty(df2); df2 = nigp.df2; else nigp.df2 = df2; end
    ipK = df2toipK(lh, df2);  % turn df2 into ipK - df2 is NOT recalculated
end

% R and alpha
sn2I = bsxfun(@times,permute(exp(2*lh.seard(end,:)),[1,3,2]),eye(N)); % sn2
Kn = nigp.K + sn2I + ipK;                         % Noisy covariance matrix
nigp.R = chol3(Kn); nigp.alpha = findAlpha(nigp.R,nigp.y); % chol(Kn) & alpha