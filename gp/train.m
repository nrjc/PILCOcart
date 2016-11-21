function [gp, nlml] = train(gp, iter)

% train a GP model with SE covariance function (ARD). First the hypers are
% trained using full GPs. Then, if gp.induce exists, indicating sparse
% approximation, if enough training exmples are present, train the inducing
% inputs. If no inducing inputs are present, then initialize these to be a
% random subset of the training cases.
%
% gp                 struct gaussian process dynamics model 
%   hyp       1xE    struct of hyperparameters (ignored on the input)
%     m       Dx1    mean function coefficients
%     b       1x1    mean function bias
%     l       Dx1    ARD log lenghtscale parameters
%     s       1x1    log of signal std dev
%     n       1x1    log of noise std dev
%   inputs    nxD    training inputs
%   target    nxE    training targets
%   induce    Mxdxe  [optional] inducing inputs
%   trainMean 1x1    [optional] switch for training mean (1) or keeping it
%                    fixed at its set value (0). Defaults to 1.
% iter       1x2   [optional] number of training iterations
% nlml       1xE   negative log marginal likelihood for each GP (incl curb)
%
% Carl Edward Rasmussen, Andrew McHutchon 2014-03-10

if nargin < 2, iter = [-500 -1000]; end       % default training iterations
if isfield(gp,'trainMean'); mS = gp.trainMean; else mS = 0; end
[N, D] = size(gp.inputs); E = size(gp.target,2);       % get variable sizes
nlml = zeros(1,E);

curb.snr = 300; curb.ls = 100; curb.std = std(gp.inputs);    % set hyp curb
[gp.hyp(1:E).l] = deal(log(std(gp.inputs)')); t = log(std(gp.target)); % initialize hyp
s = num2cell(t); [gp.hyp.s] = deal(s{:});
s = num2cell(t-log(10)); [gp.hyp.n] = deal(s{:});
if mS; [gp.hyp(1:E).m] = deal(zeros(D,1)); [gp.hyp.b] = deal(0); end

if isfield(gp.hyp,'on'); gp.hyp = rmfield(gp.hyp,'on'); end
if isfield(gp.hyp,'pn'); gp.hyp = rmfield(gp.hyp,'pn'); end

for i = 1:E                                      % train each GP separately
  [gp.hyp(i), v] = minimize(gp.hyp(i), @hypCurb, iter(1), gp.inputs, ...
                                                gp.target(:,i), mS, curb);
  nlml(i) = v(end);
end

if isfield(gp,'induce')              % are we using a sparse approximation?
  [M, d, e] = size(gp.induce);
  if M < N;              % only call FITC if we have enough training points
  
   if d == 0                            % we don't have inducing inputs yet?
    gp.induce = zeros(M,D,e);                              % allocate space
    for i = 1:e
      j = randperm(N); gp.induce(:,:,i) = gp.inputs(j(1:M),:); % random subset
    end
   end
  
   [gp.induce, nlml2] = minimize(gp.induce, 'fitc', iter(end), gp);
   fprintf('GP NLML, full: %e, sparse: %e, diff: %e\n', ...
                              sum(nlml), nlml2(end), nlml2(end)-sum(nlml));
  else
   fprintf('GP NLML: %e\n', sum(nlml));
  end
end

% We must now use the single learnt noise level to assign the three 
% separate noise levels: observation noise, process noise, and the noise on
% the GP training targets. The AR GP trained here assumes all the noise is
% observation noise and that the training targets are observations. Thus we
% keep "n" the same, copy it into "on", and set "pn" to zero.
gp.noise = exp(2*[gp.hyp.n]);           % the total amount of noise
[gp.hyp.pn] = deal(-10);                % process noise set to zero
[gp.hyp.on] = deal(gp.hyp.n);           % observation noise

gp = gpPreComp(gp);        % make the precomputations needed for prediction
