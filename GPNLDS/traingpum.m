function [gp, nlml] = train(gp, iter)

% train a GP model with SE covariance function (ARD). First the hypers are
% trained using full GPs. Then, if gp.induce exists, indicating sparse
% approximation, if enough training exmples are present, train the inducing
% inputs. If no inducing inputs are present, then initialize these to be a
% random subset of the training cases.
%
% gp               struct gaussian process dynamics model 
%   hyp      1xE   struct of hyperparameters (ignored on the input)
%     m      Dx1   mean function coefficients
%     b      1x1   mean function bias
%     l      Dx1   ARD log lenghtscale parameters
%     s      1x1   log of signal std dev
%     n      1x1   log of noise std dev
%   inputs   nxD   training inputs
%   target   nxE   training targets
%   induce  Mxdxe  [optional] inducing inputs  
% iter       1x2   [optional] number of training iterations
% nlml       1xE   negative log marginal likelihood for each GP (incl curb)
%
% Carl Edward Rasmussen, Andrew McHutchon 2013-07-05

if nargin < 2, iter = [-500 -1000]; end           % default training iterations
[N, D] = size(gp.inputs); E = size(gp.target,2);           % get variable sizes
nlml = zeros(1,E);

curb.snr = 300; curb.ls = 100; curb.std = std(gp.inputs);       % set hyp curb
[gp.hyp(1:E).l] = deal(log(std(gp.inputs)')); t = log(std(gp.target)); % initialize hyp
s = num2cell(t); [gp.hyp.s] = deal(s{:});
s = num2cell(t-log(10)); [gp.hyp.n] = deal(s{:});

for i = 1:E                                          % train each GP separately
  [gp.hyp(i), v] = minimize(gp.hyp(i), @hypCurbum, iter(1), gp.inputs, ...
                                                         gp.target(:,i), curb);
  nlml(i) = v(end);
end

if isfield(gp,'induce')                  % are we using a sparse approximation?
  [M, d, e] = size(gp.induce);
  if M >= N; return; end     % if too few training examples, we don't need FITC
  
  if d == 0                                % we don't have inducing inputs yet?
    gp.induce = zeros(M,D,e);                                  % allocate space
    for i = 1:e
      j = randperm(N); gp.induce(:,:,i) = gp.inputs(j(1:M),:);  % random subset
    end
  end
  
  [gp.induce, nlml2] = minimize(gp.induce, 'fitc', iter(end), gp);
  fprintf('GP NLML, full: %e, sparse: %e, diff: %e\n', ...
                                  sum(nlml), nlml2(end), nlml2(end)-sum(nlml));
end
