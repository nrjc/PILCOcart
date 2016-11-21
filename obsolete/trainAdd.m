function [gpmodel nlml] = trainAdd(gpmodel, plant, iter)

% train a GP model with additive SE covariance functions (ARD). Two orders of 
% kernel can be used: 1-dimensional and D-dimensional. Lengthscale and singal 
% std dev hyperparameters are trained for each kernel. If dynmodel.induce 
% exists, indicating sparse approximation, and if enough training exmples are 
% present, train the inducing inputs. If no inducing inputs are present, then 
% initialize these to be a random subset of the training cases.
% The hyperparameters are stacked into a 3D-by-E matrix as follows:
%       [ D-by-E lengthscales for 1D kernels   ]
%       [ D-by-E lengthscales for DD kernel    ] 
%       [ D-by-E signal std dev for 1D kernels ]
%       [ 2-by-E signal std dev for DD kernel  ]
%       [ 1-by-E noise variances               ]
%
% Carl Edward Rasmussen, Andrew McHutchon & Marc Deisenroth, 2012-01-12

% Control which orders of interaction are used
%covfunc = {'covSum', {'covSEard1D', 'covNoise'}}; % Just 1D interactions
covfunc = {'covSum', {'covSEard1D','covSEard', 'covNoise'}}; % Both 1D & DD

if nargin < 3, iter = [-500 -1000]; end           % default training iterations

D = size(gpmodel.inputs,2); E = size(gpmodel.target,2);  % get variable sizes
curb.snr = 100; curb.ls = 100; curb.std = std(gpmodel.inputs); % set hyp curb
nlml = zeros(1,E); lh = [];

oneD = any(strcmp('covSEard1D',covfunc{2})); DD = any(strcmp('covSEard',covfunc{2}));
if ~isfield(gpmodel,'hyp');
    if oneD; lh = repmat(log(std(gpmodel.inputs))',1,E); end % Lengthscales for 1D terms
    if DD; lh = [lh;repmat(log(std(gpmodel.inputs))',1,E)]; end % Lengthscales for DD terms
    if oneD; lh = [lh;repmat(log(std(gpmodel.target)),D,1)]; end % sf for 1D terms
    if DD; lh = [lh;log(std(gpmodel.target))]; end             % sf for DD terms
    lh = [lh; log(std(gpmodel.target)/curb.snr)];                % noise std dev
    gpmodel.hyp = zeros(size(lh));
else
    lh = gpmodel.hyp;
end
opt.length = iter(1); opt.verbosity = 1;
inputs = gpmodel.inputs; target = gpmodel.target;
if oneD && DD; curb.std = repmat(curb.std,1,2); end

for i = 1:E                                          % train each GP separately  
  [lh(:,i) v] = minimize(lh(:,i), @hypCurb, opt, covfunc, inputs, target(:,i),curb);
  nlml(i) = v(end);
end
gpmodel.hyp = lh;

if isfield(gpmodel,'induce')            % are we using a sparse approximation?
  [N D] = size(gpmodel.inputs); [M uD uE] = size(gpmodel.induce);
  if M >= N; return; end     % if too few training examples, we don't need FITC

  if uD == 0                               % we don't have inducing inputs yet?
    gpmodel.induce = zeros(M,D,uE);                           % allocate space
    for i = 1:uE 
      j = randperm(N);
      gpmodel.induce(:,:,i) = gpmodel.inputs(j(1:M),:);       % random subset
    end
  end

  dyn = struct; W = gpmodel.induce;
for i=1:E;
    dyn(i).inputs = gpmodel.inputs; dyn(i).hyp = gpmodel.hyp(:,i);
    dyn(i).target = gpmodel.target(:,i);
end
nlml2 = cell(1,E); fv = zeros(1,E); opt.verbosity = 0; opt.length = iter;
  
parfor i=1:E
    [W(:,:,i) nlml2{i}] = minimize(W(:,:,i), 'fitc', opt, dyn(i));
    fv(i) = nlml2{i}(end);
    fprintf('GP NLML variable %i, full: %e, sparse: %e, diff: %e\n', ...
                               i, nlml(i), fv(i), fv(i)-nlml(i));
end
gpmodel.induce = W;
fprintf('GP NLML, full: %e, sparse: %e, diff: %e\n', ...
                                  sum(nlml), sum(fv), sum(fv)-sum(nlml));
  
end