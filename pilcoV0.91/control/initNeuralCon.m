function policy = initNeuralCon(poli, maxU, N)
% function to initialise a neural network controller into a simple set up:
%
%     D-dimensional              1-dimensional             E-dimensional
%
%       |---> [ linear controller 1 ] ---> [ RBF controller 1]  --->|
%       |---> [ linear controller 2 ] ---> [ RBF controller 2]  --->| S
%  x ---|---> [ linear controller 3 ] ---> [ RBF controller 3]  --->| U ---> U   
%       |---> [ linear controller 4 ] ---> [ RBF controller 4]  --->| M
%       |---> [ linear controller 5 ] ---> [ RBF controller 5]  --->|
%
% In this setup the linear controller stages each take in the full state and 
% output a one dimensional quantity, which is passed to the RBF
% controllers. You can specify the intermediate dimension to be as large or
% small as you like. This setup also passes all 'poli' variables to each of
% the 'neurons', but again this can be changed and only a subset passed in,
% where the subset can change per neuron.
%
% The basis function locations are fixed to be linearly spaced between -1
% and 1. This helps training as they don't have to be optimised. The linear
% controllers could be left to learn to keep their output in the relevant part 
% of the state space, but here we use a saturation function on each of the
% linear controllers (gSin).
%
% The policy struct is organised as follows
%
%                                      gSat
%                                       |
%                                     conAdd
%                                       |
%        --------------------------------------------------------------
%        |               |              |             |               |
%       {1}             {2}            {3}           {4}             {5}
%        |               |              |             |               | 
%     conChain        conChain       conChain      conChain        conChain
%     |      |        |      |       |      |      |      |        |      |
%    {1}    {2}      {1}    {2}     {1}    {2}    {1}    {2}      {1}    {2}
%     |      |        |      |       |      |      |      |        |      |
%  conlin conGauss conlin conGauss  ...    ...  conlin conGauss conlin conGauss
%
% Each of the above controllers could be changed to something different,
% plus any number of controllers could be chained together in each neuron,
% any number of extra layers could be added with further conChain and
% conAdd ... the combinations are (almost) endless!
%
% Andrew McHutchon, October 2012

if nargin < 3; N = 5; end                                    % Number of neurons
D = length(poli);                           % Dimension of input to linear stage
Di = 1;                                  % Dimension of output from linear stage
E = length(maxU);                                           % Number of controls
Nb = 5;                           % Number of basis functions per RBF controller
cen = linspace(-1,1,Nb)';         % RBF basis functions spread linearly, -1 to 1
ll = repmat(log(mean(diff(cen))/2),1,E);   % RBF stds set to be half cen spacing

policy.fcn = @(policy,m,s)conCat(@conAdd,@gSat,policy,m,s); % top level function
policy.maxU = maxU;

% Loop over neurons
for i=1:N
    policy.sub{i}.fcn = @conChain;     % function to join the linear & RBF parts
    policy.sub{i}.poli = poli;      % the input vars for each neuron set to poli
    
    % The linear controller                      the first function in the chain
    policy.sub{i}.sub{1}.fcn = @(policy,m,s)conCat(@conlin,@gSin,policy,m,s);  
    policy.sub{i}.sub{1}.maxU = 1;               % saturation function amplitude
    policy.p{i}{1}.w = 1e-3*randn(Di,D);    
    policy.p{i}{1}.b = 1e-3*randn(Di,1);
    
    % The Gaussian RBF controller
    policy.sub{i}.sub{2}.fcn = @conGaussd;    % the second function in the chain
    policy.sub{i}.sub{2}.cen = cen;
    policy.sub{i}.sub{2}.ll = ll;
    policy.p{i}{2}.w = 1e-3*randn(Nb,E);
end