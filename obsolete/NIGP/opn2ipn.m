function [lh, dynin, dyniangi] = opn2ipn(lh, plant)
% Function to build the input noise vector from the current output noise
% hyperparameters. 
%
% This version assumes the input vector is made up of:
%  x = [<subset of output variables> <trigauged variables> <controls>]
% some of the output variables are predicted via predicted differences and
% so have sqrt(2) times as much noise on them. This needs to be removed for
% when they are considered inputs.
%
% The trigauged variables are much harder to deal with. The angles (or
% their differences) are predicted outputs. When they are used as inputs
% however they are converted to sine and cosine representation. The sines
% and cosines are added to the end of the input vector, in the order that
% the angles appear in the output vector, but before the control variables.
% Not all trigauged angles might be used as inputs - some of them might be
% used for the policy or loss function. Therefore we must compare dyni and
% angi to work out the correct angles. A simplifying assumption made here 
% is that sin(x) and cos(x) have the same level of noise on them as x does.
%
% Inputs
%   lh         struct of log hyperparameters
%     .seard   D+2-by-E squared exp with ARD log hyperparameters
%     .lsipn   dimU-by-1 log input noise on controls
%   plant      struct of variables indexes
%
% Andrew McHutchon, 16/01/2012

lsn = lh.seard(end,:)';                              % log output noise values
dyni = plant.dyni; difi = plant.difi; angi = plant.angi;  % subsets of dyno
lsipn = zeros(length(dyni),1);              % Initialise input noise vector

% 1) Non-angle variables
Do = length(plant.dyno);                   % The number of output variables
dynin = dyni(dyni<=Do);   % input variables not inc. trigauged or controls
Dna = length(dynin);         % # of ip vars not inc. trigauged or controls
[jnk,difidyni] = intersect(dyni,difi); % Input vars predicted as differences

lsipn(1:length(dynin)) = lsn(dynin);                   % Normal variables
lsipn(difidyni) = lsipn(difidyni) - 0.5*log(2);            % difi variables

% 2) Angle variables
trigi = reshape(dyni(dyni>Do)-Do,2,[]);                  % 2-by-# of angles
Da = size(trigi,2);                      % # of angles which appear in dyni
dyniangi = angi(trigi(2,:)/2);            % Index into dyno of input angles
[jnk,difiangi] = intersect(dyniangi,difi);             % Input angles in difi

lsipnang = lsn(dyniangi);   % ip noise on input angle variables not in difi
lsipnang(difiangi) = lsipnang(difiangi) - 0.5*log(2);  % Now also difi ones

lsipn(Dna+1:2:Dna+2*Da-1) = lsipnang;  % sin put in correct place in vector
lsipn(Dna+2:2:Dna+2*Da) = lsipnang;    % cos put in correct place in vector

% 3) Controls
lh.lsipn = [lsipn;lh.lsipn]; % add on ip noise for controls
