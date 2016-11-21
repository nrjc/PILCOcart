 function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
                                                 = conFeedfwd(policy, m, ~)
% Function to implement a feed-forward contoller. Control values are set
% explicitly rather than via any policy. Control values are stored in a 
% matrix of size H-by-Du, where Du is the number of control variables. 
% this matrix is stored either in the field policy.p.c or in policy.c
% If the field policy.p.c exists then the control values are optimised
% during learning. If the values are stored in policy.c however they are
% treated as fixed during policy learning and will not be optimised. As
% this is a feedforward controller the input state distribution has no 
% effect of the output control value. Furthermore this controller returns a
% deterministic control value in the M output argument rather than a
% distribution. Gaussian uncertainty can be added around the control value
% by use of the policy.S field. 
%
% Andrew McHutchon, Feb 2014

t = policy.t; % the current time step 

if isfield(policy,'p') && isfield(policy.p,'c'); 
     c = policy.p.c; pc = 1;            % we are optimising control values
else c = policy.c;   pc = 0;            % control values are fixed
end

D = length(m); [Tc, E] = size(c);

% Matrix initialisations
C = zeros(D,E);
dMdm = zeros(E,D); dSdm = zeros(E^2,D); dCdm = zeros(E*D,D);
dMds = zeros(E,D^2); dSds = zeros(E^2,D^2); dCds = zeros(E*D,D^2);

% Set controller output and handle derivatives
if pc;
    M = c(t,:)';       % the mean control value set by c
    
    % derivatives so that controls can be optimised
    np = numel(policy.p.c); dMdp = zeros(E,np);
    I = false(size(policy.p.c)); I(t,:) = true; dMdp(:,I(:)) = eye(E);
    dSdp = zeros(E^2,np); dCdp = zeros(E*D,np);
    
else
    M = c(min(t,Tc),:)'; % if t > size of c then reuse last row of c
    
    % Controls are fixed therefore we have no output derivatives wrt p
    dMdp = zeros(E,0); dSdp = zeros(E^2,0); dCdp = zeros(E*D,0);
end

% Output variance is (nearly) zero unless policy.S exists
if isfield(policy,'S'); S = policy.S; else S = 1e-9*eye(E); end
