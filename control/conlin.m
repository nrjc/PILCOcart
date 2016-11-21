function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, ...
                                  dMdp, dSdp, dVdp] = conlin(policy, m, s)
% Linear controller with input size D and output size E
%
% policy      policy structure
%   .p         parameters which are modified by training
%     .w        linear weights                                 [ E  x  D ]
%     .b        biases                                         [ E       ]
% m           mean of state distribution                       [ D       ]
% s           covariance matrix of state distribution          [ D  x  D ]
%
% M           mean of the action                               [ E       ]
% S           variance of action                               [ E  x  E ]
% C           inv(s) times input output covariance             [ D  x  E ]
% dMdm        mean output by mean input derivative             [ E  x  D ]
% dSdm        covariance by mean input derivative              [E*E x  D ]
% dCdm        input output covariance by mean input derivative [D*E x  D ]
% dMds        mean by covariance derivative                    [ E  x D*D]
% dSds        covariance by covariance derivative              [E*E x D*D]
% dCds        C by covariance derivative                       [D*E x D*D]
% dMdp        mean by parameters derivative                    [ E  x  P ]
% dSdp        covariance by parameters derivative              [E*E x  P ]
% dCdp        C by parameters derivative                       [D*E x  P ]
%
% where P = (D+1)*E
%
% Copyright (C) by Carl Edward Rasmussen and Marc Deisenroth, 2012-06-25
% Edited by Joe Hall 2012-07-03

w = policy.p.w; b = policy.p.b; [E D] = size(w);% dim of control and state
M = w*m + b;                                                        % mean
S = w*s*w'; S = (S+S')/2;                                     % covariance
V = w';                                   % inv(s)*input-output covariance

if nargout > 3
  dMdm = w;            dSdm = zeros(E*E,D); dVdm = zeros(D*E,D);
  dMds = zeros(E,D*D); dSds = kron(w,w);    dVds = zeros(D*E,D*D);
  
  X=reshape(1:D*D,[D D]); XT=X'; dSds=(dSds+dSds(:,XT(:)))/2; % symmetrise
  X=reshape(1:E*E,[E E]); XT=X'; dSds=(dSds+dSds(XT(:),:))/2;
  
  wTdw =reshape(permute(reshape(eye(E*D),[E D E D]),[2 1 3 4]),[E*D E*D]);
  dMdp = [eye(E) kron(m',eye(E))];
  dSdp = [zeros(E*E,E) kron(eye(E),w*s)*wTdw + kron(w*s,eye(E))];
  dSdp = (dSdp + dSdp(XT(:),:))/2;                            % symmetrise
  dVdp = [zeros(D*E,E) wTdw];
end