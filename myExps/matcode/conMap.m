function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, ...
                        dMdp, dSdp, dCdp] = conMap(con, map, policy, m, s)
% conMap is a utility function which linearly maps a state to a different
% state which will be the input states for the controller. It also applies
% the chain rule to find the correct derivatives.
% 
% Paavo Parmas, 2014-04-06

mappings = policy.copyMappings; E=length(policy.maxU); D=length(m); T=length(mappings(:, 1)); j=1:E; i=1:D;
M = zeros(E,1); S = zeros(E); Mm = zeros(T,1); Sm = zeros(T);        % init M and S

if nargout < 4                                       % without derivatives

  [Mm, Sm, Q] = map(policy, m, s);
  [M, S, R] = con(policy, Mm, Sm);   
  C = Q*R;
  
else                                                    % with derivatives
  
  % Mapping -----------------------------------------------------------
  [Mm, Sm, Q, dMmdm, dSmdm, dQdm, dMmds, ...
                  dSmds, dQds] = map(policy, m, s); 

  %Mm
  %Sm
  %Q
 
  % Control -----------------------------------------------------------
  [M, S, R, dMdMm, dSdMm, dRdMm, dMdSm, dSdSm, dRdSm, dMdp, dSdp, dRdp] = con(policy, Mm, Sm); 
  
  dMdm = dMdMm*dMmdm + dMdSm*dSmdm;
  dSdm = dSdMm*dMmdm + dSdSm*dSmdm;
  C = Q*R;
  %dQdm*R
  %Q
  %dRdMm*dMmdm
  %dRdSm*dSmdm
  dCdm = Q*(dRdMm*dMmdm); %+ dRdSm*dSmdm) ;%+ dQdm*R ;
  dMds = dMdMm*dMmds + dMdSm*dSmds;
  dSds = dSdMm*dMmds + dSdSm*dSmds;
  dCds = Q*(dRdMm*dMmds + dRdSm*dSmds); %+dQds*R ;
  dCdp = Q*dRdp;  % Because dQdp is 0
  %R
  %dRdm = dRdMm*dMmdm + dRdSm*dSmdm
  %dRds = dRdMm*dMmdm + dRdSm*dSmdm
  %dQdm
  %dQds
  
  %RR = kron(R',eye(T)) 
  %QQ = kron(eye(T),Q)
  %dCdm = QQ*dRdm + RR*dQdm;
  %dCds = QQ*dRds + RR*dQds;
  %dCdp = QQ*dRdp;% + RR*dQdp;

  
end