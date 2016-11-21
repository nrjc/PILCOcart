function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] ...
                                                        = conChain(policy, m, s)
% conChain is a utility function which chains a number of controllers
% together, such that the output of one is the input to the next.
%
% Inputs
%   policy          policy struct
%      .fcn         @conChain - called to arrive here
%      .sub{n}      cell array of sub controllers to chain together
%         .fcn      handle to sub function
%         .< >      all fields in sub will be passed onto the sub function
% 
% Andrew McHutchon, 28th September 2012

Nc = length(policy.sub); % Number of controllers to chain
Np = 0; D = length(m);

for n = 1:Nc
    
    pol = policy.sub{n}; pol.p = policy.p{n};
    np = length(unwrap(pol.p));
    if nargout < 4                                       % without derivatives

        [m, s, c] = pol.fcn(pol, m, s);
        
        if 1 == n; C = c; else C = C*c; end
         
    else                                                    % with derivatives
        
        [m, s, c, mdm, sdm, cdm, mds, sds, cds, mdp, sdp, cdp] = pol.fcn(pol, m, s);
        
        if 1 == n; 
            C = c;
            dMdm = mdm; dSdm = sdm; dCdm = cdm; 
            dMds = mds; dSds = sds; dCds = cds;
            dMdp = mdp; dSdp = sdp; dCdp = cdp;
        
        
        else
            C = Co*c; 
            [Dn E] = size(c);  
        
            dMdm = mdm*Mdm + mds*Sdm; dMds = mdm*Mds + mds*Sds;
            dSdm = sdm*Mdm + sds*Sdm; dSds = sdm*Mds + sds*Sds;
            dcdm = cdm*Mdm + cds*Sdm; dcds = cdm*Mds + cds*Sds;
            
            kCI = kron(eye(E),Co); kIc = kron(c',eye(D));
            dCdm = kCI*dcdm + kIc*Codm; dCds = kCI*dcds + kIc*Cods;
                 
            dMdp = zeros(E,Np+np); dSdp = zeros(E^2,Np+np); 
            dcdp = zeros(Dn*E,Np+np); dCdp = zeros(D*E,Np+np);
            
            dMdp(:,1:Np) = mdm*Mdp + mds*Sdp; dMdp(:,Np+(1:np)) = mdp;
            dSdp(:,1:Np) = sdm*Mdp + sds*Sdp; dSdp(:,Np+(1:np)) = sdp;
            dcdp(:,1:Np) = cdm*Mdp + cds*Sdp; dcdp(:,Np+(1:np)) = cdp;
            dCdp(:,1:Np) = kIc*Cdp; dCdp = dCdp + kCI*dcdp;
        end
        Co = C;
        Mdm = dMdm; Sdm = dSdm; Codm = dCdm;
        Mds = dMds; Sds = dSds; Cods = dCds;
        Mdp = dMdp; Sdp = dSdp; Cdp = dCdp;
    end
    Np = Np + np;
end

M = m; S = s;