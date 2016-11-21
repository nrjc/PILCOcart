function d = test_partPropd(m, s, plant, dynmodel, policy, oldx, oldy, doludm, doldudp, deriv)

E = size(dynmodel.target,2); Nsamp = plant.Nsamp; szm = size(m,2);
d = zeros(E,Nsamp);
delta = 1e-3;
%if ~isempty(oldx); x = [oldx permute(m,[1,3,2])]; else x = m; end;   % D-by-t-by-Nsamp

switch deriv
    case 'dXdm'
        for n = 1:Nsamp
            for e = 1:E
                d(e,n) = checkgrad('test_partPropdSub',m(:,min(szm,n)),delta,...
                             m,s,plant, dynmodel, policy, oldx, oldy, doludm, doldudp, deriv,e,n);
            end
        end
        
    case 'dXdp'
        p = unwrap(policy);
        for n = 1:Nsamp
            for e = 1:E
                d(e,n) = checkgrad('test_partPropdSub',p,delta,m,s,plant, ...
                                        dynmodel, policy, oldx, oldy, doludm, doldudp, deriv,e,n);
            end
        end
        
    otherwise
        error('Unrecognised derivative, options are ''dXdm'' or ''dXdp''');
end
            