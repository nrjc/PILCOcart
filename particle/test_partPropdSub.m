function [f df] = test_partPropdSub(Z, m, s, plant, dynmodel, policy, oldx, oldy, doludm, doldudp,deriv,e,n)

% Z is D-by-t with the current test point as the last column

E = size(dynmodel.hyp,2); Nm = size(m,2);
reset(plant.rStream);
fcn = @partPropJ; dfcn = @partPropdJ;


switch deriv
    case 'dXdm'
%         if Nm > 1; m(:,n) = Z(:,end); oldx(:,n,1:end-1) = permute(Z(:,1:end-1),[1,3,2]);
%         else m = Z(:,end); oldx = Z(:, 1:end-1); end
        m(:,n) = Z(:,end);
%        if ~isempty(oldx); oldx(:,:,n) = Z(:,1:end-1); end;
        if nargout == 1;
            X = fcn(m, s, plant, dynmodel, policy, oldx, oldy);
        else
            [X, dXdm] = dfcn(m, s, plant, dynmodel, policy, oldx, oldy, doludm, doldudp);
            df = permute(dXdm(e,n,end,:),[4,3,2,1]); % D-by-t
        end
        f = X(e,n);         
  
    case 'dXdp'
        policy = rewrap(policy,Z);
        if nargout == 1;
            X = fcn(m, s, plant, dynmodel, policy, oldx, oldy, doludm, doldudp);
        else
            [X, ~, dXdp] = dfcn(m, s, plant, dynmodel, policy, oldx, oldy, doludm, doldudp);
            df = squeeze(dXdp(e,n,:));
        end
        f = X(e,n);
        
    otherwise
        error('Unrecognised derivative. Options are ''dXdm'' or ''dXdp''\n');

end
        
    
        