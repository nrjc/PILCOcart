function [f df] = test_predSub(X,dynmodel,xs,mode,yc,deriv,e,ns,k)
% joint posterior mean is Ns-by-E, joint posterior covariance is
% Ns-by-Ns-by-E. Marginals are both 1-by-E

% Variable being changed by finite difference
if strcmp(deriv,'dMdyc')
    yc(:,e) = X;
elseif strcmp(mode,'conditional') 
    xs = reshape(X,size(xs));
else
    xs(ns,:) = X;
end

% Functions
if isfield(dynmodel,'induce'); 
    if strcmp(mode,'marginal'); fn = @fitcPred; else fn = @fitcCond; end 
else
    if strcmp(mode,'marginal'); fn = @gprPred; else fn = @gpCond; end 
end


% Function calls
if nargout == 1
    
    % No derivatives
    [M S] = fn(dynmodel,xs,yc);
    if strcmp(deriv,'dMdxs')
        if strcmp(mode,'joint'); f = M(end,e); else f = M(e); end
    elseif strcmp(deriv,'dSdxs')
        if strcmp(mode,'joint'); f = S(ns,e,k); else f = S(e); end
    elseif strcmp(deriv,'dMdyc'); f = M(e);
    end
    
else
    
    % With derivatives
    if strcmp(deriv,'dMdxs')
        [M,~,~,dMdxs] = fn(dynmodel,xs,yc);
        if strcmp(mode,'joint');
            f = M(ns,e); df = squeeze(dMdxs(end,ns,:));
        elseif strcmp(mode,'conditional')
            f = M(e); df = reshape(dMdxs(e,:,:),[],1);
        else
            f = M(e); df = squeeze(dMdxs(ns,e,:));
        end
        
    elseif strcmp(deriv,'dSdxs')
        [~,S,~,~,dSdxs] = fn(dynmodel,xs,yc);
        if strcmp(mode,'joint'); 
            f = S(ns,e,k); df = squeeze(dSdxs(ns,e,k,:));
        elseif strcmp(mode,'marginal')
            f = S(e); df = squeeze(dSdxs(ns,e,:));
        elseif strcmp(mode,'conditional')
            f = S(e); df = reshape(dSdxs(e,:,:),[],1);
        end
        
    elseif strcmp(deriv,'dMdyc')
        [M,~,~,~,~,dMdyc] = fn(dynmodel,xs,yc);
         f = M(e); df = reshape(dMdyc(e,:,e),[],1);
        
        
    end
end