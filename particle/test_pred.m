function test_pred(dynmodel,xs,yc,deriv,mode)

E = size(dynmodel.target,2); Ns = size(xs,1);
delta = 1e-6;
if nargin < 5; mode = 'marginal'; end

switch deriv
    case 'dMdxs'
        if strcmp(mode,'conditional')
           % For testing conditional posterior
          d = zeros(E,1);
           for e = 1:E
              d(e) = checkgrad('test_predSub',xs(:),delta,dynmodel,xs,mode,yc,deriv,e);
           end
        else
        d = zeros(E,Ns);
        for e=1:E
          for ns = 1:Ns
            d(e,ns) = checkgrad('test_predSub',xs(ns,:),delta,dynmodel,xs,mode,yc,deriv,e,ns);
          end
        end
        end
        
    case 'dSdxs'
        switch mode
        % For testing joint posterior
        case 'joint'
          d = zeros(Ns,Ns,1);
          for i = 1:Ns
           for j = 1:Ns
            for k = 1:2
              d(i,j,k) = checkgrad('test_predSub',xs(i,:),delta,dynmodel,xs,mode,yc,deriv,i,j,k);
            end
           end
          end

        % For testing marginal posterior
         case 'marginal'
         d = zeros(E,Ns);
         for e=1:E
           for ns = 1:Ns
             d(e,ns) = checkgrad('test_predSub',xs(ns,:),delta,dynmodel,xs,mode,yc,deriv,e,ns);
           end
         end
         
         case 'conditional'
         % For testing conditional posterior
          d = zeros(E,1);
           for e = 1:E
              d(e) = checkgrad('test_predSub',xs(:),delta,dynmodel,xs,mode,yc,deriv,e);
           end
        end
         
    case 'dMdyc'
        if ~strcmp(mode,'conditional'); 
            fprintf('Changing mode to ''conditional''\n'); mode = 'conditional';
        end
        d = zeros(E,1);
        for e = 1:E
            d(e) = checkgrad('test_predSub',yc(:,e),delta,dynmodel,xs,mode,yc,deriv,e);
        end
        
        
    otherwise
        error('Unrecognised derivative, options are dMdxs or dSdxs');
end
disp(d);