function [Cz dynmodel dCzdm dCzds] = calcCz(dynmodel,m,s)

try chol(s); catch; fprintf('calcCz: input matrix not positive definite\n'); keyboard; end

D = size(m,1)/2; E = size(dynmodel.target,2); N = size(dynmodel.inputs,1); 
Cz = zeros(E); sf2 = exp(2*dynmodel.hyp(end-1,:));                      % 1-by-E

% Calculate beta and iK
if ~isfield(dynmodel,'iK')
    if isfield(dynmodel,'induce') && size(dynmodel.induce,2) ~= 0;
        dynmodel = preCalcFitc(dynmodel); iK = dynmodel.iK2; 
    else
        iK = zeros(N,N,E); dynmodel = preCalcDyn(dynmodel); 
        for k=1:E; iK(:,:,k) = solve_chol(dynmodel.R(:,:,k),eye(N)); end
    end
    dynmodel.iK = iK;
end

for k = 1:E
    if isfield(dynmodel,'induce') && size(dynmodel.induce,2) ~= 0;
         x = dynmodel.induce(:,:,min(k,size(dynmodel.induce,3)));
    else x = dynmodel.inputs;
    end
    
    iL = diag(exp(-2*dynmodel.hyp(1:D,k)));  
    xm1 = bsxfun(@minus,x,m(1:D)'); xm2 = bsxfun(@minus,x,m(D+1:end)');      

    % Covariance between y*(k) and y**(k)
    % Expectation of k(x*, x**)
    iS1 = [iL -iL;-iL iL]; 
    c = sf2(k)/sqrt(det(s*iS1 + eye(2*D)));
    t1 = c*exp(-0.5*m'*iS1/(s*iS1 + eye(2*D))*m);

    % Expectation of k* iK k**
    iSl = [iL zeros(D);zeros(D) iL];
    iSil = iSl/(s*iSl + eye(2*D));
        
    % Calculate [xi-m1, xj-m2] * iSIL * [xi-m1; xj-m2]
%      for i = 1:N
%          for j=1:N
%              e1(i,j) = [xm1(i,:) xm2(j,:)]*iSIL*[xm1(i,:)'; xm2(j,:)'];
%          end
%      end
%      e = exp(-0.5*e1);
        
      iSILa = iSil(1:D,1:D); iSILb = iSil(D+1:end,D+1:end); iSILc = iSil(1:D,D+1:end);
      xm1axm1 = sum(xm1*iSILa.*xm1,2); xm2bxm2 = sum(xm2*iSILb.*xm2,2); xm1cxm2 = xm1*iSILc*xm2';
      e = exp(-0.5*(bsxfun(@plus,xm1axm1,xm2bxm2') + 2*xm1cxm2));
        
      c = sf2(k)^2/sqrt(det(s*iSl + eye(2*D)));
      t2 = c*sum(sum((dynmodel.iK(:,:,k)).*e));
        
      % Covariance between z* and z**
      Cz(k,k) = t1 - t2;      
end

dCzdm = zeros(E,E,2*D); dCzds = zeros(E,E,2*D,2*D);