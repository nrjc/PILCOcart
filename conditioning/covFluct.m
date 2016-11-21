function [C dynmodel dCdm dCds] = covFluct(dynmodel, m, s)

% covFluct: compute the covariance of the fluctuations under a posterior
% Gaussian process at two inputs with joint N(m, s), and its derivatives.
%
% Andrew McHutchon and Carl Edward Rasmussen, 2012-05-29

try chol(s); catch; 
  fprintf(['covFluct: input covariance matrix not pos def\n']); keyboard;
end

D = size(m,1)/2; E = size(dynmodel.target,2); N = size(dynmodel.inputs,1); 
C = zeros(E,1); sf2 = exp(2*dynmodel.hyp(end-1,:));                      % 1-by-E
if nargout > 2, 
    dCdm = zeros(E,E,2*D); dCds = zeros(E,E,2*D,2*D); 
    dCdmt = zeros(E,2*D); dCdst = zeros(E,2*D,2*D);
end 

if ~isfield(dynmodel,'iK')                              % calculate beta and iK
  if isfield(dynmodel,'induce') && size(dynmodel.induce,2)
    dynmodel = preCalcFitc(dynmodel); iK = dynmodel.iK2; 
  else
    iK = zeros(N,N,E); dynmodel = preCalcDyn(dynmodel); 
    for k = 1:E; iK(:,:,k) = solve_chol(dynmodel.R(:,:,k), eye(N)); end
  end
  dynmodel.iK = iK;
end

for k = 1:E                                              % loop over GP outputs
  if isfield(dynmodel,'induce') && size(dynmodel.induce,2)
    x = dynmodel.induce(:,:,min(k,size(dynmodel.induce,3)));
    N = size(x,1);
  else
    x = dynmodel.inputs;
  end
  iL = diag(exp(-2*dynmodel.hyp(1:D,k)));  

  iS = [iL -iL;-iL iL];                             % expectation of k(x*, x**)
  c1 = sf2(k)/sqrt(det(s*iS + eye(2*D)));
  iSiS = iS/(s*iS + eye(2*D));
  miS = m'*iSiS;
  e1 = exp(-0.5*miS*m); t1 = c1*e1;

  iSl = blkdiag(iL,iL); iSil = iSl/(s*iSl+eye(2*D)); % expectation of k* iK k**
  xm1 = bsxfun(@minus,x,m(1:D)'); xm2 = bsxfun(@minus,x,m(D+1:end)');      
  A = xm1*iSil(1:D,1:D); B = xm2*iSil(D+1:end,D+1:end);
  xm1axm1 = sum(A.*xm1,2); xm2bxm2 = sum(B.*xm2,2); 
  E = xm1*iSil(1:D,D+1:end); F = iSil(1:D,D+1:end)*xm2'; xm1cxm2 = E*xm2';
  e = exp(-bsxfun(@plus,xm1axm1,xm2bxm2')/2 - xm1cxm2);
  c = sf2(k)^2/sqrt(det(s*iSl + eye(2*D)));
  eiK = dynmodel.iK(:,:,k).*e;
  iKe = sum(sum(eiK));
  t2 = c*iKe;
      
  %C(k,k) = t1 - t2;             % covariance between the posterior fluctuations
  C(k) = t1 - t2;

  if nargout > 2                                         % compute derivatives?
    Oeik = ones(1,N)*eiK; eikO = eiK*ones(N,1);
    
    dCdm1 = -c*(A'*eikO + F*Oeik'); dCdm2 = -c*(E'*eikO + B'*Oeik');
    %dCdm(k,k,:) = squeeze(dCdm(k,k,:))-t1*iS/(s*iS+eye(2*D))*m;
    dCdmt(k,:) = [dCdm1;dCdm2]-t1*iSiS*m;
    
    dc1 = -0.5*c1*(iSiS)';
    de1 = 0.5*e1*miS'*miS;
    dt1 = dc1*e1 + c1*de1;
    
    dc = -0.5*c*(iSl/(s*iSl + eye(2*D)))';
    
%     diSil = bsxfun(@times,permute(iSil,[1,3,2]),permute(iSil,[3,2,4,1]));
%     dA = etprod('1234',xm1,'15',diSil(1:D,1:D,:,:),'5234');
%     dB = etprod('1234',xm2,'15',diSil(D+1:end,D+1:end,:,:),'5234');
%     dxm1axm1 = sum(bsxfun(@times,dA,xm1),2); dxm2bxm2 = sum(bsxfun(@times,dB,xm2),2);
%     dE = etprod('1234',xm1,'15',diSil(1:D,D+1:end,:,:),'5234');
%     dxm1cxm2 = etprod('1234',dE,'1534',xm2,'25');
%     bsx = bsxfun(@plus,dxm1axm1,permute(dxm2bxm2,[2,1,3,4]));
%     de = bsxfun(@times,eiK,-bsx/2 - dxm1cxm2);
%     diKe = -squeeze(sum(sum(de)));
%     dt2 = c*diKe + dc*iKe;
%     %dCds(k,k,:,:) = dt1 - dt2;
%     dCdst(k,:,:) = dt1 - dt2;
    
%    diKe2 = zeros(2*D); diKe2a = zeros(D); diKe2c = diKe2a; 
%    diKe2ct = diKe2a; diKe2b = diKe2a;
%     for i = 1:D
%         for j=1:D
%             bsx = bsxfun(@plus,A(:,i).*A(:,j),F(i,:).*F(j,:));
%             %diKe2(i,j) = sum(sum((bsx/2+A(:,i)*F(j,:)).*eiK));
%             diKe2(i,j) = etprod('1',bsx/2+A(:,i)*F(j,:),'23',eiK,'23');
%             
%             bsx = bsxfun(@plus,A(:,i).*E(:,j),F(i,:).*B(:,j)');
%             %diKe2(i,D+j) = sum(sum((bsx/2+A(:,i)*B(:,j)').*eiK));
%             diKe2(i,D+j) = etprod('1',bsx/2+A(:,i)*B(:,j)','23',eiK,'23');
% 
%             bsx = bsxfun(@plus,E(:,i).*A(:,j),B(:,i)'.*F(j,:));
%             %diKe2(D+i,j) = sum(sum((bsx/2+E(:,i)*F(j,:)).*eiK));
%             diKe2(D+i,j) = etprod('1',bsx/2+E(:,i)*F(j,:),'23',eiK,'23');
%  
%             bsx = bsxfun(@plus,E(:,i).*E(:,j),B(:,i)'.*B(:,j)');
%             %diKe2(D+i,D+j) = sum(sum((bsx/2+E(:,i)*B(:,j)').*eiK));
%             diKe2(D+i,D+j) = etprod('1',bsx/2+E(:,i)*B(:,j)','23',eiK,'23');
%                         
%         end
%     end
%   end

   diKea = zeros(D); diKec = diKea; diKect = diKea; diKeb = diKea;
   Aeik = A'*eiK; Eeik = E'*eiK; 
   Oeik = Oeik/2; eikO = eikO/2;
   for i = 1:D
        Aeiki = Aeik(i,:); Eeiki = Eeik(i,:); 
        for j=1:D
            diKea(i,j) = (A(:,i).*A(:,j))'*eikO + Oeik*(F(i,:).*F(j,:))' + Aeiki*F(j,:)';
            
            diKec(i,j) = (A(:,i).*E(:,j))'*eikO + Oeik*(F(i,:)'.*B(:,j)) + Aeiki*B(:,j);

            diKect(i,j) = (E(:,i).*A(:,j))'*eikO + Oeik*(B(:,i).*F(j,:)') + Eeiki*F(j,:)';

            diKeb(i,j) = (E(:,i).*E(:,j))'*eikO + Oeik*(B(:,i).*B(:,j)) + Eeiki*B(:,j);
                        
        end
    end
    
    dt2 = c*[diKea diKec; diKect diKeb] + dc*iKe;
    dCdst(k,:,:) = dt1 - dt2;
    
 end
end

if isfield(dynmodel,'ratio'); C = C.*dynmodel.ratio; end

C = diag(C);
if nargout > 2
    for i=1:2*D; 
        dCdm(:,:,i) = diag(dCdmt(:,i)); 
        for j=1:2*D; dCds(:,:,i,j) = diag(dCdst(:,i,j)); end; 
    end
end
