function [M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] = ...
                                                               consq(par, m, s)
% linear controller using pairwise product features
%
% pi(x) = w * y + p, where y is a d + 0.5*d(d+1) vector containing all the
% elements of x and all unique pairwise products of elements of x
% (including squared values of elements of x).
%
% par     policy structure
%   .p    parameters which are modified by training
%     .w  linear weights, D by d
%     .w2 linear weights for product features, D by 0.5*d*(d+1)
%         (this is a vector because we only need weights for the upper
%         triangle of the product matrix, and it's otherwise awkward to
%         define).
%     .b  biases, D by 1
%
% m       mean of state distribution, d by 1
% s       covariance matrix of state distribution, d by d
% 
% M       mean of action, D by 1
% S       covariance of action, D by D
% C       inv(s) times input output covariance D by d
% d<>d<>  derivatives of output variable w.r.t input variable
%
% for more detailed documentation, see doc/consq.pdf
%
% Copyright (C) 2010 by Carl Edward Rasmussen and Philipp Hennig, 2012-07-05

%addpath './util/tprod';
[D d] = size(par.p.w);
w2 = par.p.w2;
mm   = m * m';

% expected products (2dim)
mu   = s + mm;

% covariance between x and y (3dim)
Xi   = etprod('ijl',s,'jl',m,'i') + etprod('ijl',s,'il',m,'j');

% covariance between y and y (4dim):
Sigma = etprod('ijkl',s,'ik',s,'jl') + ...
        etprod('ijkl',s,'il',s,'jk') + ...
        etprod('ijkl',s,'ik',mm,'jl') + ...
        etprod('ijkl',s,'il',mm,'jk') + ...
        etprod('ijkl',s,'jk',mm,'il') + ...
        etprod('ijkl',s,'jl',mm,'ik');

% turn into matrix:
selectors = logical(triu(ones(d)));       % matrix defining an upper triangular
muV       = mu(selectors); muV = muV(:);                % turn mu into a vector
XiM       = permute(Xi,[3 1 2]);    % Matlab only does this on trailing indices
XiM       = XiM(:,selectors)';
SigmaM    = Sigma(:,:,selectors);                                    % now 3dim
SigmaM    = permute(SigmaM,[3 1 2]);
SigmaM    = SigmaM(:,selectors)';

% turn w2 into a 3D tensor: w2_[u(ij=k)] = w2_[uij]
w2mat = zeros(D,d,d);
w2mat(:,selectors) = w2(:,:);               % this matrix is not symmetric!

ssm = zeros(d,d,d);
for k = 1:d % currently no idea how to avoid this for-loop.
    ssm(k,k,:) = squeeze(ssm(k,k,:)) + m(:);
    ssm(k,:,k) = squeeze(ssm(k,:,k)) + m(:)';
end

% output variables
M = par.p.w * m + w2 * muV + par.p.b;
S = par.p.w * s * par.p.w' + ...
    w2 * XiM * par.p.w' + ...
    par.p.w * XiM' * w2' + ...
    w2 * SigmaM * w2';

C = par.p.w' + etprod('ku',ssm,'kij',w2mat,'uij');      % input-output covariance

if nargout > 3                                                    % derivatives        
    dMdm = par.p.w + etprod('uk',w2mat,'ukj',m,'j') + etprod('uk',w2mat,'uik',m,'i');
    
    swt = s * par.p.w';
    ms  = etprod('lij',m,'l',s,'ij');

    osw = etprod('utr',w2mat,'urj',swt,'jt') + etprod('utr',w2mat,'ujr',swt,'jt');
    smsm = permute(ms,[2 3 1]) + permute(ms,[2 1 3]);
    
    term1 = etprod('utr',etprod('urkl',w2mat,'urj',smsm,'jkl'),'urkl',w2mat,'tkl');
    term2 = etprod('utr',etprod('urkl',w2mat,'ujr',smsm,'jkl'),'urkl',w2mat,'tkl');
    
    dSdm = osw + permute(osw,[2 1 3]) + ...
           term1 + term2 + permute(term1,[2 1 3]) + permute(term2,[2 1 3]);
   
    ww = etprod('uvrt',par.p.w,'ur',par.p.w,'vt');
    oo = w2mat + permute(w2mat,[1 3 2]);
    
    dCdm = permute(oo,[2 1 3]); %etprod('kur',sinv,'kl',etprod('lur',s,'jl',oo,'urj'),'lur');
    
    dMds = w2mat; % notice that this means we only depend on half of s.
    
    oow = etprod('uvrtj',oo,'urj',par.p.w,'vt');
    
    dSds = ww + etprod('uvrt',oow + permute(oow,[2 1 3 4 5]),'uvrtj',m,'j') + ...
        etprod('uvrt',etprod('uvrtjl',oo,'urj',oo,'vtl'),'uvrtjl',mu,'jl');
    
    dCds = zeros(d,D,d,d);
    
    dMdb = eye(D);
    
    dSdb = zeros(D,D,D);
    
    dCdb = zeros(d,D,D,1);
    
    dMdw = zeros(d,D,D);
    dMdw(:,logical(eye(D))) = repmat(m,1,D);
    dMdw = permute(dMdw,[2 3 1]);
    
    oXi  = w2 * XiM;
    dSdw = zeros(D,D,D,d);
    for i = 1:D
        dSdw(i,:,i,:) = swt' + oXi;
    end
    dSdw = dSdw + permute(dSdw,[2 1 3 4]);
    
    dCdw = zeros(d,D,D,d); % after much trying, I have no better idea than this for-loop
    for i = 1:d
        dCdw(i,:,:,i) = eye(D);
    end
    
    dMdw2 = zeros(D,D,0.5*d*(d+1));
    for i =1:D
        dMdw2(i,i,:) = muV;
    end
    
    wXit = par.p.w * XiM';
    oSig = w2 * SigmaM;
    dSdw2 = zeros(D,D,D,0.5*d*(d+1));
    for j = 1:D
        dSdw2(j,:,j,:,:) = wXit + oSig;
    end
    dSdw2 = dSdw2 + permute(dSdw2,[2 1 3 4 5]);
    
    dCdw2 = zeros(d,D,D,0.5*d*(d+1));
    for a = 1:d
        for u = 1:D
            dCdw2(a,u,u,:) = ssm(a,selectors);
        end
    end
    
    dMds = reshape(dMds,[D d*d]);                       % vectorise derivatives
    dSds = reshape(dSds,[D*D d*d]); dSdm = reshape(dSdm,[D*D d]);
    dCds = reshape(dCds,[d*D d*d]); dCdm = reshape(dCdm,[d*D d]);

    dMdp = [reshape(dMdb,D,[])   reshape(dMdw,D,[])   reshape(dMdw2,D,[])];
    dSdp = [reshape(dSdb,D*D,[]) reshape(dSdw,D*D,[]) reshape(dSdw2,D*D,[])];
    dCdp = [reshape(dCdb,d*D,[]) reshape(dCdw,d*D,[]) reshape(dCdw2,d*D,[])];
end
