function [nlp, nlpdp] = GPf(p,dyn,plant,data)
% Implement forward filtering using a GP transition model and
% an additive Gaussian noise observation model.

useprior = 0;
if length(data) > 1
    error('data is a struct array - call multiTrial or use indexing\n'); end
y = data.state; u = data.action;

% Initialisations ----------------------------------------------------
dyn = updateHyp(dyn,p); [dyn, ddyn] = preComp(dyn); dyn.iKdl = ddyn.iKdll;

y = y(:,plant.dyno); [T, E] = size(y); Np = length(unwrap(p));
Mf = zeros(E,T); Sf = zeros(E,E,T); nlp = zeros(1,T); nlpdp = zeros(Np,T);

% ---------------------------------------------------------------------

% for t = 1 -----------------------------------------------------------
if isfield(dyn,'priorm'); m = dyn.priorm; S = dyn.priorS; 
else m = zeros(E,1); S = eye(E); end                % Prior on first state            
if nargout < 2; [Mf(:,1), Sf(:,:,1), nlp(1)] = GPfstep(p,[],plant,m,S,y(1,:)');
else            [Mf(:,1), Sf(:,:,1), nlp(1), nlpdp(:,1), Mdp, Sdp] ...
                              = GPfstep(p,[],plant,m,S,y(1,:)',[]);
end
% ---------------------------------------------------------------------

% The forward filtering sweep for t = 2:T -----------------------------
for i=2:T
    if nargout < 2
        [Mf(:,i), Sf(:,:,i), nlp(i)] ...
              = GPfstep(p,dyn,plant,Mf(:,i-1),Sf(:,:,i-1),y(i,:)', ...
                        [y(i-1,end) u(i-1,:)]');
    else
        [Mf(:,i), Sf(:,:,i), nlp(i), nlpdp(:,i), Mdp, Sdp] ...
               = GPfstep(p,dyn,plant,Mf(:,i-1),Sf(:,:,i-1),y(i,:)', ...
                         [y(i-1,end) u(i-1,:)]',Mdp,Sdp);
    end
    if any(diag(Sf(:,:,i))<-1e-9); keyboard; end
end % -------------------------------------------------------------------

% Sum up nlp and add together derivatives
nlp = sum(nlp); if nargout == 2; nlpdp = sum(nlpdp,2); end

% a penalty prior on the pseudo targets to keep them reasonable ----------
if useprior; 
    if nargout < 2; pp = penPrior3(p,dyn); 
    else     [pp, dpp] = penPrior3(p,dyn,ddyn); nlpdp = nlpdp + unwrap(dpp); 
    end
    nlp = nlp + pp;
end % --------------------------------------------------------------------

% Rewrap to struct
if nargout == 2; 
    nlpdp = rewrap(p,nlpdp); 
    if ~dyn.trainMean
        [nlpdp.m] = deal(zeros(size(nlpdp(1).m)));
        [nlpdp.b] = deal(zeros(size(nlpdp(1).b)));
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%                      Sub Functions                   %%%%%%%%%%%   

function [pp, dpp] = penPrior(p,dyn,ddyn)

[N, E] = size(dyn.beta); bn = zeros(N,E);

pp = 0;
for i=1:E;
    Rnn = chol(dyn.Knn(:,:,i));
    bn(:,i) = dyn.Knn(:,:,i)\(dyn.K(:,:,i)*p(i).beta);
    pp = pp + E/2*log(2*pi) + trace(log(Rnn)) ...
                                      + 0.5*bn(:,i)'*dyn.Knn(:,:,i)*bn(:,i);
end

if nargout == 2
    pn2 = exp(2*[p.n]);
    dpp = rewrap(p,zeros(size(unwrap(p),1)));
    triKKdll = etprod('12',dyn.iKnn,'342',ddyn.Kdll,'3412'); % D-by-E
    Kdlbn = etprod('123',ddyn.Kdll,'1423',bn,'43');          % N-by-D-by-E
    bKdllb = etprod('12',bn,'32',Kdlbn,'312');               % D-by-E
    Kdlb2 = etprod('123',ddyn.Kdll,'1423',[p.beta] - bn,'43'); % N-by-D-by-E
    iKKdlb = etprod('123',dyn.iKnn,'143',Kdlb2,'423');       % N-by-D-by-E
        
    for i=1:E
        Knn = dyn.Knn(:,:,i); Kc = dyn.Kclean(:,:,i); iKnn = dyn.iKnn(:,:,i);
        
        bndn = Knn\(2*pn2(i)*p(i).beta);
        bnds = Knn\(2*Kc*(p(i).beta - bn(:,i)));
        bndl = iKKdlb(:,:,i);
            
        dpp(i).n = bn(:,i)'*Knn*bndn;
        dpp(i).l = bndl'*Knn*bn(:,i) + 0.5*triKKdll(:,i) + 0.5*bKdllb(:,i);
        dpp(i).s = iKnn(:)'*Kc(:) + bn(:,i)'*Kc*bn(:,i) + bn(:,i)'*Knn*bnds;
%         dpp(i).m = bn(:,i)'*Knn*squeeze(dbn.m(:,i,:));
%         dpp(i).b = bn(:,i)'*Knn*dbn.b(:,i);
        dpp(i).beta = bn(:,i)'*dyn.K(:,:,i);
    end
end

function [pp, dpp] = penPrior2(p,dyn,ddyn)
[dyn.hyp.l] = deal(p.l); [dyn.hyp.s] = deal(4); [dyn,ddyn] = preComp(dyn);
[N, E] = size(dyn.beta); f = zeros(N,E); b = [p.beta];

pp = 0;
for i=1:E;
    R = chol(dyn.K(:,:,i));
    f(:,i) = dyn.K(:,:,i)*b(:,i);
    pp = pp + E/2*log(2*pi) + trace(log(R)) + 0.5*b(:,i)'*f(:,i);
end

if nargout == 2
    dpp = rewrap(p,zeros(length(unwrap(p))));
    triKKdll = etprod('12',dyn.iK,'342',ddyn.Kdll,'3412'); % D-by-E
    fdl = etprod('123',ddyn.Kdll,'1423',b,'43');          % N-by-D-by-E
        
    for i=1:E
        Kc = dyn.Kclean(:,:,i); iK = dyn.iK(:,:,i);
            
        dpp(i).l = 0.5*fdl(:,:,i)'*b(:,i) + 0.5*triKKdll(:,i);
%         dpp(i).s = iK(:)'*Kc(:) + b(:,i)'*Kc*b(:,i);
%         dpp(i).m = bn(:,i)'*Knn*squeeze(dbn.m(:,i,:)); wrong
%         dpp(i).b = bn(:,i)'*Knn*dbn.b(:,i); wrong
        dpp(i).beta = f(:,i);
    end
end

function [pp, dpp] = penPrior3(p,dyn,ddyn)
persistent K
% [dyn.hyp.l] = deal(p.l); [dyn.hyp.s] = deal(p.s); [dyn,ddyn] = preComp(dyn);
if isempty(K); K = dyn.K; end
[N, E] = size(dyn.beta); f = zeros(N,E); b = [p.beta];

pp = 0;
for i=1:E;
    f(:,i) = dyn.K(:,:,i)*b(:,i);
    pp = pp + 0.5*f(:,i)'*(K(:,:,i)\f(:,i));
end

if nargout == 2
    dpp = rewrap(p,zeros(length(unwrap(p))));
    fdl = etprod('123',ddyn.Kdll,'1423',b,'43');          % N-by-D-by-E
    fds = etprod('12',2*dyn.Kclean,'132',b,'32');        % N-by-E
        
    for i=1:E
        dpp(i).beta = f(:,i);
        dpp(i).l = f(:,i)'*(K(:,:,i)\fdl(:,:,i));
        dpp(i).s = f(:,i)'*(K(:,:,i)\fds(:,i));
    end
end

function [pp, dpp] = reg(p,dyn,ddyn)
% [dyn.hyp.l] = deal(p.l); [dyn.hyp.s] = deal(4); [dyn,ddyn] = preComp(dyn);
[N, E] = size(dyn.beta); f = zeros(N,E); b = [p.beta];


for i=1:E; f(:,i) = dyn.K(:,:,i)*b(:,i); end
fD = bsxfun(@minus,permute(f,[1,3,2]),permute(f,[3,1,2])); % N-by-N-by-E
pp = dyn.K(:)'*fD(:).^2;

if nargout == 2
    dpp = rewrap(p,zeros(length(unwrap(p))));
    fdl = etprod('123',ddyn.Kdll,'1423',b,'43');          % N-by-D-by-E
        
    for i=1:E
        Kc = dyn.Kclean(:,:,i);
            
        dpp(i).l = 0.5*fdl(:,:,i)'*b(:,i) + 0.5*triKKdll(:,i);
%         dpp(i).s = iK(:)'*Kc(:) + b(:,i)'*Kc*b(:,i);
%         dpp(i).m = bn(:,i)'*Knn*squeeze(dbn.m(:,i,:)); wrong
%         dpp(i).b = bn(:,i)'*Knn*dbn.b(:,i); wrong
        dpp(i).beta = f(:,i);
    end
end  

function [pp, dpp] = reg2(p,dyn,ddyn)
% [dyn.hyp.l] = deal(p.l); [dyn.hyp.s] = deal(4); [dyn,ddyn] = preComp(dyn);
[N, E] = size(dyn.beta); f = zeros(N,E); b = [p.beta];


for i=1:E;  f(:,i) = dyn.K(:,:,i)*b(:,i); end
fD = bsxfun(@minus,permute(f,[1,3,2]),permute(f,[3,1,2])); % N-by-N-by-E
pp = dyn.K(:)'*fD(:).^2;

if nargout == 2
    dpp = rewrap(p,zeros(length(unwrap(p))));
    fdl = etprod('123',ddyn.Kdll,'1423',b,'43');          % N-by-D-by-E
        
    for i=1:E
        Kc = dyn.Kclean(:,:,i);
            
        dpp(i).l = 0.5*fdl(:,:,i)'*b(:,i) + 0.5*triKKdll(:,i);
%         dpp(i).s = iK(:)'*Kc(:) + b(:,i)'*Kc*b(:,i);
%         dpp(i).m = bn(:,i)'*Knn*squeeze(dbn.m(:,i,:)); wrong
%         dpp(i).b = bn(:,i)'*Knn*dbn.b(:,i); wrong
        dpp(i).beta = f(:,i);
    end
end  
