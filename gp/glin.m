function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = glin(dynmodel,m,s)
% Function to predict the next state given the joint Gaussian system of an
% uncertain linear model and the current state. The uncertain linear model
% has a fixed mean and variance.
%
% The next state is given by,
%       X = A * [x u] + b  
% and x, A, b  and u are all jointly Gaussian. This function concatenates
% these three variables together as follows:
%       m = [ mx; mb; mA(:); mu]
%           mx is D x 1                     
%           mb is E x 1
%           mA is E x (D+C)
%           mu is C x 1
%      ni = length(m) =  D+E+E*(D+C)+C
%
%       M = [mX; mb; mA(:)]
%     no = length(M) = 2*E + DC*E
%
%
% INPUT
% m         ni x 1   mean of the test distribution
% s         ni x ni covariance matrix of the test distribution
% OUTPUT
% M         no x 1,  mean of pred. distribution
% S         no x no covariance of the pred. distribution
% V         ni x no inv(s) times covariance between input and output
% dMdm      no   x ni, deriv of output mean w.r.t. input mean
% dSdm      no^2  x ni, deriv of output covariance w.r.t input mean
% dVdm      ni*no x ni, deriv of input-output cov w.r.t. input mean
% dMds      no    x ni^2, deriv of ouput mean w.r.t input covariance
% dSds      ni^2  x ni^2, deriv of output cov w.r.t input covariance
% dVds      ni*no x ni^2, deriv of inv(s)*input-output covariance w.r.t input cov
%
% Andrew McHutchon, Jonas Umlauft
% 2014-04-09
%% Dimension Indizes and Permutation Matrizes
E = dynmodel.outputDIM;    % Output dimension
D = dynmodel.inputDIM;     % Input dimension
C = dynmodel.controlDIM;   % Control dimension
DC = D+C;                  % size of augmented state


ni = numel(m);           % size of m
no = 2*E + DC*E;         % size of M

% Indices for inputs
ix = [1:D  D+E+DC*E+1 : ni];
ib = D+1 : D+E;
iA = D+E+1 : D+E+DC*E;

% Indices for outputs
Ix = 1:E;
Ib = E+1:2*E;
IA = 2*E+1 : 2*E + E*DC;


P2=reshape(permute(reshape(eye(E*E),E,E,E,E),[1 4 2 3]),E^2,E^2);


%% Extract x, A and b distributions and covariances
mx = m(ix); mb = m(ib); mA = reshape(m(iA),E,DC);
Vx = s(ix,ix); Vb = s(ib,ib); VA = s(iA,iA);
CAx = s(iA,ix); Cxb = s(ix,ib); CAb = s(iA,ib);
CbA = CAb'; CxA = CAx';Cbx = Cxb';


i = bsxfun(@plus,(1:E)',(0:(DC*E+E):E*DC*DC) );

%% Output mean
mX = mA*mx + sum(CAx(i),2) + mb;  % the mean of the state x_t
M = [mX;mb;mA(:)];                % glue on the linear model means

%% Output variance
% Var(A*x)
VAx = V_Ax(mx,mA,Vx,VA,CxA);

% cov(A*x,b)
CAxb = mA*Cxb + sum(bsxfun(@times,reshape(CbA,E,E,DC),permute(mx,[3,2,1])),3)';

% Var(X) = var(A*x) + var(b) + cov(A*x,b) + cov(b,A*x)
VX =VAx + Vb+ CAxb  + CAxb' ;

% Var(A*x,A)
CAxA = C_Ax_A(mx,mA,VA,CxA);


% Cov(X,A) = Cov(A*x,A) + Cov(b,A)
CXA = CAxA + CbA;
CAX = CXA';

% Cov(X,b) = Cov(A*x,b) + Var(b)
CXb = CAxb + Vb;
CbX = CXb';

% Construct full output covariance matrix
S = [VX  CXb  CXA;
    CbX Vb   CbA;
    CAX CAb  VA];

%% input-output covariance
% CXx =cov(A*x,x)   + Cbx

CXx = mA*Vx + sum(bsxfun(@times,reshape(CxA,DC,E,DC),permute(mx,[3,2,1])),3)' + Cbx;
CxX = CXx';

% Construct full input output covariance matrix
V = [ CxX   Cxb  CxA;
    CbX   Vb   CbA;
    CAX   CAb  VA ];

% Swap input u column to bottom
V = [V(1:D,:);
    V(DC+1:end,:);
    V(D+1:DC,:)];

V = s\V;





%% Derivatives
%% dMdm
if nargout > 3
    dMdm  = zeros(no,ni);
    % dmXd...
    dMdm(Ix,ix)   = mA;                % dmXdmx
    dMdm(Ix,ib)   = eye(E);            % dmXdmb
    dMdm(Ix,iA)   = kron(mx',eye(E));  % dmXdmA
    % dMbdmb
    dMdm(Ib,ib)   = eye(E);            % dMbdmb
    % dMAdmA
    dMdm(IA,iA)   = eye(E*DC);         % dMAdmA
end

%% dSdm
if  nargout > 4
    dSdm = zeros(no,no,ni);  
    % dVXd...
    dSdm(Ix,Ix,ix) = reshape(mA*CAx',E,E,DC) ...                                                                            % dVXdmx
        + permute(reshape(CAx*mA', E, DC, E),[1 3 2])...
        + permute(sum(bsxfun(@times,permute(reshape(VA,E,DC,E,DC),[1 3 2 4]),permute(mx,[3 2 1 4])),3),[2 1 4 3])...         % for V(A*x)
        + permute(sum(bsxfun(@times,permute(reshape(VA,E,DC,E,DC),[1 3 2 4]),permute(mx,[3 2 1 4])),3),[1 2 4 3])...         % for V(A*x)
        + permute(reshape(CAb, E, DC, E), [1 3 2])...                                                                        % for cov(A*x,b)
        + permute(reshape(CAb, E, DC, E), [3 1 2]);                                                                          % for cov(b,A*x)
    dSdm(Ix,Ix,iA) =  reshape(kron(mA*Vx,eye(E)) + P2*kron(mA*Vx,eye(E)),E,E,DC*E) ...                                       % dVXdmA  for V(A*x)
        + reshape((P2+eye(E^2))*kron((sum(bsxfun(@times,reshape(CAx',DC,E,DC),permute(mx,[3 2 1])),3))',eye(E)),E,E,DC*E)... % for V(A*x)
        + reshape(kron(Cbx,eye(E)),E,E,DC*E)...                                                                              % for cov(A*x,b)
        + permute(reshape(kron(Cbx,eye(E)),E,E,DC*E), [2 1 3]);                                                              % for cov(b,A*x)
    % dCbXd...
    dSdm(Ib,Ix,ix) = reshape(permute(reshape(CAb, E, DC, E), [3 1 2]), E,E,DC);                        % dCbXdmx
    dSdm(Ib,Ix,iA) =  reshape(permute(reshape(kron(Cxb',eye(E)),E,E,E,DC),[2 3 1 4]),E,E,DC*E);        % dCbXdmA
    % dCAXd...
    dSdm(IA,Ix,ix) = reshape(permute(reshape(VA, E, DC, E, DC), [3 4 1 2]),DC*E,E,DC);                 % dCAXdmx
    dSdm(IA,Ix,iA) = reshape(permute(reshape(kron(CAx,eye(E)),E,E,E,DC,DC),[2 3 4 1 5]),DC*E,E,DC*E);  % dCAXdmA
    % dCXbd...
    dSdm(Ix,Ib,ix) = reshape(permute(reshape(CAb, E, DC, E), [1 3 2]), E,E,DC);                        % dCXbdmx
    dSdm(Ix,Ib,iA) = reshape(kron(Cxb',eye(E)),E,E,E*DC);                                              % dCXbdmA
    % dCXAd...
    dSdm(Ix,IA,ix) = reshape(permute(reshape(VA, E, DC, E,DC),[3 1 2 4]),E,E*DC,DC);                   % dCXAdmx
    dSdm(Ix,IA,iA) = reshape(kron(CAx,eye(E)),E,E*DC,E*DC);                                            % dCXAdmA
    % Reshape to Output format
    dSdm = reshape(dSdm,no^2,ni);
end

%%  dVdm
if nargout > 5
    dVdm = zeros(ni,no,ni);  
    % dCAXdmx
    dVdm(iA,Ix,ix) = reshape(permute(reshape(eye(DC*E), E, DC, E, DC), [4 3 2 1]),DC*E,E,DC);  % dCAXdmx
    % dCxXdmA
    dVdm(ix,Ix,iA)  = reshape(permute(reshape(eye(DC*E),DC,E,DC,E),[3 4 2 1]),DC,E,E*DC);      % dCxXdmA
    % Reshape to Output format
    dVdm = reshape(dVdm,ni*no,ni);
end

%% dMds
if nargout > 6
    dMds = zeros(no,ni,ni);
    % dmXdCxA
    dMds(Ix,ix,iA) = reshape(eye(E*DC),E,DC, DC*E);     % dmXdCxA
    % Reshape to Output format
    dMds  = reshape(dMds,no,ni^2);
end

%% dSds
if nargout > 7
    dSds = zeros(no,no,ni,ni); 
    % dVXd...
    dSds(Ix,Ix,ix,ix) = bsxfun(@times, permute(mA, [1 4 2 3]), permute(mA, [3 1 4 2])) + permute(reshape(VA,E,DC,E,DC),[1 3 2 4]);  % dVXdVx  
    dSds(Ix,Ix,iA,ix) = reshape(permute(dVX_dCAx(mx, mA, CxA),[1 2 3 4 5]),E,E,DC*E,DC);                                            % dVXdCAx
    dSds(Ix,Ix,ix,ib) = reshape(kron(eye(E),mA),E,E,DC,E) + permute(reshape(kron(eye(E),mA'),DC,E,E,E),[2 3 1 4]);                  % dVXdCxb 
    dSds(Ix,Ix,ib,ib) = reshape(eye(E^2),E,E,E,E);                                                                                  % dVXdVb  
    dSds(Ix,Ix,iA,ib) = dVX_dCAb(mx,E);                                                                                             % dVXdCAb 
    dSds(Ix,Ix,iA,iA) = dVX_dVA(mx,Vx,E);                                                                                           % dVXdVA  
    % dCbXd...
    dSds(Ib,Ix,ib,ix) = permute(reshape(kron(eye(E),mA),E,E,DC,E),[2 1 4 3]);           % dCbXdCbx
    dSds(Ib,Ix,ib,ib) = reshape(P2,E,E,E,E);                                            % dCbXdVb 
    dSds(Ib,Ix,iA,ib) = reshape(P2*kron(eye(E),kron(mx',eye(E))),E,E,DC*E,E);           % dCbXdCAb
    % dCAXd...
    dSds(IA,Ix,iA,ix) = permute(reshape(kron(eye(DC*E),mA),E,DC*E,DC,DC*E),[2 1 4 3]);  % dCAXdCAx
    dSds(IA,Ix,iA,ib) = reshape(eye(DC*E*E),DC*E,E,DC*E,E);                             % dCAXdCAb
    dSds(IA,Ix,iA,iA) = dCAX_dVA(mx,E);                                                 % dCAXdVA 
    % dCXbd...
    dSds(Ix,Ib,ix,ib) = reshape(kron(eye(E),mA),E,E,DC,E);                              % dCXbdCxb
    dSds(Ix,Ib,ib,ib) = reshape(eye(E^2),E,E,E,E);                                      % dCXbdVb 
    dSds(Ix,Ib,iA,ib) = reshape(kron(eye(E),kron(mx',eye(E))),E,E,DC*E,E);              % dCXbdCAb
    % dVbd...
    dSds(Ib,Ib,ib,ib) = reshape(eye(E^2),E,E,E,E);                                      % dVbdVb   
    % dCAbd...
    dSds(IA,Ib,iA,ib) = reshape(eye(E*E*DC),E*DC,E,E*DC,E);                             % dCAbdCAb
    % dCXAd...
    dSds(Ix,IA,ib,iA) = reshape(eye(DC*E*E),E,DC*E,E,DC*E);                             % dCXAdCbA  
    dSds(Ix,IA,ix,iA) = reshape(kron(eye(E*DC),mA),E,DC*E,DC,DC*E);                     % dCXAdCxA  
    dSds(Ix,IA,iA,iA) = dCXA_dVA(mx,E);                                                 % dCXAdVA   
    % dCbAd...
    dSds(Ib,IA,ib,iA) = reshape(eye(E*E*DC),E,DC*E,E,DC*E);                             % dCbAdCbA  
    % dVAd...
    dSds(IA,IA,iA,iA) = reshape(eye((DC*E)^2),DC*E,DC*E,DC*E,DC*E);                     % dVAdVA   
    % Reshape to Output format
    dSds = reshape(dSds,no^2,ni^2);
end


%% dVds
if nargout > 8
    dVds = zeros(ni*no,ni^2);
end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   Sub-functions    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function VAx = V_Ax(mx,mA,Vx,VA,CxA)
% Variance of A*x when they are jointly Gaussian distributed
DC = size(mA,2);
E = size(mA,1);

VA = reshape(VA,E,DC,E,DC);
VA = permute(VA,[1 3 2 4]); 
CxA = reshape(CxA,DC,E,DC); 
CAx = permute(CxA,[2 3 1]); 

VAx = zeros(E);
for l = 1:DC
    for k = 1:DC
        VAx = VAx + VA(:,:,l,k)*Vx(l,k) + CAx(:,l,k)*CxA(l,:,k)...
            + mA(:,l)*mA(:,k)'*Vx(l,k) + mx(l)*mx(k)*VA(:,:,l,k)...
            + mx(l)*mA(:,k)*CxA(k,:,l) + mx(k)*CAx(:,k,l)*mA(:,l)';
    end
end

end


function CAxA = C_Ax_A(mx,mA,VA,CxA)

DC = size(mA,2);
E = size(mA,1);
CxA = reshape(CxA,DC,E,DC);
CAxA = zeros(E,E,DC);
VA = reshape(VA,E,DC,E,DC);
for j = 1:DC
    SAmx =0;
    for i = 1:DC
        SAmx  = SAmx  + permute(VA(:,i,:,j),[1 3 2 4])*mx(i);
    end
    CAxA(:,:,j) = SAmx  + mA*CxA(:,:,j);
end
CAxA = reshape(CAxA,E,DC*E);

end

function dVXdCAx = dVX_dCAx(mx, mA, CxA)
E = size(mA,1);
DC = size(mA,2);
CxA = reshape(CxA,DC,E,DC);
mxA = reshape(mA(:)*mx',1,E,DC,DC);
summe = permute(CxA,[4 2 3 1]) + mxA;
dVXdCAx = zeros(E,E,E,DC,DC);
for i=1:E
    for j=1:DC
        for k=1:DC
            dVXdCAx(i,:,i,j,k) = summe(1,:,k,j) ;
        end
    end
end
dVXdCAx = dVXdCAx +permute(dVXdCAx,[2 1 3 4 5]);

end

function dVXdCAb = dVX_dCAb(mx,E)
DC = numel(mx);
dVXdCAb=zeros(E,E,E,DC,E);

for l=1:E
    for m=1:DC
        for n=1:E
            dVXdCAb(l,n,l,m,n) = mx(m);
            dVXdCAb(n,l,l,m,n) = dVXdCAb(n,l,l,m,n) +  mx(m);
        end
    end
end
dVXdCAb = reshape(dVXdCAb,E,E,E*DC,E);
end

function dVXdVA = dVX_dVA(mx,Vx,E)
DC = numel(mx);
dVXdVA=zeros(E,E,E,DC,E,DC);

for k=1:E
    for l=1:DC
        for m=1:E
            for n=1:DC
                dVXdVA(k,m,k,l,m,n) = Vx(l,n) + mx(l)*mx(n);
            end
        end
    end
end
dVXdVA = reshape(dVXdVA,E,E,E*DC,E*DC);
end

function dCXAdVA= dCXA_dVA(mx,E)
DC = numel(mx);
dCXAdVA = zeros(E,E,DC,E,DC,E,DC);
for k= 1:E
    for l=1:DC
        for m = 1:E
            for n=1:DC
                dCXAdVA(k,m,n,k,l,m,n) = mx(l);
            end
        end
    end
end
dCXAdVA = reshape(dCXAdVA,E,DC*E,DC*E,DC*E);

end

function dCAXdVA = dCAX_dVA(mx,E)
DC = numel(mx);
dCAXdVA = zeros(E,DC,E,E,DC,E,DC);
for k= 1:E
    for l=1:DC
        for m = 1:E
            for n=1:DC
                dCAXdVA(k,l,m,k,l,m,n) = mx(n);
            end
        end
    end
end
dCAXdVA = reshape(dCAXdVA,DC*E,E,DC*E,DC*E);
end

%% Loop free version for square A

% function VAx = V_Ax(m,s)
% % Variance of A*x when they are jointly Gaussian distributed
%
% D = (sqrt(1+4*length(m))-1)/2;
%
% mx = m(1:D); mA = m(D+1:end);
% VA = s(D+1:end,D+1:end); Vx = s(1:D,1:D); CxA = s(1:D,D+1:end);
%
% % ai = reshape(1:D^2,D,D);
% % bi = repmat(1:D,D,1);
% emx = reshape(repmat(mx,1,D)',D^2,1);
% eVx = kron(Vx,ones(D));
% eCxA = kron(CxA,ones(D,1));
%
% CAbAb = (VA + mA*mA').*(eVx + emx*emx') + (eCxA + emx*mA').*(eCxA' + mA*emx') ...
%     - 2*(mA.*emx)*(mA.*emx)';
%
% i = bsxfun(@plus,1:D,(0:D:(D-1)*D)');
%
% CAbAb = CAbAb(i,i);
% VAx = cellfun(@(x)(sum(sum(x))), mat2cell(CAbAb,D*ones(1,D),D*ones(1,D)));
% end


%% Comutation for s*V  -> input output covariance without mulitplying by inv(s)
%
% %% dVdm
% dVdm = zeros(E,E,E);    %dim1 --> dC...YdmY    dim2 --> dCY...dmY   dim3 --> dCYYdm...
% %% dCxXd...
% dVdm(1:D,     1:D,      1:D    )  = permute(reshape(CAx, D, D, D), [3 1 2]);                      % dCxXdmx
% dVdm(1:D,     1:D,      2*D+1:end)  = reshape(P2*kron(Vx,eye(D)),D,D,D^2);                          % dCxXdmA
% %% dCbXd...
% % dVdm(D+1:2*D, 1:D,      1:D    ) = permute(reshape(CAb, D, D, D), [3 1 2]);                       % dCbXdmx
% % dVdm(D+1:2*D, 1:D,      2*D+1:end) = reshape(P2*kron(Cxb',eye(D)),D,D,D^2);                         % dCbXdmA
% %% dCAXd...
% dVdm(2*D+1:end, 1:D,      1:D    ) = reshape(permute(reshape(VA, D, D, D, D), [3 4 1 2]),D^2,D,D);  % dCAXdmx
% dVdm(2*D+1:end, 1:D,      2*D+1:end) = reshape(P3'*kron(CAx,eye(D)),D^2,D,D^2);                       % dCAXdmA
%
% dVdm = reshape(dVdm,E^2,E);
%
%
%
% %% dVds
% dVds = zeros(E,E,E,E);   %dim1 --> dC...YdCYY    dim2 --> dCY...dCYY   dim3 --> dCYYdC...Y   dim4 --> dCYYdCY...
% %% dCxXd...
% dVds(1:D,     1:D,     1:D,     1:D    ) = reshape(P2*kron(eye(D),mA),D,D,D,D);                       % dCxXdVx
% dVds(1:D,     1:D,     2*D+1:end, 1:D    ) = reshape(P2*kron(eye(D),kron(mx',eye(D))),D,D,D^2,D);       % dCxXdCAx
% dVds(1:D,     1:D,     1:D,     D+1:2*D) = reshape(eye(D^2),D,D,D,D);                                 % dCxXdCxb
% %% dCbXd...
% % dVds(D+1:2*D, 1:D,     D+1:2*D, 1:D    ) = permute(reshape(kron(eye(D),mA),D,D,D,D),[2 1 4 3]);       % dCbXdCbx
% % dVds(D+1:2*D, 1:D,     D+1:2*D, D+1:2*D) = reshape(P2,D,D,D,D);                                       % dCbXdVb
% % dVds(D+1:2*D, 1:D,     2*D+1:end, D+1:2*D) = reshape(P2*kron(eye(D),kron(mx',eye(D))),D,D,D^2,D);       % dCbXdCAb
% %% dCAXd...
% dVds(2*D+1:end, 1:D,     2*D+1:end, 1:D    ) = permute(reshape(kron(eye(D^2),mA),D,D^2,D,D^2),[2 1 4 3]);  % dCXAdCAx
% dVds(2*D+1:end, 1:D,     2*D+1:end, D+1:2*D) = reshape(eye(D^3),D^2,D,D^2,D);                              % dCXAdCAb
% dVds(2*D+1:end, 1:D,     2*D+1:end, 2*D+1:end) = reshape(P3'*kron(eye(D^2),kron(mx',eye(D))),D^2,D,D^2,D^2); % dCXAdVA
%
% dVds(1:D,     D+1:2*D, 1:D,     D+1:2*D) = reshape(eye(D^2),D,D,D,D);                                  % dCxbdCxb
% dVds(D+1:2*D, D+1:2*D, D+1:2*D, D+1:2*D) = reshape(eye(D^2),D,D,D,D);                                  % dVbdVb
% dVds(2*D+1:end, D+1:2*D, 2*D+1:end, D+1:2*D) = reshape(eye(D^3),D^2,D,D^2,D);                              % dCAbdCAb
% dVds(1:D,     2*D+1:end, 1:D,     2*D+1:end) = reshape(eye(D^3),D,D^2,D,D^2);                              % dCxAdCxA
% dVds(D+1:2*D, 2*D+1:end, D+1:2*D, 2*D+1:end) = reshape(eye(D^3),D,D^2,D,D^2);                              % dCbAdCbA
% dVds(2*D+1:end, 2*D+1:end, 2*D+1:end, 2*D+1:end) = reshape(eye(D^4),D^2,D^2,D^2,D^2);                          % dVAdVA