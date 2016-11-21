function [M V C dMdmu dVdmu dCdmu dMdS dVdS dCdS dMdp dVdp dCdp] = contrig(par, mu, S)

% Function to compute predictive mean, variance, input-output 
% covariance, and all relevant derivatives for a trigonometric model 
% with an input distribution. The predictive equation is:
%
%   f_* = theta'*phi
%       = [alpha,...,beta][cos(wx_*);..;sin(wx_*)]
%
% Inputs:
%   par     : policy struct
%    .p.w   : angular frequencies, can be shared between controls, m-by-D(-by-E)
%    .p.a   : weights on the cosines, m-by-E
%    .p.b   : weights on the sines, m-by-E
%
%   mu      : the mean of the test distribution D-by-1
%   S       : the variance of the test distribution D-by-D
%
% Outputs:
%   M     : mean of output state,                               E-by-1
%   V     : variance of output state,                           E-by-E
%   C     : inv(S)*input/output covariance,                     D-by-E
%   dMdmu : derivative of output wrt input mean,                E-by-D
%   dVdmu : derivative of output var wrt input mean,            E-by-E-by-D
%   dCdmu : derivative of IO covariance wrt input mean,         D-by-E-by-D
%   dMdS  : derivative of output wrt input mean,                E-by-D-by-D
%   dVdS  : derivative of output var wrt input mean,            E-by-E-by-D-by-D
%   dCdS  : derivative of IO covariance wrt input mean,         D-by-E-by-D-by-D
%   dMdp  : derivative of output wrt policy parameters,         E-by-Np
%   dVdp  : derivative of output var wrt policy parameters,     E-by-E-by-Np
%   dCdp  : derivative of IO covariance wrt policy parameters,  D-by-E-by-Np
%
% Andrew McHutchon, 5 July 2012

% Dimensions
D = length(mu);              % number of training samples, dimension
wE = size(par.p.w,3);        % are we sharing frequencies between controls?
[m E] = size(par.p.a);              % number of basis

% Initialisations
M = zeros(E,1); V = zeros(E,E); C = zeros(D,E);
dMdmu = zeros(E,D); dVdmu = zeros(E,E,D); dCdmu = zeros(D,E,D);
dMdS = zeros(E,D,D); dVdS = zeros(E,E,D,D); dCdS = zeros(D,E,D,D);
dMdw = zeros(E,m,D,E); dVdw = zeros(E,E,m,D,E); dCdw = zeros(D,E,m,D,E);
dMda = zeros(E,m,E); dVda = zeros(E,E,m,E); dCda = zeros(D,E,m,E);
dMdb = zeros(E,m,E); dVdb = zeros(E,E,m,E); dCdb = zeros(D,E,m,E);

for i=1:E 
     
    %%%%%% Mean %%%%%%%%%%
    
    a = par.p.a(:,i); b = par.p.b(:,i);             % weights
    theta = [a; b];                                 % 2m-by-1
    w = par.p.w(:,:,min(i,wE));                     % angular freqs m-by-D
    wmu = w*mu;                                     % m-by-1 vector of w_i'*mu
    e = exp(-dot(w*S,w,2)/2);                   % m-by-1 vector exp(-w'*S*w/2)
    gamma = [e.*cos(wmu);e.*sin(wmu)];          % 2m-by-1
    
    M(i) = theta'*gamma;
    
    %% Derivatives of the Mean %%%%%%%%%
% 1) wrt input mean, mu
    dgamdm_ = [-gamma(m+1:end);gamma(1:m)];   % [-e.*sin(wmu);e.*cos(wmu)];
    dgamdm = bsxfun(@times,dgamdm_,[w;w]);    % derivative of gamma wrt mu, 2m-by-D
    dMdmu(i,:) = dgamdm'*theta;   % D-by-1

% 2) wrt input variance, S
    dwSwdS = -0.5*etprod('123',w,'12',w,'13');          % m-by-D-by-D
    dgamdS = bsxfun(@times,[dwSwdS;dwSwdS],gamma);      % 2m-by-D-by-D
    dMdS(i,:,:) = reshape(theta'*dgamdS(:,:),D,D);      % D-by-D
    
% 3) wrt w
    dgamdw = dgamdm_*mu' - bsxfun(@times,[w;w]*S,gamma); % Collapsed into 2m-by-D (should be 2m-by-m-by-D)
    dgamdwthe = bsxfun(@times,theta,dgamdw);                        % 2m-by-D
    dMdw(i,:,:,i) = dgamdwthe(1:m,:)+dgamdwthe(m+1:end,:);          % m-by-D
    
% 4) wrt a & b
    dMda(i,:,i) = gamma(1:m); dMdb(i,:,i) = gamma(m+1:end);         % m-by-1
    
    %%%%%%%%% Variance %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % The variance equation is:
    %  V_ij = 0.5*alpha_i^T(ep*cosp+em*cosm)*alpha_j +
    %         0.5*alpha_i^T(ep*sinp-em*sinm)*beta_j +
    %         0.5*beta_i^T(ep*sinp-em*sinm)*alpha_j +
    %         0.5*beta_i^T(em*cosm-ep*cosp)*beta_j - M(i)*M(j)
    
    for j=1:i
        aj = par.p.a(:,j); bj = par.p.b(:,j);                           % m-by-1
        wj(1,:,:) = par.p.w(:,:,min(j,wE));                        % 1-by-m-by-D
        wipj = reshape(bsxfun(@plus,permute(w,[1,3,2]),wj),m^2,D);    % m^2-by-D
        wimj = reshape(bsxfun(@minus,permute(w,[1,3,2]),wj),m^2,D);   % m^2-by-D
        wpSwp = dot(wipj*S,wipj,2); wmSwm = dot(wimj*S,wimj,2);       % m^2-by-1
        wpmu = wipj*mu;             wmmu = wimj*mu;                   % m^2-by-1
        ep = exp(-0.5*wpSwp);       em = exp(-0.5*wmSwm);             % m^2-by-1
        cosp = cos(wpmu);           cosm = cos(wmmu);                 % m^2-by-1
        sinp = sin(wpmu);           sinm = sin(wmmu);                 % m^2-by-1
        v1 = reshape(ep.*cosp+em.*cosm,m,m); v2 = reshape(ep.*sinp-em.*sinm,m,m);
        v3 = reshape(ep.*sinp+em.*sinm,m,m); v4 = reshape(em.*cosm-ep.*cosp,m,m);   
    
        % Bottom left triangle of variance matrix
        V(i,j) = 0.5*(a'*v1*aj + a'*v2*bj + b'*v3*aj + b'*v4*bj) - M(i)*M(j);
        
        %% Derivatives of the Variance %%
 % 1) wrt input mean, mu
        dv1 = bsxfun(@times,wipj,-ep.*sinp)+bsxfun(@times,wimj,-em.*sinm);  % m^2-by-D
        dv2 = bsxfun(@times,wipj,ep.*cosp)-bsxfun(@times,wimj,em.*cosm);    % m^2-by-D
        dv3 = bsxfun(@times,wipj,ep.*cosp)+bsxfun(@times,wimj,em.*cosm);    % m^2-by-D
        dv4 = bsxfun(@times,wimj,-em.*sinm)-bsxfun(@times,wipj,-ep.*sinp);  % m^2-by-D
        aaj = reshape(a*aj',[],1); abj = reshape(a*bj',[],1);               % m^2-by-1
        baj = reshape(b*aj',[],1); bbj = reshape(b*bj',[],1);               % m^2-by-1
        dv1 = aaj'*dv1; dv2 = abj'*dv2; dv3 = baj'*dv3; dv4 = bbj'*dv4;     % 1-by-D
        
        dVdmu(i,j,:) = 0.5*(dv1+dv2+dv3+dv4) - M(i)*dMdmu(j,:) - dMdmu(i,:)*M(j); % 1-by-D
        
 % 2) wrt input variance, S
        depdS = -0.5*bsxfun(@times,reshape(bsxfun(@times,wipj,permute(wipj,[1,3,2])),m^2,D^2),ep); % m^2-by-D^2
        demdS = -0.5*bsxfun(@times,reshape(bsxfun(@times,wimj,permute(wimj,[1,3,2])),m^2,D^2),em); % m^2-by-D^2
        % Form the terms in the brackets in V equation
        dv1 = bsxfun(@times,depdS,cosp) + bsxfun(@times,demdS,cosm);        % m^2-by-D^2
        dv2 = bsxfun(@times,depdS,sinp) - bsxfun(@times,demdS,sinm);        % m^2-by-D^2
        dv3 = bsxfun(@times,depdS,sinp) + bsxfun(@times,demdS,sinm);        % m^2-by-D^2
        dv4 = bsxfun(@times,demdS,cosm) - bsxfun(@times,depdS,cosp);        % m^2-by-D^2
        % Multiply by alpha and beta
        dv1 = aaj'*dv1; dv2 = abj'*dv2; dv3 = baj'*dv3; dv4 = bbj'*dv4; % 1-by-D^2
        
        dVdS(i,j,:,:) = reshape(0.5*(dv1+dv2+dv3+dv4) - M(i)*dMdS(j,:) - dMdS(i,:)*M(j),D,D); % D-by-D
        
 % 3) wrt controller weights, w
  % Both the exponential term and the trigonometric terms depend on w.
  % There is a quadratic form in the exponential term so this will be a
  % derivative plus its transpose.
        % Compute intermediate terms: derivatives of exp, and sin and cos
        dwipjSdw = wipj*S;                   dwimjSdw = wimj*S;                   % m^2-by-D
        depdw = -bsxfun(@times,dwipjSdw,ep); demdw = -bsxfun(@times,dwimjSdw,em); % m^2-by-D
        
        dcospdw = -bsxfun(@times,mu',sinp);  dcosmdw = -bsxfun(@times,mu',sinm);  % m^2-by-D
        dsinpdw = bsxfun(@times,mu',cosp);   dsinmdw = bsxfun(@times,mu',cosm);   % m^2-by-D
        
        % Form the terms in the brackets in V equation
        depcp = bsxfun(@times,depdw,cosp) + bsxfun(@times,ep,dcospdw); % m^2-by-D
        demcm = bsxfun(@times,demdw,cosm) + bsxfun(@times,em,dcosmdw); % m^2-by-D
        depsp = bsxfun(@times,depdw,sinp) + bsxfun(@times,ep,dsinpdw); % m^2-by-D
        demsm = bsxfun(@times,demdw,sinm) + bsxfun(@times,em,dsinmdw); % m^2-by-D
        dv1i = depcp+demcm; dv1j = depcp-demcm; % m^2-by-D
        dv2i = depsp-demsm; dv2j = depsp+demsm; % m^2-by-D
        dv3i = depsp+demsm; dv3j = depsp-demsm; % m^2-by-D
        dv4i = demcm-depcp; dv4j = -demcm-depcp; % m^2-by-D       

        % Multiply by alpha and beta
        dv1i = bsxfun(@times,a,etprod('12',reshape(dv1i,m,m,D),'132',aj,'3')); % m-by-D
        dv1j = bsxfun(@times,etprod('12',a,'3',reshape(dv1j,m,m,D),'312'),aj); % m-by-D
        dv2i = bsxfun(@times,a,etprod('12',reshape(dv2i,m,m,D),'132',bj,'3')); % m-by-D
        dv2j = bsxfun(@times,etprod('12',a,'3',reshape(dv2j,m,m,D),'312'),bj); % m-by-D
        dv3i = bsxfun(@times,b,etprod('12',reshape(dv3i,m,m,D),'132',aj,'3')); % m-by-D
        dv3j = bsxfun(@times,etprod('12',b,'3',reshape(dv3j,m,m,D),'312'),aj); % m-by-D
        dv4i = bsxfun(@times,b,etprod('12',reshape(dv4i,m,m,D),'132',bj,'3')); % m-by-D
        dv4j = bsxfun(@times,etprod('12',b,'3',reshape(dv4j,m,m,D),'312'),bj); % m-by-D
        
        dVdw(i,j,:,:,i) = 0.5*(dv1i+dv2i+dv3i+dv4i)-squeeze(dMdw(i,:,:,i))*M(j); % D-by-1
        if i==j; dVdw(i,j,:,:,i) = 2*dVdw(i,j,:,:,i);
        else dVdw(i,j,:,:,j) = 0.5*(dv1j+dv2j+dv3j+dv4j)-squeeze(dMdw(j,:,:,j))*M(i); end; % D-by-1
  
 % 4) wrt weights a & b, 
        dVda(i,j,:,i) = 0.5*((v1+(i==j)*v1')*aj + (v2+(i==j)*v3')*bj) - (1+(i==j))*squeeze(dMda(i,:,i))'*M(j);
        dVdb(i,j,:,i) = 0.5*((v3+(i==j)*v2')*aj + (v4+(i==j)*v4')*bj) - (1+(i==j))*squeeze(dMdb(i,:,i))'*M(j);      
        
        if i~=j; 
            dVda(i,j,:,j) = 0.5*(v1'*a + v3'*b) - squeeze(dMda(j,:,j))'*M(i);
            dVdb(i,j,:,j) = 0.5*(v2'*a + v4'*b) - squeeze(dMdb(j,:,j))'*M(i);
        end
    end
        
   
    %% Input-Output Covariance %%
    gamsth = [-gamma(m+1:end);gamma(1:m)].*theta;
    C(:,i) = [w' w']*(gamsth);     % D-by-E
    
% 1) wrt mu
    dCdmu(:,i,:) = [w' w']*(bsxfun(@times,[-dgamdm(m+1:end,:); dgamdm(1:m,:)],theta));   % D-by-D

% 2) wrt S
    dCdS(:,i,:,:) = reshape([w' w']*bsxfun(@times,[-dgamdS(m+1:end,:); dgamdS(1:m,:)],theta),[D,D,D]);   % D-by-D-by-D

% 3) wrt w
    dgdw = zeros(2*m,D);
    for k = 1:m; dgdw([k k+m],(k-1)*D+1:k*D) = [-dgamdw(k+m,:);dgamdw(k,:)]; end  % m-by-D*m
    chain1 = [w' w']*bsxfun(@times,dgdw,theta);                          % D-by-m*D
    chain2 = kron(gamsth(1:m)'+gamsth(m+1:end)',eye(D));                 % D-by-m*D
    dCdw(:,i,:,:,i) = permute(reshape(chain1 + chain2,D,D,m),[1,3,2]);   % D-by-m*D; 

% 4) wrt a & b
    dCda(:,i,:,i) = w'*diag(-gamma(m+1:end));        % RHS: D-by-m
    dCdb(:,i,:,i) = w'*diag(gamma(1:m));             % RHS: D-by-m
 
 end         

% Fill in other half of variance matrices
V = V + V' - diag(diag(V)); 
dVdmu = dVdmu + permute(dVdmu,[2 1 3]) - bsxfun(@times,eye(E),dVdmu);
dVdS = dVdS + permute(dVdS,[2 1 3 4]) - bsxfun(@times,eye(E),dVdS);
dVdw = dVdw + permute(dVdw,[2 1 3 4 5]) - bsxfun(@times,eye(E),dVdw);
dVda = dVda + permute(dVda,[2 1 3 4]) - bsxfun(@times,eye(E),dVda);
dVdb = dVdb + permute(dVdb,[2 1 3 4]) - bsxfun(@times,eye(E),dVdb);

if 1==wE; dMdw = sum(dMdw,4); dVdw = sum(dVdw,5); dCdw = sum(dCdw,5); end

% Vectorise derivatives
dMdS = reshape(dMdS,[E D*D]);
dVdS = reshape(dVdS,[E*E D*D]); dVdmu = reshape(dVdmu,[E*E D]);
dCdS = reshape(dCdS,[D*E D*D]); dCdmu = reshape(dCdmu,[D*E D]);

dMdp = [reshape(dMda,E,[])   reshape(dMdb,E,[])   reshape(dMdw,E,[])];
dVdp = [reshape(dVda,E*E,[]) reshape(dVdb,E*E,[]) reshape(dVdw,E*E,[])];
dCdp = [reshape(dCda,D*E,[]) reshape(dCdb,D*E,[]) reshape(dCdw,D*E,[])];