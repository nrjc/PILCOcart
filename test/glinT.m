% The script tests the output of the function gp/glin.m
% Jonas Umlauft 
% 2014-04-09

%clear
%clc
try
  rd = '../';
  addpath([rd 'util'],[rd 'gp']);
catch
end
dynmodel.outputDIM   = 2; % Output dimension
dynmodel.inputDIM    = 4; % Input dimension
dynmodel.controlDIM  = 1; % Control dimension

E = dynmodel.outputDIM;    % Output dimension
D = dynmodel.inputDIM;     % Input dimension
C = dynmodel.controlDIM;   % Control dimension
DC = D+C;                  % size of augmented state


nni = D+E+DC*E+C;           % size of m
nno = 2*E + DC*E;         % size of M


% Generate mean of x,b and A
if not(exist('m','var')); m = rand(nni,1);end
% Ge~nerate positive definite symmetric covarianz matrix
if not(exist('s','var')); s = rand(D+E+(D+C)*E+C);  s = s'*s; end


% Indices for inputs
ix = [1:D  D+E+DC*E+1 : nni];
ib = D+1 : D+E;
iA = D+E+1 : D+E+DC*E;

% Indices for outputs
Ix = 1:E; 
Ib = E+1:2*E;
IA = 2*E+1 : 2*E + E*DC;


mx = m(ix); mb = m(ib); mA = reshape(m(iA),E,DC);
Vx = s(ix,ix); Vb = s(ib,ib); VA = s(iA,iA);
CAx = s(iA,ix); Cxb = s(ix,ib); CAb = s(iA,ib);
CbA = CAb'; CxA = CAx';Cbx = Cxb';


%% Test M S V
n=1e5;

xbA = (bsxfun(@plus,m,chol(s)'*randn(nni,n)))';
s=cov(xbA);
m=mean(xbA)';

% Run linear model
%[M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = glin(dynmodel,m,s);
tic
[M, S, V] = glin(dynmodel,m,s);
toc
% Extract returned means
mX = M(1:E); mb = M(E+1:2*E); mA = reshape(M(2*E+1:end),E,DC);
% Extract returned variances and covariances
VX = S(1:E,1:E); Vb = S(E+1:2*E,E+1:2*E); VA = S(2*E+1:end,2*E+1:end);
CAx = S(2*E+1:end,1:E); Cxb = S(1:E,E+1:2*E);




% Generate  samples from normal distribution with mean m and variance s 
% and run linear model on these samples

x = xbA(:,ix); b = xbA(:,ib);
A = zeros(E,DC,n);
X = zeros(E,n);
for ii=1:n
    A(:,:,ii) = reshape(xbA(ii,iA),E,DC);
    X(:,ii) = A(:,:,ii)*(x(ii,:))' + (b(ii,:))';
end
X = X';

XbA = [X xbA(:,[ib iA])];

Difference_M = M' - mean(XbA);
display(['Relative Error M: ', num2str(norm(Difference_M)/norm(M'))]);
Diff_S = S  - cov(XbA);
display(['Relative Error S: ', num2str(norm(Diff_S)/norm(S))]);
C_xbA_XbA_num = cov([xbA XbA]);
Diff_sV = s*V - C_xbA_XbA_num(1:nni,nni+1:end);
display(['Relative Error s*V: ', num2str(norm(Diff_sV)/norm(s*V))]);



%% Test Derivatives
dynmodel.fcn = @glin;
derivs = {'dMdm','dSdm','dVdm','dMds','dSds','dVds'};


    
for i=1:numel(derivs)
    [dd, dy, dh] = gpT(derivs{i}, dynmodel, m, s);
end




%% Symbolic represtation of the mapping for debuging 
% 
% mx_sym = sym('x',[D 1]);  mb_sym = sym('b',[E 1]); mA_sym  = sym('a',[E DC]);
% mu_sym = sym('u',[C 1]);  Mx_sym = sym('X',[E 1]); 
% 
% m_sym = [mx_sym;mb_sym;mA_sym(:);mu_sym]; 
% s_sym = m_sym*reshape(m_sym,1,numel(m_sym));
% 
% M_sym = [Mx_sym;mb_sym;mA_sym(:)];
% S_sym = M_sym*reshape(M_sym,1,numel(M_sym));
% V_sym = m_sym*reshape(M_sym,1,numel(M_sym));


%s_sym_vec_tril = [1:numel(s_sym(tril(ones(nni))==1));reshape(s_sym(tril(ones(nni))==1),1,numel(s_sym(tril(ones(nni))==1)))]
%S_sym_vec_tril = [1:numel(S_sym(tril(ones(nno))==1));reshape(S_sym(tril(ones(nno))==1),1,numel(S_sym(tril(ones(nno))==1)))]

%s_sym_vec = [1:numel(s_sym);reshape(s_sym,1,numel(s_sym))];
%m_sym_vec = [1:numel(m_sym);reshape(m_sym,1,numel(m_sym))];
%M_sym_vec = [1:numel(M_sym);reshape(M_sym,1,numel(M_sym))];

%S_sym_vec =[1:numel(S_sym);reshape(S_sym,1,numel(S_sym))];
%V_sym_vec = [1:numel(V_sym);reshape(V_sym,1,numel(V_sym))];
 
