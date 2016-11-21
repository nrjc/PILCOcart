function ctrlNFT(ctrl, s, delta)

% Test the ctrlNF function. Check the three outputs using Monte Carlo, and the
% derivatives using finite differences. 
% 
% CTRLNFT(ctrl, s, delta):
%   ctrl          controller object
%   s             state structure
%   delta         finite difference (default 1e-4)
%
% Copyright (C) 2015 by Carl Edward Rasmussen, Rowan McAllister 2015-04-06

NSAMPLES = 1e4;
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
SEED = 1; rng(SEED);
dbstop if error

% Gradients to test:
douts ={'uM', 'uS', 'uC', 'm', 's'};
dins ={'m', 's', 'p'};

% 1. SET CTRLNF INPUTS --------------------------------------------------------

plant = create_test_object('plant', 'cartDoublePendulum');
if ~exist('dyn','var'); dyn = create_test_object('dyn', plant); end
if ~exist('ctrl','var'); ctrl = create_test_object('CtrlNF', plant, dyn); end
assert(isa(ctrl,'CtrlNF'), 'ctrlNFT: controller input muct be class CtrlNF');
if ~exist('s','var'); s = create_test_object('state', plant, ctrl); end
if nargin < 3; delta = 1e-4; end                % checkgrad's finite difference

F = ctrl.F; on = ctrl.on; U = ctrl.U;

% 2. TEST CTRLNF OUTPUT-DERIVATIVES -------------------------------------------

ntests = numel(dins)*numel(douts);
test_names = cell(ntests,1); cg = cell(ntests,1); i = 0;
for din = dins;
  for dout = douts;
    i = i+1;
    test_names{i} = strcat('d(',dout{:},')/d(',din{:},'): ');
    cg{i} = cg_wrap(s, ctrl.policy.p, ctrl, nan, delta, dout{:}, din{:});
  end
end
print_derivative_test_results(test_names, cg, EPSILON)

% 3. TEST CTRLNF OUTPUTS ------------------------------------------------------

% 3.1. Compute Analytic Outputs
[uMa , uSa , uCa, snext] = ctrl.fcn(s);
[uMa_, uSa_, uCa_, snext_,~,~,~,~,~,~,~,~] = ctrl.fcn(s);
max_diff = @(a,b) (max(abs(unwrap(a)-unwrap(b))));
md1 = max_diff({uMa_,uSa_,uCa_,snext_}, {uMa,uSa,uCa,snext});
if md1 > 1e-10;
  warning('MATLAB:paramAmbiguous',...
    'ctrlNFT: requesting derivatives alters non-derivative outputs by %2.1e', md1);
end
assert(max_diff(s, snext) < 1e-10, ...
  'ctrlNFT: ctrlNF should not alter the state structure.');
[uMa , uSa , uCa] = ctrl.fcn(s);

% 3.2. Compute Numeric Outputs: Case {uM,uS} outputs                            % TODO: fix s.t. we do not have 2 test cases.
%x = mvnrnd(s.m,s.s+n,NSAMPLES);                              % sample state
x = mvnrnd(s.m,s.s,NSAMPLES);      % noise-free samples
y = x + mvnrnd(0*s.m,on,NSAMPLES); % noisy samples
uMi = nan(NSAMPLES,U); uSi = nan(NSAMPLES,U,U);
si.s = zeros(F);
for i=1:NSAMPLES
  print_loop_progress(i,NSAMPLES,'Testing U outputs with MC');
  si.m = x(i,:)';
  [uMi(i,:), uSi(i,:,:)] = ctrl.fcn(si);
end
uMn = mean(uMi,1)';
uSn = cov(uMi) + squeeze(mean(uSi,1));
uCn = nan(F,U);
for d=1:F, for u=1:U, c=cov(x(:,d)',uMi(:,u)); uCn(d,u)=c(1,2); end; end

% dislpay numeric vs. analytic
str = @(x) (num2str(unwrap(x)'));
fprintf('\nCTRLNF.M MONTE CARLO TEST RESULTS: (nsamples %3.0e):\n', NSAMPLES);
fprintf('==================\n');
fprintf('uM   numeric  : %s\n', str(uMn));
fprintf('uM   analytic : %s\n', str(uMa));
fprintf('------------------\n');
fprintf('uS   numeric  : %s\n', str(uSn));
fprintf('uS   analytic : %s\n', str(uSa));
fprintf('------------------\n');
fprintf('uC   numeric  : %s\n', str(uCn));
fprintf('uC   analytic : %s\n', str(uCa));
fprintf('==================\n');
mdiff = max_diff({uMn,uSn,uCn}, {uMa,uSa,uCa});
fprintf('Maximum difference = %4.2e\n', mdiff);

% 4. FUNCTIONS ----------------------------------------------------------------

% Checkgrad wrapper. Swaps arg order required for, then executes, checkgrad.
function cg = cg_wrap(varargin)
[orig_policyp, ~, ~, delta, dout, din] = deal(varargin{end-5:end});
disp([mfilename,': derivative test: d(',dout,')/d(',din,')']);
if din == 'p', x = orig_policyp; else x = varargin{1}.(din); end
[d,dy,dh] = checkgrad(@cg_f,x,delta,varargin{:}); cg = {d dy dh};

% Checkgrad input function. Updates test function f's inputs w.r.t. checkgrad's
% pertubation, then evaluates f.
% varargin: fcn inputs: ctrl.fnc(s)
function [f, df] = cg_f(x,varargin)
[orig_policyp,ctrl,~,~,dout,din] = deal(varargin{end-5:end});
s = varargin{1}; is = ctrl.is;
if strcmp(din,{'s'}); x = (x+x')/2; end
if strcmp(din,'p'), ctrl.set_policy_p(x); else s.(din) = x; end
if nargout == 1 && any(strcmp(dout,{'uM','uS','uC'}))
  [uM, uS, uC] = ctrl.fcn(s);
elseif nargout == 1
  [~,~,~,s] = ctrl.fcn(s);
else
  [uM,uS,uC,s,duMds,duSds,duCds,dsds,duMdp,duSdp,duCdp,dsdp]=ctrl.fcn(s);
end
switch dout
  case 't'; f = t; df = dt;
  case 'uM'; f = uM;
    if nargout == 2
      if din == 'p', df = duMdp; else df = duMds(:,is.(din)); end
    end
  case 'uS'; f = uS;
    if nargout == 2
      if din == 'p', df = duSdp; else df = duSds(:,is.(din)); end
    end
  case 'uC'; f = uC;
    if nargout == 2
      if din == 'p', df = duCdp; else df = duCds(:,is.(din)); end
    end
  otherwise; f = s.(dout);
    if nargout == 2
      if din == 'p', df = dsdp(is.(dout),:);
      else df = dsds(is.(dout),is.(din)); end
    end
end
ctrl.set_policy_p(orig_policyp);  % reset