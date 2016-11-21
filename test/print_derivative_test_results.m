function print_derivative_test_results(test, cg, epsilon)

% Summarises the result of individual derivative test results.
%
% test       individual test descriptions (cell array of string)
% cg         individual checkgrad outputs (cell of cells of doubles [d dy dh])
% epsilon    'pass' accuracy-threshold (double)
%
% Rowan McAllister 2014-10-07

calling_fcn = dbstack; calling_fcn = calling_fcn(2).name;
if isempty(test)
  disp([calling_fcn,': No output-gradients selected for testing.']); return
end

if nargin < 3; epsilon = 1e-5; end
ntest = numel(test); n = nan(ntest,1);
max_d=n; max_reld=n; max_rawd=n; min_dh=n; max_dh=n;
for i=1:ntest
  max_d(i)  = max(cg{i}{1}(:));           % norm of diff divided by norm of sum
  dh = abs(cg{i}{3}(:));
  dy = abs(cg{i}{2}(:));
  max_reld(i) = 2*max(max(abs((dh-dy)./(dh+dy))));          % max relative diff
  max_rawd(i) = max(max(abs(dh-dy)));                       % max raw diff
  ldh = round(log10(dh(dh>0)));
  if isempty(ldh); ldh = -inf; end
  min_dh(i) = min(ldh);
  max_dh(i) = max(ldh);
end

% Overall result
if ~any(max_d>epsilon); result = 'PASS'; else result = 'FAIL'; end
fprintf('\n%s: Overall derivative test result: %s (max error %3.1e)',...
  calling_fcn, result, max(max_d));
fprintf(['\n%s: Individual derivative test results', ...
  ' (''pass'' means checkgrad error < %3.0e):'], calling_fcn, epsilon);

% Individual results
fprintf('\n   Derivative \t\t\tError (d) \tRel. Error \tRaw Error \tScale (log10 dh)');
for i = 1:ntest
  if ~(max_d(i)>epsilon); result = 'pass\t'; else result = '\tFAIL'; end
  fprintf(['\n   %-10s\t ',result,...
    ' \ter. %3.0e \trel. %3.0e \traw. %3.0e \t[%4i,%4i]'], ...
    test{i}, max_d(i), max_reld(i), max_rawd(i), min_dh(i), max_dh(i));
end
fprintf('\n');