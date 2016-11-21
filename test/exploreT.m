function exploreT

% Tests each explore function.
% Rowan McAllister 2015-08-04

clc
DELTA = 1e-5;
EPSILON = 1e-6;               % 'pass' threshold for low enough checkgrad error
r = round(rand*10000)
rng(r)

this_dir = mfilename('fullpath');
this_dir = this_dir(1:end-length(mfilename)-1); % remove file name
addpath([this_dir,'/../explore']);

dbstop if error

% random inputs:
cc_prev.m = rand(1);  %#ok<*AGROW>
cc_prev.s = rand(1);
cc.m = 0.8*cc_prev.m;
cc.s = 0.8*cc_prev.s;
n = 4; % number of trials left
gamma = 0.7;

i = 0; ntests = 6; tests = cell(ntests,1); cg = cell(ntests,1); 
i = i+1; tests{i} = 'explorePI';                           expl = [];     [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, n,     tests{i}, expl); cg{i} = {d,dy,dh};
i = i+1; tests{i} = 'exploreEI';                           expl = [];     [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, n,     tests{i}, expl); cg{i} = {d,dy,dh};
i = i+1; tests{i} = 'exploreUCB';                          expl.beta=0.5; [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, n,     tests{i}, expl); cg{i} = {d,dy,dh};
i = i+1; tests{i} = 'exploreMyopicGittinsFiniteHorizon';   expl.on=0;     [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, n,     tests{i}, expl); cg{i} = {d,dy,dh};
i = i+1; tests{i} = 'exploreMyopicGittinsFiniteHorizon';   expl.on=1;     [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, n,     tests{i}, expl); cg{i} = {d,dy,dh};
i = i+1; tests{i} = 'exploreMyopicGittinsInfiniteHorizon'; expl.on=0;     [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, gamma, tests{i}, expl); cg{i} = {d,dy,dh};
i = i+1; tests{i} = 'exploreMyopicGittinsInfiniteHorizon'; expl.on=1;     [d,dy,dh] = checkgrad(@swap, cc, DELTA, cc_prev, gamma, tests{i}, expl); cg{i} = {d,dy,dh};
print_derivative_test_results(tests, cg, EPSILON)

function [f, df] = swap(cc, cc_prev, n_or_gamma, test, expl)
if nargout < 2
  f = feval(test, expl, cc, cc_prev, n_or_gamma);
else
  [f, df] = feval(test, expl, cc, cc_prev, n_or_gamma);
end
