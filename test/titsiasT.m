% test derivatives gp/titsias
% Rown McAllister 2014-12-17

SEED=1;
rng(1);
clc
dbstop if error

gp.E = 2;
gp.D = 3;
F = gp.D + 2;
[M, uF, uE] = deal(50,F,gp.E);
N = 2*M;
induce = randn(M, uF, uE);
gp.inputs = randn(N, F);

gp.target = randn(N,gp.E);
for i = 1:gp.E
  hyps(i).l = log(100*rand(F,1));
  hyps(i).m = randn(F,1);
  hyps(i).b = randn(1,1);
  hyps(i).n = log(rand(1,1));
  hyps(i).s = log(rand(1,1));
end
gp.hyp = hyps;

profile on
[nlml, dnlml] = titsias(induce,gp);
profile off
profile viewer

[dd] = checkgrad(@titsias, induce, 1e-4, gp)