function trainDirect(dyn, data, dyni, dyno, init)

% train the dynamics model using the direct method: the (approx) negative log
% marginal likelihood wrt hyper parameters and beta values for the dynamics
% model gp. First, turn the non-parametric model into a parametric one.
%
% Copyright (C) 2015 by Carl Edward Rasmussen, 2015-03-24

if init
  dyn.train(data, dyni, dyno);
  if numel(dyn.induce)                  % if sparse, then convert to parametric
    dyn.pre2();
  end
end

p = dyn.hyp;                                 % finally, set up the parameters p
for i=1:dyn.E, p(i).beta = dyn.beta(:,i); end
for i=1:dyn.E, p(i).on = dyn.on(i); p(i).pn = dyn.pn(i); end
p = minimize(p, @multiTrial, dyn.opt, dyn, data, dyni, dyno);
