ndata = struct([]);
myi=1;

mythree=mvnpdf(news.m - diag(news.s), news.m, news.s);

for myj=1:j
likelihoods=mvnpdf(data(myj).state, news.m', news.s);
myind=find(likelihoods>mythree);
if (isempty(myind) == 0)
nstate=data(myj).state(myind:end,:);
naction=data(myj).action(myind:end);
ndata(myi).state=nstate;
ndata(myi).action=naction;
myi=myi+1;
end
end




npolicy.p=struct([]);
for myi=23:40
	npolicy.p(myi-22).w = ctrl.policy.p(myi).w;
	npolicy.p(myi-22).b = ctrl.policy.p(myi).b;
end
ctrl.set_policy_p(npolicy.p);