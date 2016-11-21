function [nlp, nlpdp] = multiTrial(p,dyn,plant,data)

NT = length(data); Np = length(unwrap(p)); 
nlp = zeros(1,NT); nlpdp = zeros(Np,NT);

if nargout < 2
     parfor i=1:NT; nlp(i) = GPf(p,dyn,plant,data(i)); end
else
    parfor i=1:NT;
        [nlp(i), nlpdpi] = GPf(p,dyn,plant,data(i));
        nlpdp(:,i) = unwrap(nlpdpi);
    end
end

nlp = sum(nlp); 
if nargout ==2; nlpdp = rewrap(p,sum(nlpdp,2)); end