function df2 = calcdf2m(lh,x,preC,xs)

if nargin < 4; xs = x; end
[N D] = size(x); E = size(preC.y,2);

S = diag(exp(2*lh.lsipn));

dynmodel.hyp = lh.seard; dynmodel.inputs = x; dynmodel.target = preC.y;
ipK = df2toipK(lh, preC.df2); dynmodel.noise = zeros(N,E); 
for i = 1:E; dynmodel.noise(:,i) = diag(ipK(:,:,i)); end

Ns = size(xs,1); df2 = zeros(Ns,E,D);

for i = 1:Ns
  [m S1] = gp0(dynmodel,xs(i,:)',S);
  
  for j = 1:D
      SS = S; SS(j,j) = 0;
      [m S2] = gp0(dynmodel,xs(i,:)',SS);
      df2(i,:,j) = diag(S1-S2)/exp(2*lh.lsipn(j)); 
  end
end

df2 = max(df2,0);