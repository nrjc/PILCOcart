% simple test of gp.m. Carl Edward Rasmussen, 2012-06-27

n = 300;
x = randn(n,2);
y = x*[2; -1] + sin(2*x(:,1)) + cos(3*x(:,2)) + 0.01*randn(n,1);
 
h.l = randn(2,1);
h.s = randn;
h.n = randn;

checkgrad('gp',h,1e-4,x,y);  % check derivs with no mean

h.m = randn(2,1);
h.b = randn;

checkgrad('gp',h,1e-4,x,y);  % check derivs with mean

h2 = minimize(h,'gp',struct('length',-100,'verbosity',3),x,y);
