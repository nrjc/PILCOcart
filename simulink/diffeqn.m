g=9.82;
lq = 0.6;
lp = 0.6;
M = 0.5;
p = 0.5;
q = 0.5;
Atopright = [0 (-q*g*p-p^2*g)/(p*M) 0 ; ...
    0 (-q*g*M-q*g*p-p*g*M-p^2*g)/(lp*p*M) (q*g)/(p*lp); ...
    0 -(q*g+g*p)/(p*lq) (g*p+g*q)/(p*lq)];
    
A = [zeros(3,3) Atopright;eye(3,3) zeros(3,3)];
B = [1 -1/lp zeros(1,4)]';
C = eye(6,6);
D = zeros(6,1);

%Create state space tf; 
G2 = inv(tf([1 0],[1])*eye(6,6)-A) 
G = ss(A,B,C,D);
delayterms(1)=struct('delay',0.05,'a',zeros(6,6),'b',B,'c',zeros(6,6),'d',zeros(6,1));
Gdel = pade(delayss(A,zeros(6,1),C,D,delayterms),10);
[K,CL,gam,info]=ncfsyn(G);


pole(feedback(Gdel,-K))
gap = gapmetric(G,Gdel)
maxmargin = info.emax
%Why? How to interpret K? Why does the gapmetric mispredict?
