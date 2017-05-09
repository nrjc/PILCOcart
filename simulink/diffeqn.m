clear all;
g=9.82;
lq = 0.6;
lp = 0.6;
M = 100;
p = 0.1;
q = 0.1;
%Vx, Vtheta, vPsi, x, theta(theta1), psi
Atopright = [0 (-q*g*p-p^2*g)/(p*M) 0 ; ...
    0 (-q*g*M-q*g*p-p*g*M-p^2*g)/(lp*p*M) (q*g)/(p*lp); ...
    0 -(q*g+g*p)/(p*lq) (g*p+g*q)/(p*lq)];
    
A = [zeros(3,3) Atopright;eye(3,3) zeros(3,3)];
B = [1 -1/lp zeros(1,4)]';
C = eye(6,6);
D = zeros(6,1);

%Create state space tf; 
G = ss(A,B,C,D);
delayterms(1)=struct('delay',0.05,'a',zeros(6,6),'b',B,'c',zeros(6,6),'d',zeros(6,1));
Gdel = pade(delayss(A,zeros(6,1),C,D,delayterms),10);


%Symbolic 
x = sym('x');
y = sym('y');
k = [x;y;zeros(4,1)];

%% Alternate approach: Lyapunov
Anew = A + B*k';
lambda = eig(Anew)
%% Failed approach: Gap metric. 

[K,CL,gam,info]=ncfsyn(G);
pole(feedback(G,-K))
gap = gapmetric(G,Gdel)
maxmargin = info.emax
[K,CL,gam,info]=ncfsyn(G);
Kctrl=tf(K);
[num, den]=tfdata(Kctrl);
%Why? How to interpret K? Why does the gapmetric mispredict?



%% Using their linearization:
sm_cart_dpen_linearize;
%Create state space tf; 
G = ss(sys_cart_dpen.A,sys_cart_dpen.B,sys_cart_dpen.C,sys_cart_dpen.D);
[K,CL,gam,info]=ncfsyn(G);
Kctrl=tf(K);
[num, den]=tfdata(Kctrl);