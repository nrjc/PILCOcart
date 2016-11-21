function	C= coriolis(m,l,cl,Q,Qv);

s2= sin(Q(2));
C= zeros(2,1);
C(1)= -m(2)*l(1)*cl(2)*s2*Qv(2)*(2*Qv(1) + Qv(2));
C(2)= m(2)*l(1)*cl(2)*s2*Qv(1)^2;



