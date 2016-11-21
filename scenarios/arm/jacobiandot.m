function Jdot=jacobiandot(l,q,qv);

l1=l(1);l2=l(2);
q1=q(1);q2=q(2);
q12=q1+q2;
qv1=qv(1);qv2=qv(2);
qv12=qv1+qv2;

Jdot(1,1)=-l1*qv1*cos(q1)-l2*qv12*cos(q12);
Jdot(1,2)=-l2*qv12*cos(q12);
Jdot(2,1)=-l1*qv1*sin(q1)-l2*qv12*sin(q12);
Jdot(2,2)=-l2*qv12*sin(q12);