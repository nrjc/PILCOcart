function	H= mass(m,l,cl,I,Q)

c2= cos(Q(2));

h11= I(1) + m(1)*cl(1)^2 + I(2) + m(2)*(l(1)^2 + cl(2)^2 + 2*l(1)*cl(2)*c2);
h12= I(2) + m(2)*(cl(2)^2 + l(1)*cl(2)*c2);
h22= I(2) + m(2)*cl(2)^2;

H= [h11, h12; h12, h22];



