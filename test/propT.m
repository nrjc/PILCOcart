%function [s dsdp] = propT(p, s, dyn, ctrl, i)
%ctrl2 = ctrl;
%ctrl2.policy.p = p;
%[s, ~, sdp] = propagated(s, dyn, ctrl2);
%if i < 6, s = s.m(i); else s = s.s(i-5); end
%dsdp = sdp(i,:);


function [s dsds] = propT(s, dyn, ctrl, i)
e = length(s); d = round(sqrt(8*e+9)/2-3/2);
t.m = s(1:d); t.s = zeros(d); 
t.s(tril(ones(d))==1) = s(d+1:e); t.s = t.s+t.s'-diag(diag(t.s));
[s sds] = propagated(t, dyn, ctrl);
if i < 6, s = s.m(i); else s = s.s(i-5); end
dsds(1:d) = sds(i,1:d);
z = reshape(sds(i,d+1:end),d,d); z = 2*z-diag(diag(z)); 
dsds(d+1:e) = z(tril(ones(d))==1);
