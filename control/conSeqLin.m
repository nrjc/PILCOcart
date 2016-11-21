function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, ...
                                  dMdp, dSdp, dVdp] = conSeqLin(policy, m, s)
% Sequence of linear controllers                                  

%build policy params
global currT;
tpolicy.p=policy.p(currT);

%run linear policy with built policy params
[M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds, tdMdp, tdSdp, tdVdp] = conlin(tpolicy, m, s);  

%complete dMdp, dSdp, dVdp                               
Nparams=length(tpolicy.p.w)+length(tpolicy.p.b); H=length(policy.p);
startPos=(currT-1)*Nparams+1; endPos=currT*Nparams;
dMdp=zeros(1,H*Nparams);	
dMdp(1,startPos:endPos)=tdMdp;
dSdp=zeros(1,H*Nparams);
dSdp(1,startPos:endPos)=tdSdp;
dVdp=zeros(length(tpolicy.p.w),H*Nparams);
dVdp(:,startPos:endPos)=tdVdp;

%update time step
currT=currT+1;