function [M, S, V, Mdm, Sdm, Vdm, Mds, Sds, Vds] = lin0d(model, m, s)
% Linear model y = A*x + B*e where e ~ N(0,1)
% Joe Hall 2012-06-30
A = model.A; [E D] = size(A);
if isfield(model,'B'), B = model.B; else B = zeros(E,1); end
M = A*m;                                                           % mean
S = A*s*A' + B*B'; S = (S+S')/2;                             % covariance
V = A';                                  % inv(s)*input-output covariance
Mdm = A;            Sdm = zeros(E*E,D); Vdm = zeros(D*E,D);
Mds = zeros(E,D*D); Sds = kron(A,A);    Vds = zeros(D*E,D*D);

X=reshape(1:D*D,[D D]); XT=X'; Sds=(Sds+Sds(:,XT(:)))/2;     % symmetrise
X=reshape(1:E*E,[E E]); XT=X'; Sds=(Sds+Sds(XT(:),:))/2;