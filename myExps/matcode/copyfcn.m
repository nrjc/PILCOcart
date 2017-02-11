function [M, S, V, dMdm, dSdm, dVdm, dMds, dSds, dVds] = copyfcn(model, m, s)
% This function is for copying through state variables to the next state in
% a dynmodel.sub framework
% copyMappings is a matrix that maps input to output in a one-to-one manner
% m         D-by-1, mean of the test distribution
% s         D-by-D, covariance matrix of the test distribution
%
% M         E-by-1, mean of pred. distribution 
% S         E-by-E, covariance of the pred. distribution             
% V         D-by-E, inv(s) times covariance between input and output
% dMdm      E-by-D, deriv of output mean w.r.t. input mean 
% dSdm      E^2-by-D, deriv of output covariance w.r.t input mean
% dVdm      D*E-by-D, deriv of input-output cov w.r.t. input mean
% dMds      E-by-D^2, deriv of ouput mean w.r.t input covariance
% dSds      E^2-by-D^2, deriv of output cov w.r.t input covariance
% dVds      D*E-by-D^2, deriv of inv(s)*input-output covariance w.r.t input cov

copyMappings = model.copyMappings;


D = length(copyMappings(1,:));
E = length(copyMappings(:, 1));

M = copyMappings*m;
S = copyMappings*s*copyMappings';
S = (S+S')/2;
V = copyMappings'; % This is the same as inv(s)*s*copyMappings' where s*copyMappings' is the input-output covariance
dMds = zeros(E, D^2);
dSdm = zeros(E^2, D);
dVdm = zeros(D*E, D);
dVds = zeros(D*E, D^2);
dMdm = copyMappings;

%if ~isfield(model, 'dSds') 
dSds = zeros(E, E, D, D);
dSds = bsxfun(@times,permute(copyMappings,[1,3,2,4]),permute(copyMappings,[3,1,4,2])); 
dSds = reshape(dSds,[E*E D*D]);
    %dynmodel.dSds = dSds;
%end
%dSds = dynmodel.dSds;
% for i=1:E
%     for j = 1:E
%         Position = zeros(D, D);
%         Position(i, j) = 1;
%         dSds(i, j, :, :) = copyMappings*Position*copyMappings';
%     end
% end
%dSds = reshape(dSds,[E*E D*D]);
end

