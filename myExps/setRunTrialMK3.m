function setRunTrial(ctrl)
	%output RBF policy
	format short
	fileID = fopen('centers.txt', 'w');
	fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f\n',ctrl.policy.p.inputs');
	fclose(fileID);

	fileID = fopen('W.txt', 'w');
	fprintf(fileID,'%f\n',ctrl.policy.p.hyp(1:end-2));
	fclose(fileID);

	nN=length(ctrl.policy.p.target);
	nD = length(ctrl.policy.p.hyp)-2;
	X = ctrl.policy.p.hyp;
	K = zeros(nD,nD); beta = zeros(nD);
	    
	inp = bsxfun(@rdivide,ctrl.policy.p.inputs,exp(X(1:nD)'));
	K = exp(2*X(nD+1)-maha(inp,inp)/2);
	L = chol(K(:,:) + exp(2*X(nD+2))*eye(nN))';
	beta = L'\(L\ctrl.policy.p.target);

	fileID = fopen('weights.txt', 'w');
	fprintf(fileID,'%f\n', beta);
	%fprintf(fileID,'%f\n', ctrl.policy.p.target);
	fclose(fileID);
