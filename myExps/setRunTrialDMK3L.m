function setRunTrialDMK3L(ctrl)
	%output series of linear models
	format short
	fileID = fopen('weights.txt', 'w');
	fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n',ctrl.policy.p.w);
	fclose(fileID);

	fileID = fopen('biases.txt', 'w');
	fprintf(fileID,'%f\n',ctrl.policy.p.b);
	fclose(fileID);
