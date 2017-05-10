function setRunTrial(ctrl)
	%output Linear Policy
	format short
	fileID = fopen('weights.txt', 'w');
	fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f\n',ctrl.policy.p.w);
	fclose(fileID);

	fileID = fopen('bias.txt', 'w');
	fprintf(fileID,'%f\n',ctrl.policy.p.b);
	fclose(fileID);
