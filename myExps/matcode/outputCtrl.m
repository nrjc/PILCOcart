format short
fileID = fopen('/home/mmk48/Project/exps/centers.txt', 'w');
fprintf(fileID,'%f %f %f %f %f %f %f %f\n',ctrl.policy.p.inputs');
fclose(fileID);
fileID = fopen('/home/mmk48/Project/exps/weights.txt', 'w');
fprintf(fileID,'%f\n',ctrl.policy.p.target);
fclose(fileID);
fileID = fopen('/home/mmk48/Project/exps/W.txt', 'w');
fprintf(fileID,'%f\n',ctrl.policy.p.hyp(1:end-2));
fclose(fileID);


format short
fileID = fopen('centers.txt', 'w');
fprintf(fileID,'%f %f %f %f %f %f\n',ctrl.policy.p.inputs');
fclose(fileID);
fileID = fopen('weights.txt', 'w');
fprintf(fileID,'%f\n',ctrl.policy.p.target);
fclose(fileID);
fileID = fopen('W.txt', 'w');
fprintf(fileID,'%f\n',ctrl.policy.p.hyp(1:end-2));
fclose(fileID);
