%RUN TRIAL

%ouput
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

%wait for the results
while ~exist('signal.txt', 'file') || ~exist('cart_pos.txt', 'file') ... 
       || ~exist('cart_vel.txt', 'file') || ~exist('cam_ang.txt', 'file')... 
       || ~exist('cam_angv.txt', 'file')
    pause(1); %sleep for 1sec
end

%load the trial
nstate=zeros(H,D);
nstate(:,1)=load('signal.txt');
nstate(:,2)=load('cart_pos.txt');
nstate(:,3)=load('cart_vel.txt');
nstate(:,4)=load('cam_ang.txt');
nstate(:,5)=load('cam_angv.txt');

%cleaning
delete('signal.txt','cart_pos.txt', 'cart_vel.txt', 'cam_ang.txt', ...
       'cam_angv.txt');

data(J+j).state = nstate; %data(J+j).action = [nstate(2:end,1)' 0]';
data(J+j).action = nstate(2:end,1);