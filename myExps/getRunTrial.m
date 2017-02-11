%wait for the results
while ~exist('signal.txt', 'file') || ~exist('cart_pos.txt', 'file') ... 
       || ~exist('cart_vel.txt', 'file') || ~exist('cam_ang.txt', 'file')... 
       || ~exist('cam_angv.txt', 'file') || ~exist('p_signal.txt', 'file')
    pause(1); %sleep for 1sec
end

%load the trial
nstate_tmp=load('p_signal.txt');
newH = min([length(nstate_tmp) H]);
nstate=zeros(newH,D);
nstate(:,1)=nstate_tmp(1:newH);
nstate_tmp=load('cart_pos.txt');
nstate(:,2)=nstate_tmp(1:newH);
nstate_tmp=load('cam_ang.txt');
nstate(:,3)=nstate_tmp(1:newH);
nstate_tmp=load('cart_vel.txt');
nstate(:,4)=nstate_tmp(1:newH);
nstate_tmp=load('cam_angv.txt');
nstate(:,5)=nstate_tmp(1:newH);

data(J+j).state = nstate;
naction_tmp=load('signal.txt');
data(J+j).action = naction_tmp(1:newH-1)';

%cleaning
delete('p_signal.txt', 'signal.txt','cart_pos.txt', 'cart_vel.txt', 'cam_ang.txt', ...
       'cam_angv.txt');