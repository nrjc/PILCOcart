%wait for the results
while ~exist('state.txt', 'file')
    pause(1); %sleep for 1sec
end

%load the trial
nstate=load('state.txt');
newH = min([size(nstate,1) H]);
data(J+j).state = nstate(1:newH,1:9);
data(J+j).action = nstate(1:newH-1,10);

%cleaning
delete('state.txt');
