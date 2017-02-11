%wait for the results
while ~exist('state.txt', 'file')
    pause(1); %sleep for 1sec
end

%load the trial
nstate=load('state.txt');
newH = min([length(nstate) H]);
data(J+j).state = nstate(1:newH,1:6);
data(J+j).action = nstate(1:newH-1,7);

%cleaning
delete('state.txt');
