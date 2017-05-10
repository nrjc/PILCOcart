%wait for the results
while ~exist('state.txt', 'file')
    pause(1); %sleep for 1sec
end

%load the trial
nstate=load('state.txt');
%newH = min([size(nstate,1) H+1]);
newH = size(nstate,1); %Overriding length of data to copy for first state.
data(J+j).state = nstate(1:newH,1:12);
data(J+j).action = nstate(1:newH-1,13);

%cleaning
delete('state.txt');
