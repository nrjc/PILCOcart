%wait for the results
while ~exist('state.txt', 'file')
    pause(1); %sleep for 1sec
end
%Number of initial trials
J=4;
%load the trial
if (j>1)
    nstate=load('state.txt');
    newH = min([size(nstate,1) H+1]);
    data(J+j).state = nstate(1:newH,1:12);
    data(J+j).action = nstate(1:newH-1,13);
else
    for i=1:J
        nstate=load(['state' int2str(i-1) '.txt']);
        newH = min([size(nstate,1) H+1]);
        data(i+j).state = nstate(1:newH,1:12);
        data(i+j).action = nstate(1:newH-1,13);
    end
end
%cleaning
delete('state.txt');
