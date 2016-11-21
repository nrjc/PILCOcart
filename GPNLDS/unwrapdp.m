function dq = unwrapdp(dp)
% dp is a struct array of derivatives. This function concatentates along
% the second dimension, the fields first then along the array.

E = length(dp);
names = fieldnames(orderfields(dp));
dq = [];

for i=1:E
    for j=1:length(names);
        dq = [dq dp(i).(names{j})];
    end
end