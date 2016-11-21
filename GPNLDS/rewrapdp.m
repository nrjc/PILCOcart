function dq = rewrapdp(p,dp)

E = length(p);
names = fieldnames(orderfields(p));
dq = [];
for i=1:E
    for j=1:length(names)
        dq(i).(names{j}) = dp(:,1:numel(p(i).(names{j})));
        dp = dp(:,numel(p(i).(names{j}))+1:end);
    end
end