figure
t = -2:0.01:2;
z1 = normpdf(t,0,0.5);
z2 = t*-4;
[ax,p1,p2] = plotyy(t,z1,t,z2);
ylabel(ax(1),'P(x_{t})') % label left y-axis
ylabel(ax(2),'u_{t}') % label right y-axis
xlabel(ax(1),'x_{t}') % label x-axis
set(ax,'FontSize',20);
set(gca,'FontSize',20);
print -depsc variance_state_t

figure
t = -2:0.01:2;
z1 = normpdf(t,0,0.2);
plot(t,z1);
ylabel('P(x_{t+1})') % label left y-axis
xlabel('x_{t+1}') % label x-axis
set(gca,'FontSize',20);
print -depsc variance_state_t_1