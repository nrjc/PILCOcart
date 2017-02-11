% load('/Users/mk74/4f13/prac1/cw1d.mat')
figure;
z = linspace(-3, 3, 101)';
x=[1 -1 0]'; y=[0.5 -0.5 1]';
hyp2 = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, likfunc, x, y);
[m s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '*', 'MarkerSize', 20);
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_sparse

figure
x=[1.05 0.95 1.03 0.99 -1.05 -1 -0.9 -0.05 0.05]'; y=[0.55 0.45 0.47 0.55 -0.55 -0.5 -0.45 0.99 1.01]';
z = linspace(-3, 3, 101)';
[m s2] = gp(hyp2, @infExact, meanfunc, covfunc, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '*', 'MarkerSize', 20);
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_full