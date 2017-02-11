%load('/Users/mk74/4f13/prac1/cw1d.mat')
meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;
hyp.cov = [log(0.5) 0];
hyp.lik = log(0.1);

figure;
x=[]; y=[];
z = linspace(-3, 3, 101)';
m=0 * ones(101,1); s2=1.01 * ones(101,1);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+');
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_nosamples

figure;
x=[1]; y=[0.5];
z = linspace(-3, 3, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '*', 'MarkerSize', 20);
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_1sample

figure;
x=[1 -1]'; y=[0.5 -0.5]';
z = linspace(-3, 3, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '*', 'MarkerSize', 20);
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_2samples

figure;
x=[1 -1 0]'; y=[0.5 -0.5 1]';
z = linspace(-3, 3, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '*', 'MarkerSize', 20);
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_3samples





figure;
x=[1 -1 0]'; y=[0.5 -0.5 1]';
z = linspace(-3, 3, 101)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '*', 'MarkerSize', 20);
set(gca,'FontSize',20);
xlabel('x_{*}');
ylabel('y_{*}');
print -depsc gp_3samples