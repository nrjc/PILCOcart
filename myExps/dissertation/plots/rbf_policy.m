load('rbfactt.mat');
imagesc(log(abs(rbfactt)))
set(gca,'FontSize',20);
xlabel('Time step');
ylabel('RBF function');
print -depsc rbf_actt

figure;
plot(rbfactt(5,:));
hold on;
plot(rbfactt(28,:))
hold on;
plot(rbfactt(5,:) + rbfactt(28,:));
set(gca,'FontSize',20);
xlabel('Time step');
ylabel('Mean output contribution');
print -depsc rbf_cancel

