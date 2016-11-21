l(1)= 0.29;     % length of upper arm [m]
l(2)= 0.34;     % length of forearm [m]
ang = [1 2];

if ~ishandle(5); figure(5); else set(0,'CurrentFigure',5); end; clf(5); hold on;
for i = 1:Nroll
    angles = lat{i}(:,ang);
    [x, y] = cartSpace(l, angles);
    plot(x, y, 'r');
end
angles = latent{j+J}(:,ang);
[x, y] = cartSpace(l, angles);
plot(x, y, 'g');

[x, y] = cartSpace(l, mu0');
plot(x, y, 'kx')            % plot starting location mean
[x, y] = cartSpace(l, cost.z');
plot(x, y, 'ko')            % plot target location

axis equal
v = axis;
axis(v+0.1*[-1 1 -1 1]);    % broaden the view of the plot slightly
title('Hand paths over multiple rollouts');
xlabel('x position (m)');
ylabel('y position (m)');
hold off