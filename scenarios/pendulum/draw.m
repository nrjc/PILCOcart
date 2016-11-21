function draw(s, u, cost, bm, bv, str1, str2)
% function draw(plant,theta, torque, cost, s, s2, maxU,M, S)
% Draws the pendulum system
%
% (C) Copyright 2009-2012 by Carl Edward Rasmussen and Marc Deisenroth,
% 2012-06-25.
% Edited Rowan 2015-08-07

l = cost.ell; % length of pendulum
xmin = -1.2*l;
xmax = 1.2*l;
height = 0;
theta = s.m(end);
maxU = 2;

% Draw pendulum
pendulum = [0, 0; l*sin(theta), l*cos(theta)];
clf; hold on
plot(pendulum(:,1), pendulum(:,2),'r','linewidth',4)

% Draw error ellipses
try
  bm = bm(theta);
  bv = bv(theta,theta);
  err = linspace(-1,1,100)*sqrt(bv);
  plot(l*sin(bm+2*err),-l*cos(bm+2*err),'b','linewidth',1)
  plot(l*sin(bm+err),-l*cos(bm+err),'b','linewidth',2)
  plot(l*sin(bm),-l*cos(bm),'b.','markersize',20)
catch
end

% Draw stuff
plot(0,l,'k+','MarkerSize',20);
plot([xmin, xmax], [height, height],'k','linewidth',2)
plot(0,0,'k.','markersize',24)
plot(0,0,'y.','markersize',14)
plot(l*sin(theta),l*cos(theta),'k.','markersize',24)
plot(l*sin(theta),l*cos(theta),'y.','markersize',14)
plot(0,-2*l,'.w','markersize',0.005)

% Draw useful information
plot([0 u/maxU*xmax],[-0.5, -0.5],'g','linewidth',10);
computed_reward = true;
try
  reward = 1-cost.fcn(s).m;
catch
  computed_reward = false;
end
if computed_reward; plot([0 reward*xmax],[-0.7, -0.7],'y', 'linewidth',10); end
text(0,-0.5,'applied  torque')
if computed_reward; text(0,-0.7,'immediate reward'); end
if exist('str1','var'); text(0,-0.9, str1); end
if exist('str2','var'); text(0,-1.1, str2); end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-2*l 2*l]);
axis off;
drawnow;