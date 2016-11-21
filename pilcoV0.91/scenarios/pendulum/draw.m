function draw(theta, torque, cost, s, s2, M, S)
% Draws the pendulum system
%
% (C) Copyright 2009-2012 by Carl Edward Rasmussen and Marc Deisenroth, 
% 2012-06-25.

l = 0.6;
xmin = -1.2*l; 
xmax = 1.2*l;    
umax = 0.5;
height = 0;

% Draw pendulum
pendulum = [0, 0; l*sin(theta), -l*cos(theta)];
clf; hold on
plot(pendulum(:,1), pendulum(:,2),'r','linewidth',4)

% Draw error ellipses
if nargin > 5
  err = linspace(-1,1,100)*sqrt(S(2,2));
  plot(l*sin(M(2)+2*err),-l*cos(M(2)+2*err),'b','linewidth',1)
  plot(l*sin(M(2)+err),-l*cos(M(2)+err),'b','linewidth',2)
  plot(l*sin(M(2)),-l*cos(M(2)),'b.','markersize',20)
end

% Draw stuff
plot(0,l,'k+','MarkerSize',20);
plot([xmin, xmax], [-height, -height],'k','linewidth',2)
plot(0,0,'k.','markersize',24)
plot(0,0,'y.','markersize',14)
plot(l*sin(theta),-l*cos(theta),'k.','markersize',24)
plot(l*sin(theta),-l*cos(theta),'y.','markersize',14)
plot(0,-2*l,'.w','markersize',0.005)

% Draw useful information 
plot([0 torque/umax*xmax],[-0.5, -0.5],'g','linewidth',10);
reward = 1-cost.fcn(cost,[0, theta]',zeros(2));
plot([0 reward*xmax],[-0.7, -0.7],'y', 'linewidth',10);
text(0,-0.5,'applied  torque')
text(0,-0.7,'immediate reward')
if nargin > 3
  text(0,-0.9, s)
end
if nargin > 4
  text(0,-1.1, s2)
end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-2*l 2*l]);
axis off;
drawnow;