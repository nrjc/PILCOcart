function draw(plant, theta1, theta2, force, cost, s, s2, M, S)
% Draws the pendubot system
%
% (C) Copyright 2009-2011 by Carl Edward Rasmussen and Marc Deisenroth, 
% 2011-05-04. Edited by Joe Hall 2012-10-04.

l = 0.6;
xmin = -2*l; 
xmax = 2*l;    
umax = 2;
height = 0;

% Draw pendubot
clf; hold on
sth1 = sin(theta1); sth2 = sin(theta2);
cth1 = cos(theta1); cth2 = cos(theta2);
pendulum1 = [0, 0; -l*sth1, l*cth1];
pendulum2 = [-l*sth1, l*cth1; -l*(sth1-sth2), l*(cth1+cth2)];
plot(pendulum1(:,1), pendulum1(:,2),'r','linewidth',4)
plot(pendulum2(:,1), pendulum2(:,2),'r','linewidth',4)

% Draw stuff
plot(0,2*l,'k+','MarkerSize',20);
plot([xmin, xmax], [-height, -height],'k','linewidth',2)
plot(0,0,'k.','markersize',24)
plot(0,0,'y.','markersize',14)
plot(-l*sth1, l*cth1,'k.','markersize',24)
plot(-l*sth1, l*cth1,'y.','markersize',14)
plot(-l*(sth1-sth2), l*(cth1+cth2),'k.','markersize',24)
plot(-l*(sth1-sth2), l*(cth1+cth2),'y.','markersize',14)
plot(0,-2*l,'.w','markersize',0.005)

% Draw sample positions of the joints
if nargin > 7
  samples = gaussian(M,S+1e-8*eye(4),1000);
  t1 = samples(3,:); t2 = samples(4,:);
  plot(-l*sin(t1),l*cos(t1),'b.','markersize',2)
  plot(-l*(sin(t1)-sin(t2)),l*(cos(t1)+cos(t2)),'r.','markersize',2)
end

% Draw useful information
plot([0 force/umax*xmax],[-0.5, -0.5],'g','linewidth',10)
reward = 1-loss(cost, struct('m',[0, 0, theta1, theta2]'), plant);
plot([0 reward*xmax],[-0.7, -0.7],'y','linewidth',10)
text(0,-0.5,'applied  torque (1st joint)')
text(0,-0.7,'immediate reward')
if exist('s','var')  
  text(0,-0.9, s)
end
if exist('s2','var')
  text(0,-1.1, s2)
end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-2*l 2*l]);
axis off
drawnow;