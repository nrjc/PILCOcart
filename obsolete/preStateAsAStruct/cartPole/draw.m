function draw(x, theta, force, cost, s, s2, M, S)
% Draw the cart-pole system with reward, applied force, and predictive
% uncertainty of the tip of the pendulum
%
% (C) Copyright 2009-2011 Carl Edward Rasmussen and Marc Deisenroth 
% 2009-05-25. Edited by Joe Hall 2012-10-02.

l = 0.6;
xmin = -3; 
xmax = 3;    
height = 0.1;
width  = 0.3;

% Compute positions 
cart = [ x + width,  height
         x + width, -height
         x - width, -height
         x - width,  height
         x + width,  height ];
pendulum = [x, 0; x-2*l*sin(theta), cos(theta)*2*l];

% Plot scale stuff
clf; hold on
plot(0,2*l,'k+','MarkerSize',20,'linewidth',2)
plot([xmin, xmax], [-height-0.03, -height-0.03],'k','linewidth',2)
plot([0 force/10*xmax],[-0.3, -0.3],'g','linewidth',10)
reward = 1-cost.fcn(cost,[x, 0, 0, 0]', zeros(4));
plot([0 reward*xmax],[-0.5, -0.5],'y','linewidth',10)

% Plot the cart
fill(cart(:,1), cart(:,2),'b','edgecolor','b');
plot(pendulum(:,1), pendulum(:,2),'r','linewidth',4)
plot(x,0,'k.','markersize',24)
plot(x,0,'y.','markersize',14)
plot(pendulum(2,1),pendulum(2,2),'k.','markersize',24)
plot(pendulum(2,1),pendulum(2,2),'y.','markersize',14)

% Error ellipses
try
  [M1, S1] = getPlotDistr(M,S,2*l);
  error_ellipse(S1,M1,'style','b');
catch
end

% Text
text(0,-0.3,'applied force')
text(0,-0.6,'immediate reward')
if exist('s','var')
  text(0,-0.9, s)
end
if exist('s2','var')
  text(0,-1.1, s2)
end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-1.4 1.4]);
axis off;
drawnow;