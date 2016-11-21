function draw(s, u, cost, bm, bv, str1, str2)
% Draw the cart-pole system with reward, applied force, and predictive
% uncertainty of the tip of the pendulum
%
% s      struct     state struct
% u      U x 1      control action(s)
% cost   function   cost function
% bm     D x 1      uncertainty mean of state
% bv     D x D      uncertainty variance of state
%
% (C) Copyright 2009-2011 Carl Edward Rasmussen and Marc Deisenroth
% 2009-05-25. Edited by Joe Hall 2012-10-02. Edited Rowan 2015-05-26

l = cost.ell; % length of pendulum
xmin = -3;
xmax = 3;
height = 0.1;
width  = 0.3;

% Compute positions
x = s.m(end-3);
theta = s.m(end-2);
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
plot([0 u/10*xmax],[-0.3, -0.3],'g','linewidth',10)
computed_reward = true;
try
  reward = 1-cost.fcn(s).m;
catch
  computed_reward = false;
end
if computed_reward; plot([0 reward*xmax],[-0.6, -0.6],'y','linewidth',10); end

% Plot the cart
fill(cart(:,1), cart(:,2),'b','edgecolor','b');
plot(pendulum(:,1), pendulum(:,2),'r','linewidth',4)
plot(x,0,'k.','markersize',24)
plot(x,0,'y.','markersize',14)
plot(pendulum(2,1),pendulum(2,2),'k.','markersize',24)
plot(pendulum(2,1),pendulum(2,2),'y.','markersize',14)

% Error ellipses
try
  [M1,S1] = getPlotDistr(bm, bv, 2*l);
  error_ellipse(S1, M1, 'style', 'b');
catch
end

% Text
text(0,-0.3,'applied force')
if computed_reward; text(0,-0.6,'immediate reward'); end
if exist('str1','var'); text(0,-0.9, str1); end
if exist('str2','var'); text(0,-1.2, str2); end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-1.4 1.4]);
axis off;
drawnow;

function [M, S] = getPlotDistr(m,s,ell)         % helper function: find ellipse
angi = length(m)-2;
[m, s] = gTrigN(m,s,angi,ell);                     % augment input distribution
ix = length(m)-5; isin = length(m)-1; icos = length(m);
M = [m(ix)-m(isin); m(icos)];                                            % mean
s11 = s(ix,ix) + s(isin,isin) + s(ix,isin) + s(isin,ix);       % x+l sin(theta)
s22 = s(icos,icos);                                             % -l*cos(theta)
s12 = -(s(ix,icos)+s(isin,icos));                 % cov(x+l*sin(th), -l*cos(th)
S = [s11 s12; s12' s22];                           % assemble covariance matrix
try chol(S); catch; disp('matrix S not pos.def. (getPlotDistr)'); end   % check
