function draw(s, force, cost, M, S, str1, str2)
% Carl Edward Rasmussen and Marc Deisenroth, Jonas Umlauft 2015-04-27

scale = 1.2;
l = 0.3*scale;
xmin = -3*scale; 
xmax = 3*scale;    
height = 0.07*scale;
width  = 0.25*scale;

font_size = 12;

% Compute positions 
x = s.m(end-2); theta2 = s.m(end-1); theta3 = s.m(end);
% x = s.m(end-5); theta2 = s.m(end-4); theta3 = s.m(end-3);
cart = [ x + width,  height
         x + width, -height
         x - width, -height
         x - width,  height
         x + width,  height ];
pendulum2 = [x, 0; x-2*l*sin(theta2), cos(theta2)*2*l];
pendulum3 = [x-2*l*sin(theta2), 2*l*cos(theta2); x-2*l*sin(theta2)-2*l*sin(theta3), 2*l*cos(theta2)+2*l*cos(theta3)];
clf
hold on

plot(0,4*l,'k+','MarkerSize',1.5*font_size,'linewidth',2); % Target
plot([xmin, xmax], [-height-0.03*scale, -height-0.03*scale], 'Color','b','LineWidth',3);  % Baseline


plot(cart(:,1), cart(:,2),'Color','k','LineWidth',3);           % Draw Cart
fill(cart(:,1), cart(:,2),'k');
plot(pendulum2(:,1), pendulum2(:,2),'Color','r','LineWidth', round(font_size/2)); % Draw Pendulum2;
plot(pendulum3(:,1), pendulum3(:,2),'Color','r','LineWidth', round(font_size/2)); % Draw Pendulum3
plot(x,0,'o','MarkerSize', round((font_size+4)/2),'Color','y','markerface','y'); % joint at cart
plot(pendulum3(1,1),pendulum3(1,2),'o','MarkerSize', round((font_size+4)/2),'Color','y','markerface','y'); % 2nd joint 
plot(pendulum3(2,1),pendulum3(2,2),'o','MarkerSize', round((font_size+4)/2),'Color','y','markerface','y'); % tip of 2nd joint 

% Force bar
plot([0 force/20*xmax]-0.5,[-1, -1].*scale, 'Color', 'g', 'LineWidth', font_size);
text2(-0.5,-1.0*scale,'applied  force','fontsize', font_size) 

% Cost bar
computed_reward = true;
try
  reward = 1-cost.fcn(s).m;
catch
  computed_reward = false;
end
if computed_reward
  plot([0 reward*xmax]-0.5,[-1.3, -1.3].*scale, 'Color', 'y', 'LineWidth', font_size);
  text2(-0.5,-1.3*scale,'immediate reward','fontsize', font_size);
end

% Predicted distribution
if nargin > 8 && max(max(S)) > 0
  [M1, S1, M2, S2] = getPlotDistr(M, S, 2*l, 2*l);
  error_ellipse(S1, M1, 'style', 'b');
  error_ellipse(S2, M2, 'style', 'r');
end
   
% Info strings   
if nargin > 5 && ~isempty(str1)
  text2(-0.5,-1.6*scale, str1,'fontsize', font_size)
end
if nargin > 6 && ~isempty(str2)
  text2(-0.5,-1.9*scale, str2,'fontsize', font_size)
end

set(gca,'DataAspectRatio',[1 1 1],'XLim',[xmin xmax],'YLim',[-1.4 1.4]*scale);
axis off;
drawnow;

function [M1, S1, M2, S2] = getPlotDistr(m, s, ell1, ell2)       % find ellipse
[m1, s1] = trigaug(m, s, [2 3], [ell1, ell2]);     % augment input distribution
M1 = [m1(1) - m1(8); m1(9)];        % E[x -l\sin\theta_2]; E[l\cos\theta_2]; p2
M2 = [M1(1) - m1(10); M1(2) + m1(11)];               % p3: mean of cart. coord.
S1(1,1) = s1(1,1) + s1(8,8) - 2*s1(1,8);                     % first covariance
S1(2,2) = s1(9,9); 
S1(1,2) = s1(1,9) - s1(8,9);
S1(2,1) = S1(1,2)';
try chol(S1); catch disp('S1 not +def (getPlotDistr)'); end             % check
S2(1,1) = S1(1,1) + s1(10,10) - 2*(s1(1,10) - s1(8,10));    % second covariance
S2(2,2) = s1(9,9) + s1(11,11) + 2*s1(9,11);
S2(1,2) = s1(1,9) - s1(8,9) - s1(10,9) + s1(1,11) - s1(8,11) - s1(10,11);
S2(2,1) = S2(1,2)';
try chol(S2); catch disp('S2 not +def (getPlotDistr)'); end             % check
