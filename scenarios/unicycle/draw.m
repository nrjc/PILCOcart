function draw(s, force, cost, ~, ~, str1, str2)
% function [allW,W] = draw(latent,plant,t2,cost,f)

% Carl Edward Rasmussen, 2012-03-27
% Modified Jonas Umlauft 2014-05-26
% Modified Rowan MCAllister 2015-07-10

cf = gcf;
clf
set(gca,'FontSize',16); set(0,'CurrentFigure',cf); % set gca alters cf

rw =  0.225;  % wheel radius
rf =  0.54;   % frame center of mass to wheel
rt =  0.27;   % frame centre of mass to turntable
rr =  rf+rt;  % distance wheel to turntable

M = 24; MM = 2*pi*(0:M)/M;
RR = ['r-';'r-';'r-';'k-';'b-';'b-';'b-'];

clear W;
theta =  s.m( 8);
phi   =  s.m( 9);
psif  =  s.m(10);
x     =  s.m(15);
y     =  s.m(16);
psiw  = -s.m(17);
psit  =  s.m(18);

A = [  cos(phi)             sin(phi)              0
  -sin(phi)*cos(theta)  cos(phi)*cos(theta)  -sin(theta)
  -sin(phi)*sin(theta)  cos(phi)*sin(theta)   cos(theta) ]';

r = rw*[cos(psiw+MM); zeros(1,M+1); sin(psiw+MM)+1];
R{1} = bsxfun(@plus,A*r,[x; y; 0]);
r = rw*[cos(psiw) -cos(psiw); 0 0; sin(psiw)+1 -sin(psiw)+1];
R{2} = bsxfun(@plus,A*r,[x; y; 0]);
r = rw*[sin(psiw) -sin(psiw); 0 0; -cos(psiw)+1 cos(psiw)+1];
R{3} = bsxfun(@plus,A*r,[x; y; 0]);
r = [0 rr*sin(psif); 0 0; rw rw+rr*cos(psif)];
R{4} = bsxfun(@plus,A*r,[x; y; 0]);
r = [rr*sin(psif)+rw*cos(psif)*cos(psit+MM); rw*sin(psit+MM); rw+rr*cos(psif)-rw*sin(psif)*cos(psit+MM)];
R{5} = bsxfun(@plus,A*r,[x; y; 0]);
r = [rr*sin(psif)+rw*cos(psif)*cos(psit) rr*sin(psif)-rw* ...
  cos(psif)*cos(psit); rw*sin(psit) -rw*sin(psit); rw+rr* ...
  cos(psif)-rw*sin(psif)*cos(psit) rw+rr*cos(psif)+rw* ...
  sin(psif)*cos(psit)];
R{6} = bsxfun(@plus,A*r,[x; y; 0]);
r = [rr*sin(psif)+rw*cos(psif)*sin(psit) rr*sin(psif)-rw* ...
  cos(psif)*sin(psit); -rw*cos(psit) rw*cos(psit); rw+rr* ...
  cos(psif)-rw*sin(psif)*sin(psit) rw+rr*cos(psif)+rw* ...
  sin(psif)*sin(psit)];
R{7} = bsxfun(@plus,A*r,[x; y; 0]);
hold off
aa = linspace(0,2*pi,201); plot3(2*sin(aa),2*cos(aa),0*aa,'k:','LineWidth',2);
hold on

r = A*[0; 0; rw] + [x; y; 0];
P = [r R{1}(:,1:M/4+1) r]; fill3(P(1,:),P(2,:),P(3,:),'r','EdgeColor','none');
P = [r R{1}(:,M/2+1:3*M/4+1) r]; fill3(P(1,:),P(2,:),P(3,:),'r','EdgeColor','none');

r = A*[rr*sin(psif); 0; rw+rr*cos(psif) ] + [x; y; 0];
P = [r R{5}(:,1:M/4+1) r]; fill3(P(1,:),P(2,:),P(3,:),'b','EdgeColor','none');
P = [r R{5}(:,M/2+1:3*M/4+1) r]; fill3(P(1,:),P(2,:),P(3,:),'b','EdgeColor','none');

for j = [1 4 5];
  plot3(R{j}(1,:),R{j}(2,:),R{j}(3,:),RR(j,:),'LineWidth',2)
end
axis equal; axis([-2 2 -2 2 0 1.5]); set(0,'CurrentFigure',cf); %axis alters cf
xlabel 'x [m]';
ylabel 'y [m]';
grid on

% draw controls:
ut = force(1);
uw = force(2);
L = cost.fcn(s).m;

utM = 10;
uwM = 50;

set(gca, 'Clipping', 'off');
oo = [4 -3.07 0]/6.4; o1 = [-0.5 2 2.0]; o2 = [-0.5 2 1.6]; o3 = [-0.5 2 1.2];
o0 = 1.5*ut/utM;
plot3([o1(1) o1(1)+o0*oo(1)],[o1(2) o1(2)+o0*oo(2)],[o1(3) o1(3)+o0*oo(3)],'b','LineWidth',5)
plot3([o1(1)-1.5*oo(1) o1(1)+1.5*oo(1) o1(1)+1.5*oo(1) o1(1)-1.5*oo(1) o1(1)-1.5*oo(1)],...
  [o1(2)-1.5*oo(2) o1(2)+1.5*oo(2) o1(2)+1.5*oo(2) o1(2)-1.5*oo(2) o1(2)-1.5*oo(2)],...
  [o1(3)+0.04 o1(3)+0.04 o1(3)-0.04 o1(3)-0.04 o1(3)+0.04], 'b');
plot3([-0.5 -0.5],[2 2],o1(3)+[-0.06 0.06],'b');
o0 = 1.5*uw/uwM;
plot3([o2(1) o2(1)+o0*oo(1)],[o2(2) o2(2)+o0*oo(2)],[o2(3) o2(3)+o0*oo(3)],'r','LineWidth',5)
plot3([o2(1)-1.5*oo(1) o2(1)+1.5*oo(1) o2(1)+1.5*oo(1) o2(1)-1.5*oo(1) o2(1)-1.5*oo(1)],...
  [o2(2)-1.5*oo(2) o2(2)+1.5*oo(2) o2(2)+1.5*oo(2) o2(2)-1.5*oo(2) o2(2)-1.5*oo(2)],...
  [o2(3)+0.04 o2(3)+0.04 o2(3)-0.04 o2(3)-0.04 o2(3)+0.04], 'r');
plot3([-0.5 -0.5],[2 2],o2(3)+[-0.06 0.06],'r');

o0 = 3*L-1.5;
plot3([o3(1)-1.5*oo(1) o3(1)+o0*oo(1)],[o3(2)-1.5*oo(2) o3(2)+o0*oo(2)],[o3(3)-1.5*oo(3) o3(3)+o0*oo(3)],'k','LineWidth',5)
plot3([o3(1)-1.5*oo(1) o3(1)+1.5*oo(1) o3(1)+1.5*oo(1) o3(1)-1.5*oo(1) o3(1)-1.5*oo(1)],...
  [o3(2)-1.5*oo(2) o3(2)+1.5*oo(2) o3(2)+1.5*oo(2) o3(2)-1.5*oo(2) o3(2)-1.5*oo(2)],...
  [o3(3)+0.04 o3(3)+0.04 o3(3)-0.04 o3(3)-0.04 o3(3)+0.04], 'k');

FontSize = 14;
text2(-0.5-1.5*oo(1), 2-1.5*oo(2), 2.2,['Disc torque    max \pm ',num2str(utM),' Nm'],'Color','b','FontSize',FontSize);
text2(-0.5-1.5*oo(1), 2-1.5*oo(2), 1.8,['Wheel torque max \pm ',num2str(uwM),' Nm'],'Color','r','FontSize',FontSize);
text2(-0.5-1.5*oo(1), 2-1.5*oo(2), 1.4,'Instantaneous Cost','Color','k','FontSize',FontSize);
text2(2,1,1.8,str1,'FontSize',FontSize);
text2(2,1,1.4,str2,'FontSize',FontSize);

%  plot3([0 1.5*ut/utM],[2 2],[1.4 1.4],'k-','LineWidth',5);
%  plot3([0 1.5*uw/uwM],[2 2],[1.2 1.2],'r-','LineWidth',5);
%
%if jj>J
%  exx = 0; for exxx = 1:jj-1; exx = exx + size(latent{exxx},1)-1; end;
%  text(2,1,1.8,['Control trial #', int2str(jj-J)],'FontSize',16);
%  text(2,1,1.4,['Experience: ',  num2str(exx*t1,'%2.1f'), ' s'],'FontSize',16);
%else
%  text(2,1,1.8,['Random trial #', int2str(jj)], 'FontSize', 16);
%end

%bar3(1,ut,0.1,'k');
%bar3(0.8,uw,0.1,'r');
%  pt = patch([-2 -2 -1.9 -1.9 -2 -2 -1.9 -1.9],[1 0.9 0.9 1 1 0.9 0.9 1],[1.5 1.5 1.5 1.5 1.5+ut*1.5/utM 1.5+ut*1.5/utM 1.5+ut*1.5/utM 1.5+ut*1.5/utM],[.2 .2 .2 .2 .2 .2 .2 .2]);
%  set(pt,'FaceColor','k');
%  pw = patch([-1.5 -1.5 -1.4 -1.4 -1.5 -1.5 -1.4 -1.4],[1 0.9 0.9 1 1 0.9 0.9 1],[1.5 1.5 1.5 1.5 1.5+uw*1.5/uwM 1.5+uw*1.5/uwM 1.5+uw*1.5/uwM 1.5+uw*1.5/uwM],[1 1 1 1 1 1 1 1]);
%  set(pw,'FaceColor','r');

drawnow
% if nargout > 0
%   W(i) = getframe;
% end
% if nargin == 5
%   if i==1 || i == size(q,1), iM = 8; else iM = 1; end
%   for iii=1:iM
%     eval(['print -dpng ', f, int2str(ii), '.png;']);
%     ii = ii + 1;
%   end
% end
% pause(0.05);
%
% if nargout > 1
%   allW{jj} = W;
% end
%
% ffmpeg -b 2M -i tmp1%04d.png tmp.mp4

% assert(gcf == 5); % debugging
