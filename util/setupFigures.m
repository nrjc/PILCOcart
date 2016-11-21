% Script the setup and dock all 6 figures - used for a visual understanding of
% how the learning the dynamics model and learning the controller
% parameterisation is progressing                   
%
% (C) Copyright 2014-2015 by Andrew McHutchon, 2015-03-30

for i=1:6; figure(i); end
if usejava('awt');
  set(1,'windowstyle','docked','name','Policy training')
  set(2,'windowstyle','docked','name','Previous policy')
  set(3,'windowstyle','docked','name','Loss')
  set(4,'windowstyle','docked','name','States')
  set(5,'windowstyle','docked','name','Rollout')
  set(6,'windowstyle','docked','name','Dynamics training')
  desktop = com.mathworks.mde.desk.MLDesktop.getInstance;
  desktop.setDocumentArrangement('Figures',2,java.awt.Dimension(2,3));
  clear desktop;
end
