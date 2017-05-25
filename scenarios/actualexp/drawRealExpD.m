% nrollouts=20;
% %j=1; J=0;
% if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end
% clf(3); errorbar(0:H, [pred(j).cost.m], 2*sqrt([pred(j).cost.s])); hold on;
% for myj=0:nrollouts
% 	if myj==0
% 		myclr='g';
% 	else
% 		myclr='r';
% 	end
% 	L = zeros(1, size(data(j+J+myj).state,1));
% 	for i=1:length(L)
%     	L(i)=cost.fcn(cost,struct('m',data(j+J+myj).state(i,plant.dyno)'));
% 	end
% 	plot(0:length(L)-1,L,myclr); drawnow; 
% end
% axis tight
close all;
figure(2);
dyno = plant.dyno; nsp = length(dyno) + ctrl.U;
dynoS = dyno - sum(setdiff(plant.odei,dyno)<=D); % shifted by # of skipped elements
stateM=[pred(j).state(:).m]; stateS=cat(3,pred(j).state(:).s);
for i=1:length(dyno)          % plot the rollouts on top of predicted errorbars
  subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), i); hold on;
  xlabel('Time Step')
  ylabel(varUnits{i});
  title(varNames{i});
  errorbar(0:length(stateM(dynoS(i),:))-1, stateM(dynoS(i),:), ...
                                   2*sqrt(squeeze(stateS(dynoS(i),dynoS(i),:))),'b');  
  plot(0:length(data(j+J-1).state(:,dynoS(i)))-1, data(j+J-1).state(:,dynoS(i)), 'g');
%   for myj=1:nrollouts
%   	plot(0:length(data(j+J+myj).state(:,dynoS(i)))-1, data(j+J+myj).state(:,dynoS(i)), 'r');
%   end
  axis tight
end
subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), length(dyno)+1);
hold on;
xlabel('Time Step')
ylabel('Action')
title('u')
errorbar(0:length([pred(j).action.m])-1, [pred(j).action.m],2*sqrt([pred(j).action.s]));
% plot(0:length(data(j+J).action)-1, data(j+J).action, 'g');
% for myj=1:nrollouts
% 	plot(0:length(data(j+J+myj).action)-1, data(j+J+myj).action, 'r');
% end
axis tight
%disptable(excxp([dyn.on; dyn.pn; [dyn.hyp.n]']), varNames, ...
%            ['observation noise|process noise std|inducing targets'], '%0.5f');

% ydyn = cell2mat(arrayfun(@(Y)Y.state(2:end, plant.dyno),data,'uniformoutput',0)');
% ysignal = std(ydyn(2:end,:));
% ysnr =  ysignal ./ exp(dyn.on)


% firststates = cell2mat(arrayfun(@(Y)Y.state(1, :),data,'uniformoutput',0)');
% 
% inputNames = ['ooou|oox|ootheta1|ootheta2|oou|ox|otheta1|otheta2|ou|x|theta1|theta2|u|sin ootheta1|cos ootheta1|sin ootheta2|cos ootheta2|sin otheta1|cos otheta1|sin otheta2|cos otheta2|sin theta1|cos theta1|sin theta2|cos theta2']
% disptable(exp([dyn.hyp.l]), varNames, inputNames, '%0.5f')
% 
% disptable([dyn.hyp.m], varNames, inputNames, '%0.5f')

%figure
%errorbar(0:H, [pred(j).cost.m], 2*sqrt([pred(j).cost.s])); hold on;
%errorbar(0:H, [pred(j-1).cost.m], 2*sqrt([pred(j-1).cost.s]), 'r');
%errorbar(0:H, [pred(j-2).cost.m], 2*sqrt([pred(j-2).cost.s]), 'g');