%j=1; J=0;
if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end
clf(3); errorbar(0:H, [pred(j).cost.m], 2*sqrt([pred(j).cost.s])); hold on;
L = zeros(1, size(data(j+J).state,1));
for i=1:length(L)
    L(i)=cost.fcn(cost,struct('m',data(j+J).state(i,plant.dyno)'));
end
plot(0:length(L)-1,L,'g'); drawnow; 
axis tight

figure(2);
dyno = plant.dyno; nsp = length(dyno) + ctrl.U;
dynoS = dyno - sum(setdiff(plant.odei,dyno)<=D); % shifted by # of skipped elements
stateM=[pred(j).state(:).m]; stateS=cat(3,pred(j).state(:).s);
for i=1:length(dyno)          % plot the rollouts on top of predicted errorbars
  subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), i); hold on;
  errorbar(0:length(stateM(dynoS(i),:))-1, stateM(dynoS(i),:), ...
                                   2*sqrt(squeeze(stateS(dynoS(i),dynoS(i),:))));  
  plot(0:length(data(j+J).state(:,dynoS(i)))-1, data(j+J).state(:,dynoS(i)), 'g');
  axis tight
end
subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), length(dyno)+1);
hold on;
errorbar(0:length([pred(j).action.m])-1, [pred(j).action.m],2*sqrt([pred(j).action.s]));
plot(0:length(data(j+J).action)-1, data(j+J).action, 'g');
axis tight

disptable(exp([dyn.on; dyn.pn; dyn.hyp.n]), varNames, ...
            ['observation noise|process noise std|inducing targets'], '%0.5f');

ydyn = cell2mat(arrayfun(@(Y)Y.state(2:end, plant.dyno),data,'uniformoutput',0)');
ysignal = std(ydyn(2:end,:));
ysnr =  ysignal ./ exp(dyn.on)


firststates = cell2mat(arrayfun(@(Y)Y.state(1, :),data,'uniformoutput',0)');

inputNames = ['ooou|oox|ootheta1|oou|ox|otheta1|ou|x|theta1|u|sin ootheta1|cos ootheta1|sin otheta1|cos otheta1|sin theta1|cos theta1|']
disptable(exp([dyn.hyp.l]), varNames, inputNames, '%0.5f')

disptable([dyn.hyp.m], varNames, inputNames, '%0.5f')