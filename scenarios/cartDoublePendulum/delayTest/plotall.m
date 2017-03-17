%Plot Stuff
%Plot predicted costs
[pred(j).state,pred(j).action,pred(j).cost] = simulate(s, dyn, ctrl, cost, H, false);
if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end
clf(3); errorbar(0:H, [pred(j).cost.m], 2*sqrt([pred(j).cost.s]));
axis tight; grid; drawnow;
hold on;
for ii=1:Nroll; plot(0:length(rC{ii})-1,rC{ii},'r'); end % plot the real losses
plot(0:length(realCost{J+j})-1,realCost{J+j},'g'); drawnow; % repeat green line

if ~ishandle(4); figure(4); else set(0,'CurrentFigure',4); end; clf(4);
nsp = E + U;
dynoS = dyno - sum(setdiff(plant.odei,dyno)<=D); % shifted by # of skipped elements
stateM=[pred(j).state(:).m]; stateS=cat(3,pred(j).state(:).s); %#ok<*IJCL>

for i = 1:E          % plot the rollouts on top of predicted errorbars
  subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), i); hold on;
  if exist('varNames','var'); xlabel(varNames{i}); end
  if exist('varUnits','var'); ylabel(varUnits{i}); end
  errorbar(0:length(stateM(dynoS(i),:))-1, stateM(dynoS(i),:), ...
                                 2*sqrt(squeeze(stateS(dynoS(i),dynoS(i),:))));  
  
  for ii=1:Nroll
    plot(0:size(lat{ii}.state(:,dyno(i)),1)-1, lat{ii}.state(:,dyno(i)), 'r');
  end
  plot(0:size(latent(j+J).state(:,dyno(i)),1)-1, ...
    latent(j+J).state(:,dyno(i)),'g','LineWidth',3);
                                 
  
  axis tight
end

actionM=[pred(j).action(:).m]; actionS=cat(3,pred(j).action(:).s); %#ok<*IJCL>
for i = 1:U
  subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), E + i);
  hold on;
  errorbar(0:length(actionM(i,:))-1, actionM(i,:), ...
                                              2*sqrt(squeeze(actionS(i,i,:)))); 
  
  for ii=1:Nroll
    plot(0:length(lat2{ii}.action(:,i))-1, lat2{ii}.action(:,i), 'r');
  end
  plot(0:length(data(j+J).action(:,i))-1, data(j+J).action(:,i), 'g','LineWidth',3);
  
  axis tight
  xlabel('u'); 
  ylabel('N'); 
end
drawnow;
figure;
costs = [];
for i=1:(j+1)
    costs = [costs sum(realCost{i})];
end
plot(1:(j+1),costs)
