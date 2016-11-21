% Script to apply the controller
%
% (C) Copyright 2010-2014 by Carl Edward Rasmussen, Marc Deisenroth and
%                                 Andrew McHutchon, Rowan McAllister 2016-03-09
%
%TODO: Realcost is not correct. 

% 1. Do rollout
if isfield(plant,'constraint') && exist('maxH','var'); HH=maxH; else HH=H; end
currT = 1;
[data(J+j), latent(j+J), realCost{J+j}] = ...
  rollout(gaussian(mu0, S0), ctrl, HH, plant, cost, 1); 
disp([data(J+j).state [data(J+j).action; zeros(1,ctrl.U)]]);
if ~ishandle(3); figure(3); else set(0,'CurrentFigure',3); end; hold on
plot(0:length(realCost{J+j})-1, realCost{J+j},'g'); drawnow;  % new trial added

% 2. Make many rollouts to test dynamics model
Nroll = 10; lat = cell(1,Nroll); rC = cell(1,Nroll);
for i=1:Nroll
  currT = 1;
  [lat2{i},lat{i},rC{i}] = rollout(gaussian(mu0, S0), ctrl, HH, plant, cost, 0);
end

% 3. Plot rollouts and training predictions for comparison
for ii=1:Nroll; plot(0:length(rC{ii})-1,rC{ii},'r'); end % plot the real losses
plot(0:length(realCost{J+j})-1,realCost{J+j},'g'); drawnow; % repeat green line

if ~ishandle(4); figure(4); else set(0,'CurrentFigure',4); end; clf(4);
nsp = E + U;
dynoS = dyno - sum(setdiff(plant.odei,dyno)<=D); % shifted by # of skipped elements
stateM=[pred(j).state(:).m]; stateS=cat(3,pred(j).state(:).s); %#ok<*IJCL>
for i = 1:E          % plot the rollouts on top of predicted errorbars
  subplot(floor(sqrt(nsp)), ceil(nsp/floor(sqrt(nsp))), i); hold on;
  %if exist('varNames','var'); xlabel(varNames{i}); end
  %if exist('varUnits','var'); ylabel(varUnits{i}); end
  errorbar(0:length(stateM(dynoS(i),:))-1, stateM(dynoS(i),:), ...
                                 2*sqrt(squeeze(stateS(dynoS(i),dynoS(i),:))));  
  for ii=1:Nroll
    plot(0:size(lat{ii}.state(:,dyno(i)),1)-1, lat{ii}.state(:,dyno(i)), 'r');
  end
  plot(0:size(latent(j+J).state(:,dyno(i)),1)-1, ...
    latent(j+J).state(:,dyno(i)),'g');
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
  plot(0:length(data(j+J).action(:,i))-1, data(j+J).action(:,i), 'g');
  axis tight
end
drawnow;

% 4. Save data
filename = [basename num2str(j) '_H' num2str(H)]; save(filename);
