% Display and store in a text file the key settings for a simulation
% Hit any key to continue after this script has run
out = sprintf('%s\n\n', date);

% Plant
out = sprintf('%sPLANT\node:\t\t%s\nnoise:\t\t%s\nctrltype:\t%s\n', out,...
              func2str(plant.ode), num2str(sqrt(diag(plant.noise)')), func2str(plant.ctrltype));

if strcmp(func2str(plant.ode), 'dynamics_stiffness')
    out = sprintf('%sStiffness:\t%s\n', out, num2str(odeA));
else
    out = sprintf('%sStiffness:\tOFF\n', out);
end
out = sprintf('%sField:\t\t%s\n', out, num2str(odeFIELD));
if odeFIELD == 2
    out = sprintf('%sF2: \t\t%s\n', out, num2str(odeF2));
elseif odeFIELD == 3
    out = sprintf('%sF3: \t\t%s\n', out, num2str(odeF3));
end
out = sprintf('%sdelay\t\t%s\n', out, num2str(plant.delay));
out = sprintf('%s\n', out);


% Policy
out = sprintf('%sPOLICY\nfcn:\t\t%s\n', out, func2str(ctrl.policy.fcn));
if isfield(ctrl.policy, 'sub')
    for i = 1:length(ctrl.policy.sub)
        out = sprintf('%ssub{%d}:\t\t\t%s\n', out, i, func2str(ctrl.policy.sub{i}.fcn));
    end
end

if strcmpi(ctrl.policy.addCtrlNoise, 'on')
    ctrlNoiseStatus = 'ON';
    out = sprintf('%sctrlNoise?:\t%s\n', out, ctrlNoiseStatus);
    out = sprintf('%sLevel:\t\t%s\n\n', out, num2str(ctrl.policy.ctrlNoise));
else
    ctrlNoiseStatus = 'OFF';
    out = sprintf('%sctrlNoise?:\t%s\n', out, ctrlNoiseStatus);
end


% Cost
out = sprintf('%sCOST\nfcn:\t\t%s\nW:  \t\t%s', out, func2str(cost.fcn), num2str(diag(cost.W)'));

disp(out)
fid = fopen('keySettings.txt', 'w');

disp('Press any key to continue...')
pause
fprintf(fid, '%s', out);
fclose(fid);
disp('Continuing...')