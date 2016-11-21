function d = test_correctVar(m,S,dynmodel,plant,Cu,Csu,C,deriv)
% state is [Dold (=Di+Du); Do;Du;Du;Do]

delta = 1e-2;
D = length(m); Du = length(plant.maxU); Do = size(dynmodel.target,2);
Dold = D - 2*Do - 2*Du;
d = zeros(D);
[t dynmodel] = correctVar(m,S,dynmodel,plant,Cu,Csu,C); % prep dynmodel

switch deriv
    case 'dsdm'
        for i=Dold+1:D
            for j=Dold+1:D
                d(i,j) = checkgrad(@test_correctVar1,m,delta,S,dynmodel,plant,Cu,Csu,C,i,j);
                disp(['i = ' num2str(i) ', j = ' num2str(j) ': d = ' num2str(d(i,j))]);
            end
        end
        
    case 'dsds'
        for i=Dold+1:D
            for j=Dold+1:D
                d(i,j) = checkgrad(@test_correctVar2,S,delta,m,S,dynmodel,plant,Cu,Csu,C,i,j);
                disp(['i = ' num2str(i) ', j = ' num2str(j) ': d = ' num2str(d(i,j))]);
            end
        end
        
    otherwise
        fprintf('Unrecognised derivative, options are: ''dsdm'' and ''dsds''\n');
end
d = d(Dold+1:end,Dold+1:end);

function [f df] = test_correctVar1(m,S,dynmodel,plant,Cu,Csu,C,i,j)
% dsdm
[S dynmodel, sdm] = correctVar(m,S,dynmodel,plant,Cu,Csu,C);

f = S(i,j);
df = squeeze(sdm(i,j,:));

function [f df] = test_correctVar2(SS,m,S,dynmodel,plant,Cu,Csu,C,i,j)
% dsds
dx = SS-S; dx = dx + dx' - diag(diag(dx)); S = S + dx;
[S dynmodel, sdm, sds] = correctVar(m,S,dynmodel,plant,Cu,Csu,C);

f = S(i,j);
df = squeeze(sds(i,j,:,:));
df = df + df' - diag(diag(df));