function[S dynmodel, sdm sds] = correctVar(m,S,dynmodel,plant,Cu,Csu,C)
% m     Dold + Do + Du + Du + Do            
% N.B. This won't work with trig auged angles yet

poli = plant.poli; dyni = plant.dyni; angi = plant.angi; Da = length(angi);
Di = length(dyni); Do = length(plant.dyno); Du = length(plant.maxU); D4 = length(m);
Dold = length(m) - 2*Do - 2*Du; D1 = Dold + Do + 2*Da; D2 = D1 + Du; D3 = D2 + Du;

if nargout > 2
    Cudm = Cu{2}; Cuds = Cu{3}; Cu = Cu{1};
    Csudm = Csu{2}; Csuds = Csu{3}; Csu = Csu{1};
    Cdm = C{2}; Cds = C{3}; C = C{1};
end

% Compute conditional term
if 0 == Dold || (isfield(plant,'correct') && 0 == plant.correct)
    
    Sc = 0; Scdm = zeros(Do,Do,D4); Scds = zeros(Do,Do,D4,D4);
    Sio = 0; Siodm = zeros(Do+Du,Do,D4); Siods =  zeros(Do+Du,Do,D4,D4);

else
    
    i = [1:Dold Dold+dyni D2+1:D3]; % dyni & controls for both old and current states
    Czdm = zeros(Do,Do,D4); Czds = zeros(Do,Do,D4,D4);
    
    if nargout > 2
        [Cz dynmodel Czdm(:,:,i) Czds(:,:,i,i)] = covFluct(dynmodel,m(i),S(i,i)); % cov(z*, z**)
    else
        [Cz dynmodel] = covFluct(dynmodel,m(i),S(i,i)); % cov(z*, z**)
    end
    
    ba = zeros(Do,Do); a = zeros(Do,Du);
    ba(dyni,:) = C(1:Di,:);       % ba: x** -> y**, Do-by-Do, only dyni non-zero
    bb = C(Di+1:end,:);                              % bb: su** -> y**, Du-by-Do
    a(poli,:) = Cu;                % a: x** -> u**, Do-by-Du, only poli non-zero
    g = Csu(end-Du+1:end,:);                          % g: u** -> su**, Du-by-Du
    
    c = ba' + bb'*g'*a';  % y** <- x** + (y** <- su**)(su** <- u**)(u** <- x**)
    Sc = c*Cz;            
    Sc = Sc + Sc';                           % Additional output covariance, y**
    
    %Sio = [Cz(dyni,:); g'*a'*Cz];      % Additional I/O covariance, x** with y**
    Sio = [Cz; g'*a'*Cz];
    if ~real(Sc); fprintf('Sc not real\n'); keyboard; end
    
    if nargout > 2
     badm = zeros(Do,Do,D4); bads = zeros(Do,Do,D4,D4);
     bbdm = zeros(Du,Do,D4); bbds = zeros(Du,Do,D4,D4);
     adm = zeros(Do,Du,D4); ads = zeros(Do,Du,D4,D4);
     gdm = zeros(Du,Du,D4); gds = zeros(Du,Du,D4,D4);
     
     i = [Dold+dyni D2+1:D3];       % C depends on current state dyni & controls
     badm(dyni,:,i) = Cdm(1:Di,:,:);                            % Do-by-Do-by-D4
     bads(dyni,:,i,i) = Cds(1:Di,:,:,:);                  % Du-by-Do-by-D4-by-D4
     bbdm(:,:,i) = Cdm(Di+1:end,:,:);                           % Du-by-Do-by-D4
     bbds(:,:,i,i) = Cds(Di+1:end,:,:,:);                 % Du-by-Do-by-D4-by-D4
     adm(poli,:,Dold+poli) = Cudm;         % poli -> unsquashed u Do-by-Du-by-D4
     ads(poli,:,Dold+poli,Dold+poli) = Cuds;              % Do-by-Du-by-D4-by-D4
     gdm(:,:,1:D2) = Csudm(D1+1:D2,:,:);               % u -> su, Du-by-Du-by-D4
     gds(:,:,1:D2,1:D2) = Csuds(D1+1:D2,:,:,:);           % Du-by-Du-by-D4-by-D4
    
     % Derivative of g'*a'
     gadm = etprod('123',g,'41',adm,'243') + etprod('123',gdm,'413',a,'24');
     gads = etprod('1234',g,'51',ads,'2534') + etprod('1234',gds,'5134',a,'25');

     % Derivative of bb'*(g'*a')
     bgadm = etprod('123',bb,'41',gadm,'423') + etprod('123',bbdm,'413',g'*a','42');
     bgads = etprod('1234',bb,'51',gads,'5234') + etprod('1234',bbds,'5134',g'*a','52');
     
     % Derivative of Sc
     cdm = permute(badm,[2,1,3]) + bgadm; cds = permute(bads,[2,1,3,4]) + bgads;
     Scdm = etprod('123',c,'14',Czdm,'423') + etprod('123',cdm,'143',Cz,'42');
     Scds = etprod('1234',c,'15',Czds,'5234') + etprod('1234',cds,'1534',Cz,'52');
     Scdm = Scdm + permute(Scdm,[2,1,3]); Scds = Scds + permute(Scds,[2,1,3,4]);

     % Derivative of Sio
     gaCdm = etprod('123',g'*a','14',Czdm,'423') + etprod('123',gadm,'143',Cz,'42');
     gaCds = etprod('1234',g'*a','15',Czds,'5234') + etprod('1234',gads,'1534',Cz,'52');
     %Siodm = [Czdm(dyni,:,:); gaCdm]; Siods = [Czds(dyni,:,:,:); gaCds];
     Siodm = [Czdm; gaCdm]; Siods = [Czds; gaCds];
    end
end

% Add in conditional term
i = [Dold+dyni D2+1:D3]; j = D3+1:D4; i = [Dold+1:Dold+Do D2+1:D3];
dynmodel.ratio = sqrt(diag(S(j,j)+Sc)./diag(S(j,j)));
try chol(S(j,j)+Sc); catch; fprintf('correctVar: S about to not be pos. def.\n'); keyboard; end
S(j,j) = S(j,j) + Sc;
S(i,j) = S(i,j) + Sio; S(j,i) = S(j,i) + Sio';
%S = S + 1e-5*eye(size(S,1));

if nargout > 2
    sdm = zeros(D4,D4,D4); sds = zeros(D4,D4,D4,D4);
    for l = 1:D4; for n = 1:D4; sds(l,n,l,n) = 1; end; end
    sdm(j,j,:) = Scdm; sdm(i,j,:) = Siodm; sdm(j,i,:) = permute(Siodm,[2,1,3]);
    sds(j,j,:,:) = sds(j,j,:,:) + Scds; 
    sds(i,j,:,:) = sds(i,j,:,:) + Siods; 
    sds(j,i,:,:) = permute(sds(i,j,:,:),[2,1,3,4]);
end

try chol(S([i j],[i j])); catch; fprintf('correctVar: S not pos. def.\n'); keyboard; end
if ~isreal(S([i j],[i j])); fprintf('S not real\n'); keyboard; end
k = [Dold+1:Dold+Do D2+1:D3];
try chol(S([k j],[k j])); catch; fprintf('correctVar: S not pos. def.\n'); keyboard; end
