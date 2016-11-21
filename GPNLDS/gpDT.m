function [dd, da, dn] = gpDT(deriv, p, dynmodel, m, s, delta)

% test derivatives of the gp family of functions, 2012-06-26
%#ok<*ALIGN>
dynmodel.fcn = @gpD;
D = length(m);  E = length(dynmodel.hyp);                     % input size

if nargin < 6; delta = 1e-4; end
if isempty(p); 
    p = dynmodel.hyp; for i=1:E; p(i).beta = dynmodel.beta(:,i); end
end

switch deriv
  
  case 'dMdm'
      [dd, da, dn] = checkgrad(@gpT0, m, delta, dynmodel, s);
 
  case 'dSdm'
      [dd, da, dn] = checkgrad(@gpT1, m, delta, dynmodel, s);
 
  case 'dMds'
      [dd, da, dn] = checkgrad(@gpT3, s(tril(ones(D))==1), delta, dynmodel, m);
    
  case 'dSds'
      [dd, da, dn] = checkgrad(@gpT4, s(tril(ones(D))==1), delta, dynmodel, m);
 
  case 'dMdp'
      [dd, da, dn] = checkgrad(@gpT6, p, delta, dynmodel, m, s) ;
 
  case 'dSdp'
      [dd, da, dn] = checkgrad(@gpT7, p, delta, dynmodel, m, s) ;
      
  case 'dadl'
        p = [dynmodel.hyp.l];
        [dd, da, dn] = checkgrad(@gpT8, p, delta, dynmodel, m, s);
        
    case 'dcdl'
        p = [dynmodel.hyp.l];
        [dd, da, dn] = checkgrad(@gpT9, p, delta, dynmodel, m, s);
  
  case 'all'
      dlist = {'dMdm' 'dSdm' 'dMds' 'dSds'};
        for i=1:length(dlist)
            disp(dlist{i}); dd{i} = gpDT(dlist{i},dynmodel,m,s,delta);
        end

end


function [f, df] = gpT0(m, dyn, s)                             % dMdm
if nargout == 1; f = gpm(dyn, dyn.inputs, m, s);
else            [f, ~, df] = dyn.fcn(dyn, m, s);
end

function [f, df] = gpT1(m, dyn, s)                             % dSdm
if nargout == 1; [~, f] = gpm(dyn, dyn.inputs, m, s);
else             [~, f, ~, df] = dyn.fcn(dyn, m, s);
end

function [f, df] = gpT3(s, dyn, m)                             % dMds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  f = gpm(dyn, dyn.inputs, m, s);
else
  [f, ~, ~, ~, dMds] = dyn.fcn(dyn, m, s);
  dd = length(f); dMds = reshape(dMds,dd,d,d); df = zeros(dd,d*(d+1)/2);
    for i=1:dd; 
        dMdsi(:,:) = dMds(i,:,:); dMdsi = dMdsi + dMdsi'-diag(diag(dMdsi)); 
        df(i,:) = dMdsi(tril(ones(d))==1);
    end
end

function [f, df] = gpT4(s, dyn, m)                             % dSds
d = length(m);
v(tril(ones(d))==1) = s; s = reshape(v,d,d); s = s+s'-diag(diag(s));
if nargout == 1
  [~, f] = gpm(dyn, dyn.inputs, m, s);
else
    [M, f, ~, ~, ~, dSds] = dyn.fcn(dyn, m, s);
    dd = length(M); dSds = reshape(dSds,dd,dd,d,d); df = zeros(dd,dd,d*(d+1)/2);
    for i=1:dd; for j=1:dd                                      
        dSdsi(:,:) = dSds(i,j,:,:); dSdsi = dSdsi+dSdsi'-diag(diag(dSdsi)); 
        df(i,j,:) = dSdsi(tril(ones(d))==1);
    end; end
end

function [f, df] = gpT6(p, dyn, m, s)                          % dMdp
dyn = p2dyn(p,dyn);
if nargout == 1; f = gpm(dyn, dyn.inputs, m, s);
else [f, ~, ~, ~, ~, ~, df] = dyn.fcn(dyn, m, s); %df = dfp(df,p);
end

function [f, df] = gpT7(p, dyn, m, s)                          % dSdp
dyn = p2dyn(p,dyn);
if nargout == 1; [~, f] = gpm(dyn, dyn.inputs, m, s);
else [~, f, ~, ~, ~, ~, ~, df] = dyn.fcn(dyn, m, s); %df = dfp(df,p);
end

function [f, df] = gpT8(p, dyn, m, s)
for i=1:size(p,2); dyn.hyp(i).l = p(:,i); end
if nargout == 1; 
    [~,~,~,~,~,~,~,~,a] = dyn.fcn(dyn, m, s);
else [~,~,~,~,~,~,~,~,a,dadl] = dyn.fcn(dyn, m, s);
    [D, E] = size(a); df = zeros(D,E,D,E);
    for i=1:E; df(:,i,:,i) = dadl(:,:,i); end
end
f = a;

function [f, df] = gpT9(p, dyn, m, s)
for i=1:size(p,2); dyn.hyp(i).l = p(:,i); end
if nargout == 1; 
    [~,~,~,~,~,~,~,~,c] = dyn.fcn(dyn, m, s);
else [~,~,~,~,~,~,~,~,c,dcdl] = dyn.fcn(dyn, m, s);
%     E = size(dadlsf,2); df = zeros(size(a,1),E,E);
%     for i=1:E; df(:,i,i) = dadlsf(:,i); end
df = dcdl;
end
f = c;

function dyn = p2dyn(p,dyn)
if isfield(p,'b');   [dyn.hyp.b] = deal(p.b);     end
if isfield(p,'beta'); dyn.beta = [p.beta];        end
if isfield(p,'l');   [dyn.hyp.l] = deal(p.l);     end
if isfield(p,'lsn'); [dyn.hyp.lsn] = deal(p.lsn); end
if isfield(p,'m');   [dyn.hyp.m] = deal(p.m);     end
if isfield(p,'n');   [dyn.hyp.n] = deal(p.n);     end
if isfield(p,'s');   [dyn.hyp.s] = deal(p.s);     end
dyn = preComp(dyn);

function df = dfp(df,p)
if ~isfield(p,'b');    df = rmfield(df,'b');    end
if ~isfield(p,'beta'); df = rmfield(df,'beta'); end
if ~isfield(p,'l');    df = rmfield(df,'l');    end
if ~isfield(p,'lsn');  df = rmfield(df,'lsn');  end
if ~isfield(p,'m');    df = rmfield(df,'m');    end
if ~isfield(p,'n');    df = rmfield(df,'n');    end
if ~isfield(p,'s');    df = rmfield(df,'s');    end