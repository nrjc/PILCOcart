function [dd, da, dn] = GPfstepT(deriv,p,dyn,plant,m,s,y,u,delta)

if nargin < 5; delta = 1e-4; end

switch deriv
  
  case 'dMdp'
      [dd, da, dn] = checkgrad(@gpT1,p,delta,dyn,plant,m,s,y,u);
 
  case 'dSdp'
      [dd, da, dn] = checkgrad(@gpT2,p,delta,dyn,plant,m,s,y,u);
 
  case 'dnlpdp'
      [dd, da, dn] = checkgrad(@gpT3,p,delta,dyn,plant,m,s,y,u);

end


function [f, df] = gpT1(p,dyn,plant,m,s,y,u)                      % dMdp
[dyn,q] = p2dyn(p,dyn);
if nargout == 1; f = GPfstep(p,dyn,plant,m,s,y,u);
else [f, ~, ~, ~, df] = GPfstep(p,dyn,plant,m,s,y,u); %df = dfp(df,p,q);
end

function [f, df] = gpT2(p,dyn,plant,m,s,y,u)                      % dSdp
[dyn,q] = p2dyn(p,dyn);
if nargout == 1; [~, f] = GPfstep(p,dyn,plant,m,s,y,u);
else [~, f, ~, ~, ~, df] = GPfstep(p,dyn,plant,m,s,y,u); %df = dfp(df,p,q);
end

function [f, df] = gpT3(p,dyn,plant,m,s,y,u)                      % dnlpdp
[dyn,q] = p2dyn(p,dyn);
if nargout == 1; [~, ~, f] = GPfstep(p,dyn,plant,m,s,y,u);
else [~, ~, f, df] = GPfstep(p,dyn,plant,m,s,y,u); %df = dfp(df,p,q);
end

function [dyn,p] = p2dyn(p,dyn)
if isfield(p,'b'); [dyn.hyp.b] = deal(p.b); else [p.b] = deal(dyn.hyp.b); end
if isfield(p,'beta'); dyn.beta = [p.beta]; 
else for i=1:size(dyn.beta,2); p(i).beta = dyn.beta(:,i); end; end
if isfield(p,'l'); [dyn.hyp.l] = deal(p.l); else [p.l] = deal(dyn.hyp.l); end
if isfield(p,'pn'); [dyn.hyp.pn] = deal(p.pn); else [p.pn] = deal(dyn.hyp.pn); end
if isfield(p,'on'); [dyn.hyp.on] = deal(p.on); else [p.on] = deal(dyn.hyp.on); end
if isfield(p,'m'); [dyn.hyp.m] = deal(p.m); else [p.m] = deal(dyn.hyp.m); end
if isfield(p,'n'); [dyn.hyp.n] = deal(p.n); else [p.n] = deal(dyn.hyp.n); end
if isfield(p,'s'); [dyn.hyp.s] = deal(p.s); else [p.s] = deal(dyn.hyp.s); end
dyn = preComp(dyn);

function df = dfp(df,p,q)
df = rewrapdp(q,df);
if ~isfield(p,'b');    df = rmfield(df,'b');    end
if ~isfield(p,'beta'); df = rmfield(df,'beta'); end
if ~isfield(p,'l');    df = rmfield(df,'l');    end
if ~isfield(p,'lsn');  df = rmfield(df,'lsn');  end
if ~isfield(p,'m');    df = rmfield(df,'m');    end
if ~isfield(p,'n');    df = rmfield(df,'n');    end
if ~isfield(p,'s');    df = rmfield(df,'s');    end
df = unwrapdp(df);