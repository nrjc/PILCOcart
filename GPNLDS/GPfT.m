function [f, df] = GPfT(p, dyn, plant, y, u)

% update dyn with parameters under test and fill in missing parameters
[dyn, q] = p2dyn(p,dyn);

if nargout < 2
    f = GPf(q, dyn, plant, y, u);
else
    [f, df] = GPf(q, dyn, plant, y, u);
    
    % remove the derivatives we are not interested in
    df = dfp(df, p);
end



function [dyn, p] = p2dyn(p,dyn)
if isfield(p,'b'); [dyn.hyp.b] = deal(p.b); else [p.b] = deal(dyn.hyp.b); end
if isfield(p,'beta'); dyn.beta = [p.beta]; 
else for i=1:size(dyn.beta,2); p(i).beta = dyn.beta(:,i); end; end
if isfield(p,'l'); [dyn.hyp.l] = deal(p.l); else [p.l] = deal(dyn.hyp.l); end
if isfield(p,'on'); [dyn.hyp.on] = deal(p.on); else [p.on] = deal(dyn.hyp.on); end
if isfield(p,'m'); [dyn.hyp.m] = deal(p.m); else [p.m] = deal(dyn.hyp.m); end
if isfield(p,'n'); [dyn.hyp.n] = deal(p.n); else [p.n] = deal(dyn.hyp.n); end
if isfield(p,'s'); [dyn.hyp.s] = deal(p.s); else [p.s] = deal(dyn.hyp.s); end
dyn = preComp(dyn);

function df = dfp(df, p)
if ~isfield(p,'b');    df = rmfield(df,'b');    end
if ~isfield(p,'beta'); df = rmfield(df,'beta'); end
if ~isfield(p,'l');    df = rmfield(df,'l');    end
if ~isfield(p,'on');  df = rmfield(df,'on');  end
if ~isfield(p,'m');    df = rmfield(df,'m');    end
if ~isfield(p,'n');    df = rmfield(df,'n');    end
if ~isfield(p,'s');    df = rmfield(df,'s');    end