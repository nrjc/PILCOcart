function dyn = updateHyp(dyn,p)
if isfield(p,'beta'); dyn.beta = [p.beta]; p = rmfield(p,'beta'); end % beta
% n = {dyn.hyp.n};                                % store n
dyn.hyp = p;                                    % set dyn.hyp to p
% [dyn.hyp.n] = deal(n{:});                       % restore dyn.hyp.n