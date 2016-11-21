function [L dLdp] = congpReg(policy)

try
  if ~isfield(policy,'inputs'); policy.inputs = policy.p.inputs; end
  if ~isfield(policy,'target'); policy.target = policy.p.target; end
  if ~isfield(policy,'hyp'); policy.hyp = policy.p.hyp; end
catch
  fprintf('congpReg');
  L = 0;
  dLdp = 0;
end

lsi = log(std(policy.inputs,[],1))';                % lengthscale center
sc = 20; ra = log(1000); p = 20;            % regularisation parameters

dLdp.hyp = 0*policy.hyp; dLdp.inputs = 0*policy.inputs;  % initialise
dLdp.target = 0*policy.target;

d = bsxfun(@minus, policy.hyp(1:end-2,:), lsi);
L = sc*sum(sum((abs(d)/ra).^p));

dLdp.hyp(1:length(lsi),:) = sc*p*((abs(d)/ra).^(p-1)).*sign(d)/ra;

dLdp = unwrap(dLdp);