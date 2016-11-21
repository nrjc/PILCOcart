function [cc, dcc] = uncertaintyNoReduction(~,~,cc,dcc,~,~,~,~,~,~)
% Dummy function which does not compute the reduced variance on the cumilative 
% cost at the next timestep, merely what is computed at the current timestep.
%
% [cc, dcc] = uncertaintyNoReduction([],[],cc_pre,dcc_pre)
%
% cc_pre          cumulative (discounted) cost structure before fantasy data
% dcc_pre         derivative structure of cc_pre
%
% Copyright (C) 2016 Carl Edward Rasmussen and Rowan McAllister 2016-05-10
