function text2(varargin)
%
% TEXT2 circumvents what seems to be an internal MATLAB bug, which can changes
% the current figure when calling text.
%
% Rowan McAllister 2015-03-24

h = gcf;
text(varargin{:})
set(0,'CurrentFigure',h);
