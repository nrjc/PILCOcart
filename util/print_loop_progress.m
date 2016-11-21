function print_loop_progress(i, max_i, loop_description)

% Textual display of a loop's progress (less overhead from MATLAB's GUI wait
% bar). To be placed as first line of the loop.
%
% i                  current iteration number (int)
% max_i              maximum number of iterations (int)
% loop_description   (string)
%
% Rowan McAllister 2015-09-11

if nargin < 3; loop_description = 'looping'; end
calling_fcn = dbstack; calling_fcn = calling_fcn(2).name;
if i == 1
  fprintf(['\n',calling_fcn,': ',loop_description,'. Progress = 00%% ']);
elseif max_i < 100 || mod(i, round(max_i/100)) == 0;
  fprintf('\b\b\b\b%02d%%\n', round(100*i/max_i));
end
