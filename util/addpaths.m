% startup script to include all necessary paths

filename = which(mfilename);                                     
filename_parts = strsplit(filename, '/');
root_dir = [strjoin(filename_parts(1:end-2),'/'),'/'];

try
  addpath([root_dir 'base'])
  addpath([root_dir 'control'])
  addpath([root_dir 'direct'])
  addpath([root_dir 'explore'])
  addpath([root_dir 'gp'])
  addpath([root_dir 'loss'])
  addpath([root_dir 'test'])
  addpath([root_dir 'util'])
  addpath([root_dir 'util/tprod'])
catch
end