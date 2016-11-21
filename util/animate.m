function animate(latent, data, dt, cost, b, movname, movtype)
% ANIMATE displays rollout trajectories, and (optional) produces a movie
%
% animate(latent, data, dt, cost, b, movname, movtype)
%
% latent    struct-array   latent rollout states
%           or struct
% data      struct-array   rollout data, comprises the sequential state and
%           or struct      control actions per time step
% dt        double         (optional) the data's time step (default 0.15)
% cost      object         (optional) cost object
% b         J structs      (optional) J belief-sequence arrays
%   mean    H+1xD          sequence of uncertainty means
%   var     H+1xDxD        sequence of uncertainty variances
% movname   string         (optional) name of movie file to create in current
%                          directory (no movie created if movname not inputted)
% movtype   string         (optional) movie file encoding, e.g. 'avi' or 'mp4'
%                          (default 'avi')
%
% Rowan McAllister 2015-10-22

% Constants
TRIAL_START_PAUSE = 1; % seconds
HFIG = 5;              % figure handle
QUALITY = 75;          % video quality (if make_movie == True)
FRAMERATE = 24;        % number of frames per second (if make_movie == True)

% Optional inputs + default values
if exist('dt','var'); dt1 = dt; else dt1 = 0.15; end
if ~exist('cost','var'); cost = []; end
eb = exist('b','var') && ~isempty(b);
make_movie = exist('movname','var') && ~isempty(movname) && ischar(movname);
if ~exist('movtype','var'); movtype = 'avi'; end
if make_movie; tmpfile = ['tmp_' movname '.avi']; end

J = length(data);                                      % number of trajectories
T = dt1*arrayfun(@(a)size(a.state,1)-1,data);      % time taken each trajectory
cT = cumsum(T);                         % cumulative time taken each trajectory

% Prepare figure
if ~ishandle(HFIG); figure(HFIG); end;
set(0,'CurrentFigure',HFIG); set(HFIG,'DoubleBuffer','on'); clf(HFIG);

% Create movie object
if make_movie
  vidObj = VideoWriter(tmpfile); % can only produce avi
  vidObj.Quality = QUALITY;
  vidObj.FrameRate = FRAMERATE;
  open(vidObj);
end

% Loop over each trajectory
for j=1:J
  t1 = (0:dt1:T(j)-dt1)';
  s1 = latent(j).state(1:end-1,:); ns = size(s1,2); assert(size(s1,1) == numel(t1));
  u1 = data(j).action; nu = size(u1,2); assert(size(u1,1) == numel(t1));
  if eb; bm1 = b(j).mean(1:end-1,:); bv1 = b(j).var(1:end-1,:,:); end
  
  % interpolate
  dt2 = 1/FRAMERATE;
  t2 = (0:dt2:T(j)-dt1)'; n2 = numel(t2);
  s2 = nan(n2,ns); u2 = nan(n2,nu); bm2 = nan(n2,ns); bv2 = nan(n2,ns,ns);
  for i=1:ns; s2(:,i) = interp1(t1,s1(:,i),t2); end
  for i=1:nu; u2(:,i) = interp1(t1,u1(:,i),t2); end
  if eb
    for i=1:ns; bm2(:,i) = interp1(t1,bm1(:,i),t2); end
    for i=1:ns; for k=1:ns; bv2(:,i,k) = interp1(t1,bv1(:,i,k),t2); end; end
  end
  
  % Loop over states in a trajectory
  n = 0; ttrailstart = clock;
  while n < n2
    tn = clock;
    
    % freeze sim at start of each trial
    if etime(clock,ttrailstart) < TRIAL_START_PAUSE; n=1; else n=n+1; end
    
    % draw
    s.m = s2(n,:)';
    if eb; bm2n=bm2(n,:); bv2n=squeeze(bv2(n,:,:)); else bm2n=[]; bv2n=[]; end
    draw(s, u2(n,:), cost, bm2n, bv2n, ...
      ['controlled trial # ' num2str(j) ', T=' num2str(T(j)) ' sec'], ...
      ['total experience (after this trial): ' num2str(cT(j)) ' sec']);
    
    % record frame (if making a movie)
    if make_movie; writeVideo(vidObj, getframe(HFIG)); end
    
    pause(dt2 - etime(clock,tn)); % the sim in figure happens in 1x time
  end
  
end

% compress movie (if exists)
if make_movie;
  close(vidObj);
  compress_failed = system(['avconv -loglevel quiet -i ' tmpfile ...
    ' -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"' ... % even # pixels for libx264
    ' -c:v libx264 -c:a copy ' movname '.' movtype]);
  if compress_failed
    warning(['\nanimate: had trouble generating compressed file %s.%s on ' ...
      'this machine. Generated uncompressed file %s.avi instead.\n'], ...
      movname, movtype, movname);
    system(['mv ' tmpfile ' ' movname '.avi']);
  else
    system(['rm ' tmpfile]);
  end
end
