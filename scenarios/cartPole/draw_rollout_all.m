% Draw all the rollouts so far
for jj=1:J;
    draw_rollout(jj,0,data,H,dt,cost)
end
for jj = 1:j
  draw_rollout(jj,J,data,H,dt,cost,S{jj})
end
clear jj