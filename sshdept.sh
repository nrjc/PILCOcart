#!/bin/sh
jump_host="nrjc2@gate.eng.cam.ac.uk"
local_path="./scenarios/cartDoublePendulum/delayTest"
destination_path="~/PILCOcart/scenarios/cartDoublePendulum/*35_H60.mat"
host="dirichlet"
host2="nrjc2@dirichlet"
scp -o ProxyCommand="ssh $jump_host nc $host 22" $host2:$destination_path $local_path 