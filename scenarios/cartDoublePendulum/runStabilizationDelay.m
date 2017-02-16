batchnum=0;
for error = [0.25 1 2 4]
    for delay = [0.051 0.053 0.055 0.057 0.060]
        doitStabilizeTiming(delay,error,batchnum);
        batchnum=batchnum+1;
    end
end