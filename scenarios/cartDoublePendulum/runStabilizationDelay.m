batchnum=20;
for error = [0.25 1 2 4]
    for delay = [0.07 0.08 0.09 0.095]
        doitStabilizeTiming(delay,error,batchnum);
        batchnum=batchnum+1;
    end
end