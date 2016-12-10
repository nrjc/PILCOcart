function i=iterateDir(P)
try
    loc = ['./result' int2str(P) '/'];
	cd(loc);
	copyfile('*','..');
	cd('..');
	i=true;
	return;
catch
	i=false;
	return;
end



end