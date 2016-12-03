function checkBoundStable = checkrollout(stateS, stateM, lat, varNum)
countOfUnstable = 0;
for rolloutNum=1:length(lat) 
	allLen = length(squeeze(stateS(varNum,varNum,:)));
	finalValue = lat{rolloutNum}.state(:,varNum);
	finalValue = finalValue(length(finalValue));
	finalPredictedMean = stateM(varNum,allLen) ;
	finalPredictedVar = squeeze(stateS(varNum,varNum,:));
	finalPredictedVar = finalPredictedVar(allLen);
	if (4*sqrt(finalPredictedVar) + finalPredictedMean < finalValue || ...
		-4*sqrt(finalPredictedVar) + finalPredictedMean > finalValue )
		countOfUnstable = countOfUnstable+1;
	end
end

checkBoundStable=0;
if (countOfUnstable > 3)
	checkBoundStable = 1;
	if (countOfUnstable > 5)
		checkBoundStable = 2;
	end
end

return;