function checkBoundStable = checkBoundStable(stateS, stateM, varNum)
checkBoundStable = true;
outerAngle = squeeze(stateS(varNum,varNum,:));
outerAngleLen = length(outerAngle);
if (outerAngle(outerAngleLen) > outerAngle(outerAngleLen-1) ...
	&& outerAngle(outerAngleLen-1) > outerAngle(outerAngleLen-2))
	checkBoundStable = false;
end
return;