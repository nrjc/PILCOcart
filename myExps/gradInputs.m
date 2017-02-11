function [fX, dFX] = gradInputs(pinputs, gp, q, myr)
	%q = [0.0000 0.2112 3.2024 -9.4025 0.1400 3.3774 -4.4303 -0.0608 -0.9982 -0.2336 -0.9723]';
	% myr = zeros(11,11);
	%myr = diag(eye(11,1));

	oldinputs = gp.inputs;
	%oldbeta = gp.beta;
	
	gp.inputs = pinputs;
	%gp.beta = pinputs;


	[M, S, C, dMdm, dSdm, dCdm, dMds, dSds, dCds, dMdp, dSdp, dCdp] = gpD(gp, q, myr);


	%check M of 1st GP wrt inputs
	%fX = M(2);
	%dFX = dMdp(2,gp.idx(2).inputs);

	%fX = M;
	%dFX = dMdp(1,gp.idx.inputs);


	%fX = C(10);
	%dFX = dCdp(10,gp.idx(1).inputs);

	%fX = S;
	fX = S(1,2);
	% keyboard;
	Sindex = 2;
	dFX=dSdp(Sindex,gp.idx(1).inputs) + dSdp(Sindex,gp.idx(2).inputs); % shared inputs
	% dFX = dSdp(4,gp.idx(2).inputs);	

	%fX = S(2,2);
	%dFX = dSdp(4,gp.idx(1).inputs);	


	%fX = M(2);
	%dFX = dMdp(2,gp.idx(2).beta)';

	%fX = S;
	%dFX = dSdp(1,gp.idx(1).beta);	

	gp.inputs = oldinputs;
	%gp.beta = oldbeta;


% [x,fx] = gradInputs(self.beta, self, q, r);
%checkgrad('gradInputs', self.beta, 1e-04, self, q, r)



%[x,fx] = gradInputs(self.inputs, self);
%checkgrad('gradInputs', self.inputs, 1e-04, self)

%[x,fx] = gradInputs(self.inputs, self, q, r);
%checkgrad('gradInputs', self.inputs, 1e-04, self, q, r)