Ctrl.m: pilco.nw
	notangle -R$@ pilco.nw | cpif ../control/$@

CtrlNF.m: pilco.nw
	notangle -R$@ pilco.nw | cpif ../control/$@

loss.m: pilco.nw
	notangle -R$@ pilco.nw | cpif ../base/$@

propagate.m: pilco.nw
	notangle -R$@ pilco.nw | cpif ../base/$@

rollout.m: pilco.nw
	notangle -R$@ pilco.nw | cpif ../base/$@

simulate.m: pilco.nw
	notangle -R$@ pilco.nw | cpif ../base/$@

pilco.pdf: pilco.nw
	noweave -latex -index -delay pilco.nw | awk -f listfilt > pilco.tex
	pdflatex pilco.tex
	pdflatex pilco.tex
	rm -rf pilco.aux pilco.log pilco.out pilco.tex pilco.toc
