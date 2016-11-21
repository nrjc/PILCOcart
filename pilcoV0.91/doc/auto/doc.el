(TeX-add-style-hook "doc"
 (lambda ()
    (TeX-add-symbols
     '("mat" 1)
     "inv"
     "T"
     "E"
     "var"
     "cov"
     "R"
     "polpar"
     "path")
    (TeX-run-style-hooks
     "listings"
     "textcomp"
     "xcolor"
     "color"
     "amsmath"
     "amssymb"
     "mathtools"
     "dsfont"
     "geometry"
     "latex2e"
     "art10"
     "article")))

