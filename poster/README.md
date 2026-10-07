# Poster

The poster presented at the Workshop on Large Language Models for Multimodal Data Fusion
(LLM4MDF), IEEE International Conference on Data Mining (ICDM 2026), Shenyang, China.
`DM2048` is the ICDM paper ID, printed in the bottom-right corner as the conference requires.

| File | Language | Notes |
| --- | --- | --- |
| [`DM2048.pdf`](DM2048.pdf) | English | the file submitted to the conference |
| [`DM2048_vi.pdf`](DM2048_vi.pdf) | Vietnamese | translation, not submitted |

Both are a single page on ICDM Template 1, 46.8 x 33.1 in landscape (3370.39 x 2383.94 pt).

## Contents

Seven sections, following the paper: Background and Challenges, Contributions, Overall
Architecture, Experimental Setup, Empirical Results, Qualitative Evidence, Conclusion. The
architecture figure is the five-stage pipeline of Section IV. Qualitative Evidence reproduces
Fig. 2 of the paper, one volume per dataset with the ground-truth report, what Reg2RG, CT-GRAPH
and MARCH wrote, and what MDEF wrote.

The architecture figure is in English in both versions, because it is a flat PDF with no
editable source.

## Building from source

The LaTeX sources are not in this repository, because they reference the camera-ready figures by
relative path and would not build on their own. The English poster is built with pdflatex and the
Vietnamese one needs XeLaTeX for the diacritics, both on beamerposter with Lato.
