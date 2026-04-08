# Thesis Documentation

Documentation and written work for the EEG-Based Motor Imagery BCI graduation project.

| Directory | Contents |
| --------- | --------- |
| **LaTeX-Bachelor/** | Final thesis — SDNU LaTeX template, compiled PDF |
| **citations/** | Cited articles (PDFs) and missing papers checklist |
| **thesis/** | Thesis work by stage: 01_Inception, 02_Midterm, 03_Final |
| **architecture/** | System architecture design docs (from inception phase) |
| **notes/** | Study notes on signal processing, feature extraction, etc. |

## Compiling the Thesis

```bash
cd docs/LaTeX-Bachelor
latexmk -xelatex sdnubachelor.tex
```

Requires TeX Live (2017+) with XeLaTeX. The main content files are in `LaTeX-Bachelor/data/resource/`:

| File | Content |
|------|---------|
| `article.tex` | Main body (Chapters 1–8) |
| `abstract.tex` | Chinese & English abstracts |
| `references.bib` | BibTeX references |
| `appendix.tex` | Appendix (core source code listings) |
| `thanks.tex` | Acknowledgments |
