# Paper

- `make_tables.py` writes every table of the paper that contains data (Tables 2 to 8, B.1, C.1, D.1) and all
  supplementary tables to `output/paper/tables/` and `output/paper/supplement_tables/`, together with a README that
  lists the source of each table. Descriptive text of the types, detectors and field mapping comes from
  `text/method_tables.tex`.
- `make_figures.py` writes Figures 1 to 6 and C.1 to `output/paper/figures/` (PDF, EPS, PNG).
- `assemble.py` builds `output/paper/build/main.tex` from the hand-written sections in `manuscript/sections/`, the
  generated tables and the figure floats in `manuscript/floats/` (every `%%FLOAT{label}` marker is replaced), and
  `output/paper/build/supplement.tex` from `supplement/supplement.tex` and the generated supplementary tables.
- `make_variants.py` derives the double-anonymized manuscript, the title page and the anonymized supplement
  (`output/paper/anon/`).

The text of the paper is written by hand; every number in its tables and figures is generated.
