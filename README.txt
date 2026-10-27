Reproduction of Sanders, Wilson & Gershman (2020), "Hippocampal remapping as
hidden state inference", eLife 9:e51140.

Reproduces Figure 3 (experience-dependent remapping), Figure 7B (the
early-vs-late morph prediction) and Figure 9A (Figure 3C for three values of
alpha), and runs a sensitivity analysis over alpha,
the normal-Wishart hyperparameters and the generative sd. Findings are written
up in REPRODUCTION_NOTES.md.


FILES

  sanders_model.py          Inference machinery, written from the paper's
                            equations: CRP (Eq 4, 5), normal-Wishart marginal
                            likelihood (Eq 2), Student-t posterior predictive
                            (Eq 10), partition posterior (Eq 6), partition and
                            state evidence ratios (Eq 7, Eq 11), probe
                            assignment (Eq 8).
  test_sanders_model.py     Consistency checks: Eq 2 vs numerical integration,
                            chained Eq 10 vs Eq 2, Eq 5 vs sequential Eq 4,
                            normalisation.
  experiments.py            Figures 3 and 7B, seed robustness, sensitivity sweeps.
  compare_original.py       Optional validation: compares this code's numbers
                            with the authors' released code (see step 4).
  requirements.txt          Packages for this code.
  requirements-original.txt 2020-era packages needed to run the authors'
                            released code, used only by compare_original.py.
  figures/                  Output figures (figures/original/ = the authors'
                            committed figures, rendered from their repo, MIT licence).
  results/                  CSV/JSON outputs behind every figure and number.


HOW TO RUN

All commands are run from this folder (sanders2020_reproduction/).

1. Install (any recent Python 3; tested on 3.13):

     python -m venv .venv
     source .venv/bin/activate          (Windows: .venv\Scripts\activate)
     pip install -r requirements.txt

2. Run the tests (about 5 s):

     pytest -q

3. Reproduce the figures and the sensitivity analysis:

     python experiments.py --quick      (about 1.5 min, fewer seeds)
     python experiments.py              (full run, about 15-20 min)

   Outputs:
     figures/fig3_reproduction.png      Fig 3B (predictive densities) and 3C
                                        (negative partition evidence ratio)
     figures/fig7_reproduction.png      Fig 7B (morph evidence ratio, 5 vs 25 pairs)
     figures/fig9a_reproduction.png     Fig 9A (Fig 3C for alpha = 1, 1e-5, 1e-10)
     results/sensitivity_table.csv      the review's Appendix table: seed-1 Fig 3
                                        run with one setting changed; first pair
                                        favouring two states, and the evidence
                                        ratio after 15 pairs in the paper's sign
                                        (Fig 3C plots it reversed)
     figures/sensitivity_1d.png         one-at-a-time sweeps of alpha, nu0,
                                        kappa0, T0, sigma
     figures/sensitivity_alpha_sd.png   alpha x sigma grid
     results/summary.json               headline numbers quoted in the notes

   If results/comparison_original.json is present (it is included), the
   released code's values are drawn as crosses on the Fig 3 and Fig 7B plots.

4. Optional: validate against the authors' released code.

   This step is not needed to run anything above. It checks that this
   reimplementation gives the same numbers as the authors' code, and it is the
   source of the agreement reported in the review. The authors' code is not
   included here and none of it is copied; compare_original.py imports their
   modules, unmodified, from a separate download.

   a. Download the authors' code (MIT licence), commit 8accdd8:

        git clone https://github.com/HoniSanders/Sanders-et-al-2020-Elife
        git -C Sanders-et-al-2020-Elife checkout 8accdd8

   b. Their code fails on NumPy 2, so make a separate Python 3.8 environment:

        python3.8 -m venv .venv-original
        source .venv-original/bin/activate
        pip install -r requirements-original.txt

   c. Run the comparison, giving the path to their folder:

        python compare_original.py /path/to/Sanders-et-al-2020-Elife

   It prints the largest differences for Fig 3C and Fig 7B and rewrites
   results/comparison_original.json. Re-run experiments.py afterwards (in the
   main environment) to redraw the crosses from the new file.


PARAMETERS

  alpha = 0.1                              paper, Methods
  normal-Wishart mu0 = 0, kappa0 = 0.001,
    nu0 = 0.02, T0 = 0.02 I                paper, Methods
  context means -1 / +1, sd = 0.3          paper, Fig 3 text (p. 7); also
                                           released code (partitionParams.py)
  Fig 3: 15 pairs, seed 1                  released code
  Fig 7B: 5 vs 25 training pairs,
    11 probes from -1 to +1                released code (morph.py)

"Pairs": each training step shows one observation from each context, so
5 and 25 pairs are 10 and 50 observations. The original Fig 7B legend says
"5/25 Observations".

The seed-1 curves use the same random stream as the released code (see
ORIGINAL_BURN_PER_PAIR in sanders_model.py). Seed bands and sweeps use clean
streams for seeds 1..N.
