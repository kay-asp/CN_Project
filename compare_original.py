"""Numerical comparison of this reimplementation against the authors' released
code (github.com/HoniSanders/Sanders-et-al-2020-Elife, commit 8accdd8).

The released code needs a 2020-era stack (see requirements-original.txt), so
run this script with that interpreter:

    python compare_original.py /path/to/Sanders-et-al-2020-Elife

The original modules are imported unmodified; nothing in their repo is edited.
Writes results/comparison_original.json and prints a summary.
"""
import json
import os
import sys

import numpy as np
from scipy.special import logsumexp

import model as m

HERE = os.path.dirname(os.path.abspath(__file__))


def load_original(repo):
    """Import the released code. It has its own model.py, so ours is taken out
    of sys.modules first; it stays available here as `m`."""
    import matplotlib
    matplotlib.use("Agg")
    sys.modules.pop("model", None)
    sys.path.insert(0, os.path.abspath(repo))
    import partitionParams  # imports their model.py; plotting only runs under __main__
    import myutils
    return partitionParams, sys.modules["model"], myutils.params


def original_fig2(their_model, params):
    """Fig 2 with the released model.py, following nosinglecue.py, which is not
    imported because it plots and saves a file when imported."""
    from scipy import stats
    np.random.seed(params["seed"])
    c1 = stats.norm(loc=np.zeros((1, 4)), scale=0.2 * np.ones((1, 4)))
    animal = their_model.Animal(worldModel=their_model.WorldModel(their_model.CRP(params["alpha"]),
                                                                   their_model.Gauss()))
    for _ in range(20):
        animal.experience(c1.rvs())
    animal.clusterAssignments = np.zeros((20, 1))
    post = np.vstack([animal.belief(np.array(p, dtype=float)).ravel() for p in FIG2_PROBES])
    return animal.experiences, post[:, 0] - post[:, 1]


FIG2_PROBES = [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 1, 1, 1]]


def ours_fig2(alpha, obs):
    Y = np.random.RandomState(1).normal(0, 0.2, (20, 4))
    z = np.zeros(20, dtype=int)
    return Y, np.array([m.state_evidence_ratio(np.array(p, dtype=float), Y, z, alpha, obs, k=0)
                        for p in FIG2_PROBES])


def ours_fig3(n_pairs, alpha, obs):
    Y, z1, z2 = m.two_context_observations(n_pairs, burn_per_pair=m.ORIGINAL_BURN_PER_PAIR)
    lp1 = [m.log_partition_posterior(Y[:2 * n], z1[:2 * n], alpha, obs) for n in range(1, n_pairs + 1)]
    lp2 = [m.log_partition_posterior(Y[:2 * n], z2[:2 * n], alpha, obs) for n in range(1, n_pairs + 1)]
    return Y[:, 0], np.array(lp1), np.array(lp2)


def ours_morph(n_pairs, probes, alpha, obs):
    Y, z1, z2 = m.two_context_observations(n_pairs, burn_per_pair=m.ORIGINAL_BURN_PER_PAIR)
    rows = [np.concatenate(m.hypothesis_posterior([p], Y, [z1, z2], alpha, obs)) for p in probes]
    return np.array(rows)


def morph_ratio(b):
    # columns: P1 same (S1), P1 new (D1), P2 square (S2), P2 circle (D2), P2 new (D3)
    return logsumexp(b[:, [0, 2]], axis=1) - logsumexp(b[:, [1, 3, 4]], axis=1)


def main(repo):
    pp, their_model, params = load_original(repo)
    alpha = pp.alpha_default
    obs = m.NormalWishart()
    out = {"alpha": alpha, "sd": pp.sd_default}

    # Fig 3: partition log posteriors after each pair of observations
    n = 15
    sim, lps, _, _, _, _, _, exps, _ = pp.experiment(N=n)
    y, lp1, lp2 = ours_fig3(n, alpha, obs)
    orig_ratio = lps[:, 1] - lps[:, 0]              # what Fig 3C plots
    our_ratio = lp2 - lp1
    out["fig3"] = {
        "samples_max_abs_diff": float(np.max(np.abs(exps - y))),
        # the released CRP drops one factor of alpha, so each partition is
        # offset by -log(alpha); the ratio is unaffected
        "per_partition_offset": [float(np.mean(lp1 - lps[:, 0])), float(np.mean(lp2 - lps[:, 1]))],
        "per_partition_offset_spread": float(np.ptp(np.r_[lp1 - lps[:, 0], lp2 - lps[:, 1]])),
        "expected_offset_log_alpha": float(np.log(alpha)),
        "orig_ratio": orig_ratio.tolist(),
        "our_ratio": our_ratio.tolist(),
        "ratio_max_abs_diff": float(np.max(np.abs(orig_ratio - our_ratio))),
    }

    # Fig 7B: morph evidence ratio after 5 and 25 training pairs
    probes = np.linspace(-1, 1, 11)
    out["fig7"] = {"probes": probes.tolist()}
    for n_pairs in (5, 25):
        rets = pp.experiment(N=n_pairs, alpha=alpha)
        animal, part1, part2 = rets[2], rets[3], rets[4]
        orig_b = np.array([pp.partitions2activity(animal, p, [part1, part2],
                                                  [np.zeros(2)] * 3)[2].ravel() for p in probes])
        ours_b = ours_morph(n_pairs, probes, alpha, obs)
        out["fig7"][str(n_pairs)] = {
            "orig_ratio": morph_ratio(orig_b).tolist(),
            "our_ratio": morph_ratio(ours_b).tolist(),
            "belief_max_abs_diff": float(np.max(np.abs(orig_b - ours_b))),
            "ratio_max_abs_diff": float(np.max(np.abs(morph_ratio(orig_b) - morph_ratio(ours_b)))),
        }

    # Fig 2: state evidence ratio for four probes after 20 four-feature
    # observations. Run last because it reseeds the global random state.
    exps2, orig2 = original_fig2(their_model, params)
    y2, ours2 = ours_fig2(params["alpha"], obs)
    out["fig2"] = {
        "samples_max_abs_diff": float(np.max(np.abs(exps2 - y2))),
        "orig_ratio": orig2.tolist(),
        "our_ratio": ours2.tolist(),
        "ratio_max_abs_diff": float(np.max(np.abs(orig2 - ours2))),
    }

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "comparison_original.json"), "w") as f:
        json.dump(out, f, indent=1)

    f3 = out["fig3"]
    print("Fig 3  training samples, max |diff|:       %.2e" % f3["samples_max_abs_diff"])
    print("Fig 3  per-partition offset (ours - orig): %.6f, %.6f  (log alpha = %.6f, spread %.1e)"
          % (f3["per_partition_offset"][0], f3["per_partition_offset"][1],
             f3["expected_offset_log_alpha"], f3["per_partition_offset_spread"]))
    print("Fig 3  evidence ratio, max |diff|:         %.2e" % f3["ratio_max_abs_diff"])
    for k in ("5", "25"):
        r = out["fig7"][k]
        print("Fig 7B %2s pairs: beliefs max |diff| %.2e, ratio max |diff| %.2e"
              % (k, r["belief_max_abs_diff"], r["ratio_max_abs_diff"]))
    f2 = out["fig2"]
    print("Fig 2  training samples, max |diff|:       %.2e" % f2["samples_max_abs_diff"])
    print("Fig 2  released code:  %s" % np.round(f2["orig_ratio"], 3))
    print("Fig 2  this code:      %s" % np.round(f2["our_ratio"], 3))
    print("Fig 2  evidence ratio, max |diff|:         %.2e" % f2["ratio_max_abs_diff"])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
