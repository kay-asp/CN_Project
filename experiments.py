"""Reproduce Figs 3 and 7B of Sanders, Wilson & Gershman (2020) and run a
sensitivity analysis.

    python experiments.py            # full run (a few minutes)
    python experiments.py --quick    # fewer seeds and grid points (~30 s)

Writes figures/*.png and results/*.csv|json. If results/comparison_original.json
exists (from compare_original.py), the released code's numbers are overlaid.
"""
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.special import logsumexp

import sanders_model as sm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG, RES = os.path.join(HERE, "figures"), os.path.join(HERE, "results")

# Parameters: paper Methods (alpha, normal-Wishart prior); released code
# partitionParams.py (sd = 0.3, means +-1, 15 pairs for Fig 3); morph.py
# (5 vs 25 training pairs, 11 morph probes from -1 to +1).
DEFAULTS = dict(alpha=0.1, sd=0.3, nu0=0.02, kappa0=1e-3, t0=0.02)
FIG3_PAIRS, EARLY, LATE, N_MAX = 15, 5, 25, 40
PROBES = np.linspace(-1, 1, 11)

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
BLUES = ListedColormap(["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                        "#256abf", "#184f95", "#0d366b"])
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "font.size": 9, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def make_obs(p):
    return sm.NormalWishart(kappa0=p["kappa0"], nu0=p["nu0"], t0=p["t0"])


def data(n_pairs, p, seed, match_original=False):
    burn = sm.ORIGINAL_BURN_PER_PAIR if match_original else 0
    return sm.two_context_observations(n_pairs, sd=p["sd"], seed=seed, burn_per_pair=burn)


def fig3_curve(Y, z1, z2, p):
    """-Eq 7 (log P(two states) - log P(one state)) after each pair, as in Fig 3C."""
    obs = make_obs(p)
    return np.array([sm.partition_evidence_ratio(Y[:2 * n], z2[:2 * n], z1[:2 * n], p["alpha"], obs)
                     for n in range(1, len(Y) // 2 + 1)])


def morph_curve(Y, z1, z2, p):
    """Fig 7B: log [P(S1) + P(S2)] - log [P(D1) + P(D2) + P(D3)] for each probe.
    S1/D1: one state, probe in it / in a new state. S2/D2/D3: two states,
    probe with the square (-1) state / with the circle (+1) state / new."""
    obs = make_obs(p)
    lps = np.array([sm.log_partition_posterior(Y, z, p["alpha"], obs) for z in (z1, z2)])
    lps -= logsumexp(lps)
    out = []
    for y in PROBES:
        b1 = lps[0] + sm.state_posterior([y], Y, z1, p["alpha"], obs)
        b2 = lps[1] + sm.state_posterior([y], Y, z2, p["alpha"], obs)
        out.append(np.logaddexp(b1[0], b2[0]) - logsumexp([b1[1], b2[1], b2[2]]))
    return np.array(out)


def crossing(curve):
    """First pair count from which two states are preferred for good (NaN if never)."""
    neg = np.nonzero(curve <= 0)[0]
    if neg.size == 0:
        return 1.0
    return float(neg[-1] + 2) if neg[-1] + 1 < len(curve) else np.nan


def summarise(p, seed):
    """Scalar summaries of one simulated animal."""
    Y, z1, z2 = data(N_MAX, p, seed)
    n_star = crossing(fig3_curve(Y, z1, z2, p))
    n_switch, early, late = np.nan, None, None
    for n in range(1, N_MAX + 1):
        c = morph_curve(Y[:2 * n], z1[:2 * n], z2[:2 * n], p)
        if n == EARLY:
            early = c
        if n == LATE:
            late = c
        if np.isnan(n_switch) and c.min() < 0 < c.max():
            n_switch = float(n)
        if n >= LATE and not np.isnan(n_switch):
            break
    holds = bool(early.min() > 0 and late.min() < 0 < late.max())
    return dict(n_star=n_star, n_switch=n_switch, holds=holds)


# ---------------------------------------------------------------- figures

def load_original():
    path = os.path.join(RES, "comparison_original.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def plot_fig3(seeds, orig):
    p = DEFAULTS
    Y, z1, z2 = data(FIG3_PAIRS, p, 1, match_original=True)
    curve = fig3_curve(Y, z1, z2, p)
    band = np.array([fig3_curve(*data(FIG3_PAIRS, p, s), p) for s in seeds])
    n = np.arange(1, FIG3_PAIRS + 1)

    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    # A: posterior predictive densities after 10 pairs, as in Fig 3B
    Y10, obs = Y[:20], make_obs(p)
    x = np.linspace(-2, 2, 801)
    one = np.exp([obs.log_predictive([v], Y10) for v in x])
    two = [np.exp([obs.log_predictive([v], Y10[k::2]) for v in x]) / 2 for k in (0, 1)]
    a.plot(x, one, color=ORANGE, label="1 state")
    a.plot(x, two[0], color=BLUE, label="2 states")
    a.plot(x, two[1], color=BLUE)
    a.plot(Y10[:, 0], np.zeros(20), "o", ms=4, color=INK, label="observations", clip_on=False)
    a.set(xlabel="feature value", ylabel="predictive density", title="A  Predictive densities after 10 pairs")
    a.legend(loc="upper left")

    # B: -partition evidence ratio vs experience, as in Fig 3C
    b.axhline(0, color=MUTED, lw=1)
    b.fill_between(n, *np.percentile(band, [5, 95], axis=0), color=BLUE, alpha=0.15, lw=0,
                   label="%d seeds, 5-95%%" % len(seeds))
    b.plot(n, np.median(band, axis=0), color=BLUE, lw=1, ls="--", label="%d seeds, median" % len(seeds))
    b.plot(n, curve, color=BLUE, marker="o", ms=4, label="this code, seed 1")
    if orig:
        b.plot(n, orig["fig3"]["orig_ratio"], "x", color=ORANGE, ms=8, mew=2, label="released code, seed 1")
    b.set(xlabel="number of experiences (pairs)", ylabel="log P(2 states) - log P(1 state)",
          title="B  Negative partition evidence ratio (-Eq 7)", ylim=(-25, 25))
    b.text(FIG3_PAIRS, 22, "favours remapping", ha="right", va="top", color=MUTED)
    b.text(FIG3_PAIRS, -22, "favours one map", ha="right", va="bottom", color=MUTED)
    b.legend(loc="upper left")
    fig.savefig(os.path.join(FIG, "fig3_reproduction.png"))
    plt.close(fig)
    return curve


def plot_fig7(seeds, orig):
    p = DEFAULTS
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    curves = {}
    for n, color in ((EARLY, ORANGE), (LATE, BLUE)):
        c = morph_curve(*data(n, p, 1, match_original=True), p)
        band = np.array([morph_curve(*data(n, p, s), p) for s in seeds])
        ax.fill_between(PROBES, *np.percentile(band, [5, 95], axis=0), color=color, alpha=0.15, lw=0)
        ax.plot(PROBES, c, color=color, marker="o", ms=4, label="after %d pairs (seed 1)" % n)
        if orig:
            ax.plot(PROBES, orig["fig7"][str(n)]["orig_ratio"], "x", color=INK, ms=7, mew=1.5,
                    label="released code" if n == LATE else None)
        curves[n] = c
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["square\n(-1)", "-0.5", "0", "0.5", "circle\n(+1)"])
    ax.set(xlabel="morph probe", ylabel="log odds: same as square vs different",
           title="Fig 7B  Morph prediction (bands: %d seeds, 5-95%%)" % len(seeds))
    ax.legend(loc="lower left")
    fig.savefig(os.path.join(FIG, "fig7_reproduction.png"))
    plt.close(fig)
    return curves


FIG9A_ALPHAS, FIG9A_PAIRS = (1.0, 1e-5, 1e-10), 20      # released code (levervariability.py)


def plot_fig9a():
    """Fig 9A: the Fig 3C curve (-Eq 7) for three values of alpha, seed 1."""
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    n = np.arange(1, FIG9A_PAIRS + 1)
    curves = {}
    for a, color in zip(FIG9A_ALPHAS, (ORANGE, BLUE, INK)):
        p = dict(DEFAULTS, alpha=a)
        curves[a] = fig3_curve(*data(FIG9A_PAIRS, p, 1, match_original=True), p)
        ax.plot(n, curves[a], color=color, marker="o", ms=3, label="alpha = %.0e" % a)
    ax.set(xlabel="number of experiences (pairs)", ylabel="log P(2 states) - log P(1 state)",
           title="Fig 9A  Negative partition evidence ratio for three alphas", ylim=(-25, 25))
    ax.legend(loc="upper left")
    fig.savefig(os.path.join(FIG, "fig9a_reproduction.png"))
    plt.close(fig)
    return curves


# Appendix table of the review: the seed-1 Fig 3 run with one setting changed.
TABLE_SETTINGS = [("published", None), ("alpha", 1e-10), ("alpha", 1e-5), ("alpha", 10.0),
                  ("nu0", 2.0), ("nu0", 10.0), ("kappa0", 1.0), ("kappa0", 10.0)]


def sensitivity_table():
    """First pair at which two states are favoured, and the evidence ratio
    after FIG3_PAIRS pairs. The ratio uses the paper's sign (Eq 7, one state
    against two), so it is the negative of what Fig 3C plots."""
    rows = []
    for name, value in TABLE_SETTINGS:
        p = dict(DEFAULTS, **({name: value} if value is not None else {}))
        c = fig3_curve(*data(FIG3_PAIRS, p, 1, match_original=True), p)
        first = np.nonzero(c > 0)[0]
        rows.append(dict(param=name, value=value if value is not None else "",
                         first_two_state_pair=int(first[0] + 1) if first.size else "none",
                         evidence_ratio_at_15=round(float(-c[-1]), 1)))
    return rows


SWEEPS = [  # (parameter, values, log x-axis, label)
    ("alpha", np.logspace(-20, 2, 23), True, "CRP concentration alpha"),
    ("nu0", np.logspace(-3, 1.5, 10), True, "Wishart dof nu0"),
    ("kappa0", np.logspace(-6, 1, 8), True, "mean pseudo-count kappa0"),
    ("t0", np.logspace(-4, 1, 11), True, "Wishart scale T0"),
    ("sd", np.linspace(0.1, 0.8, 8), False, "generative sd sigma"),
]


def run_sweeps(seeds):
    rows = []
    for name, values, _, _ in SWEEPS:
        for v in values:
            p = dict(DEFAULTS, **{name: v})
            for s in seeds:
                rows.append(dict(param=name, value=v, seed=s, **summarise(p, s)))
        print("  swept %s" % name)
    return rows


def plot_sweeps(rows):
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(14, 3.2), sharey=True)
    for ax, (name, values, logx, label) in zip(axes, SWEEPS):
        sub = [r for r in rows if r["param"] == name]
        for key, color, lab in (("n_star", BLUE, "Fig 3 switch (n*)"),
                                ("n_switch", ORANGE, "Fig 7 abrupt-morph onset")):
            vals = np.array([[r[key] for r in sub if r["value"] == v] for v in values], dtype=float)
            vals = np.where(np.isnan(vals), N_MAX + 2, vals)      # 'never' drawn above the top
            ax.fill_between(values, *np.percentile(vals, [25, 75], axis=1), color=color, alpha=0.2, lw=0)
            ax.plot(values, np.median(vals, axis=1), color=color, label=lab)
        holds = [np.mean([r["holds"] for r in sub if r["value"] == v]) for v in values]
        ax.axhspan(EARLY, LATE, color=GRID, alpha=0.6, lw=0)
        ax.axvline(DEFAULTS[name], color=INK, lw=1, ls=":")
        if logx:
            ax.set_xscale("log")
        ax.set(xlabel=label, ylim=(0, N_MAX + 3))
        frac = np.array(holds)
        ax.set_title("prediction holds: %d/%d values" % ((frac >= 0.5).sum(), len(values)), fontsize=8)
    axes[0].set_ylabel("pairs of observations")
    axes[2].legend(loc="upper left", fontsize=7)
    fig.suptitle("Sensitivity (median and IQR over seeds; grey band = between the early (5) and late (25) tests; "
                 "dotted = published value; top edge = never)", fontsize=9)
    fig.savefig(os.path.join(FIG, "sensitivity_1d.png"))
    plt.close(fig)


def run_grid(seeds, alphas, sds):
    grid = np.full((len(sds), len(alphas)), np.nan)
    holds = np.zeros_like(grid)
    for i, sd in enumerate(sds):
        for j, a in enumerate(alphas):
            res = [summarise(dict(DEFAULTS, alpha=a, sd=sd), s) for s in seeds]
            grid[i, j] = np.median([N_MAX + 2 if np.isnan(r["n_switch"]) else r["n_switch"] for r in res])
            holds[i, j] = np.mean([r["holds"] for r in res])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    la = np.log10(alphas)
    im = ax.pcolormesh(la, sds, grid, cmap=BLUES, shading="nearest", vmin=0, vmax=N_MAX + 2)
    ax.contour(la, sds, holds, levels=[0.5], colors=[ORANGE], linewidths=2)
    ax.plot(np.log10(DEFAULTS["alpha"]), DEFAULTS["sd"], "o", color=INK, ms=6)
    ax.set(xlabel="log10 alpha", ylabel="generative sd sigma",
           title="Onset of abrupt morph responses (pairs)\norange: prediction holds in >= half of seeds")
    fig.colorbar(im, ax=ax, label="pairs (top = never within %d)" % N_MAX)
    fig.savefig(os.path.join(FIG, "sensitivity_alpha_sd.png"))
    plt.close(fig)
    return grid, holds


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    band_seeds = range(1, 31 if args.quick else 201)
    sweep_seeds = range(1, 4 if args.quick else 11)
    orig = load_original()

    print("Fig 3 and Fig 7B ...")
    c3 = plot_fig3(band_seeds, orig)
    c7 = plot_fig7(band_seeds, orig)
    write_csv(os.path.join(RES, "fig3_seed1.csv"),
              [dict(pairs=i + 1, neg_partition_evidence_ratio=v) for i, v in enumerate(c3)])
    write_csv(os.path.join(RES, "fig7_seed1.csv"),
              [dict(probe=x, early_5=c7[EARLY][i], late_25=c7[LATE][i]) for i, x in enumerate(PROBES)])

    print("Fig 9A and appendix table ...")
    c9 = plot_fig9a()
    write_csv(os.path.join(RES, "fig9a_seed1.csv"),
              [dict(pairs=i + 1, **{"alpha_%.0e" % a: c9[a][i] for a in FIG9A_ALPHAS})
               for i in range(FIG9A_PAIRS)])
    write_csv(os.path.join(RES, "sensitivity_table.csv"), sensitivity_table())

    print("Seed robustness ...")
    seed_rows = [dict(seed=s, **summarise(DEFAULTS, s)) for s in band_seeds]
    write_csv(os.path.join(RES, "seeds.csv"), seed_rows)

    print("1-D sweeps ...")
    sweep_rows = run_sweeps(sweep_seeds)
    write_csv(os.path.join(RES, "sensitivity_1d.csv"), sweep_rows)
    plot_sweeps(sweep_rows)

    print("alpha x sd grid ...")
    alphas = np.logspace(-20, 2, 12)
    sds = np.linspace(0.1, 0.8, 5 if args.quick else 15)
    grid, holds = run_grid(range(1, 4 if args.quick else 6), alphas, sds)

    n_star = np.array([r["n_star"] for r in seed_rows])
    summary = {
        "defaults": DEFAULTS,
        "fig3_seed1_switch_pair": crossing(c3),
        "fig3_seeds_switch_pair_median": float(np.nanmedian(n_star)),
        "fig3_seeds_switch_pair_5_95": np.nanpercentile(n_star, [5, 95]).tolist(),
        "fig3_seeds_never_switch": int(np.isnan(n_star).sum()),
        "fig7_seed1_early_min": float(c7[EARLY].min()),
        "fig7_seed1_late_range": [float(c7[LATE].min()), float(c7[LATE].max())],
        "fig7_prediction_holds_fraction_of_seeds": float(np.mean([r["holds"] for r in seed_rows])),
        "n_seeds": len(seed_rows),
        "grid_alphas": alphas.tolist(), "grid_sds": sds.tolist(),
        "grid_holds_fraction": holds.tolist(),
    }
    for name, values, _, _ in SWEEPS:
        held = [v for v in values if np.mean([r["holds"] for r in sweep_rows
                                               if r["param"] == name and r["value"] == v]) >= 0.5]
        summary["holds_range_" + name] = [float(min(held)), float(max(held))] if held else None
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("grid")}, indent=1))


if __name__ == "__main__":
    main()
