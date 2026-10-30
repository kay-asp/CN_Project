"""Independent reimplementation of the inference machinery of
Sanders, Wilson & Gershman (2020), "Hippocampal remapping as hidden state
inference", eLife 9:e51140.

Equation numbers refer to the published eLife Version of Record:
    Eq 2   normal-Wishart marginal likelihood of the observations in one state
    Eq 4   sequential CRP prior (where the next observation goes)
    Eq 5   CRP probability of a whole partition
    Eq 6   unnormalised partition posterior = Eq 5 + sum over states of Eq 2
    Eq 7   partition evidence ratio (log posterior odds between two partitions)
    Eq 8   posterior over the state a new observation (probe) belongs to
    Eq 10  posterior predictive, a multivariate Student-t (Murphy 2007, sec 8.3)
    Eq 11  state evidence ratio (log odds of an existing state vs a new state)

Everything is in log space. Written without reference to the authors' code;
`compare_original.py` checks the two against each other. The plotting
functions at the end draw the figures computed in `experiments.ipynb`.
"""
import numpy as np
from scipy.special import gammaln, logsumexp


def log_multigamma(a, d):
    """log of the d-dimensional multivariate gamma function Gamma_d(a)."""
    return d * (d - 1) / 4.0 * np.log(np.pi) + sum(gammaln(a - j / 2.0) for j in range(d))


class NormalWishart:
    """Gaussian observation model with unknown mean and covariance.

    Prior (Methods): mu0 = 0, kappa0 = 0.001, nu0 = 0.02, T0 = 0.02 * I.
    T is the scale matrix of the (inverse-)Wishart over the covariance, so it
    plays the role of a prior sum of squares.
    """

    def __init__(self, mu0=0.0, kappa0=1e-3, nu0=0.02, t0=0.02):
        self.mu0, self.kappa0, self.nu0, self.t0 = mu0, kappa0, nu0, t0

    def _posterior(self, Y):
        n, d = Y.shape
        mu0 = np.full(d, self.mu0, dtype=float)
        T0 = self.t0 * np.eye(d)
        ybar = Y.mean(axis=0)
        S = (Y - ybar).T @ (Y - ybar)                      # scatter matrix
        kappa_n = self.kappa0 + n
        nu_n = self.nu0 + n
        mu_n = (self.kappa0 * mu0 + n * ybar) / kappa_n
        diff = (ybar - mu0)[:, None]
        T_n = T0 + S + self.kappa0 * n / kappa_n * (diff @ diff.T)
        return mu_n, kappa_n, nu_n, T_n, T0

    def log_marginal(self, Y):
        """Eq 2: log p(Y) with mean and covariance integrated out."""
        Y = np.atleast_2d(Y)
        if Y.shape[0] == 0:
            return 0.0
        n, d = Y.shape
        _, kappa_n, nu_n, T_n, T0 = self._posterior(Y)
        return (-n * d / 2.0 * np.log(np.pi)
                + log_multigamma(nu_n / 2.0, d) - log_multigamma(self.nu0 / 2.0, d)
                + self.nu0 / 2.0 * np.linalg.slogdet(T0)[1]
                - nu_n / 2.0 * np.linalg.slogdet(T_n)[1]
                + d / 2.0 * np.log(self.kappa0 / kappa_n))

    def log_predictive(self, y, Y):
        """Eq 10: log p(y | Y), the Student-t posterior predictive.

        With no previous observations this is the prior predictive, which
        equals the marginal likelihood of the single point y.
        """
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        Y = np.atleast_2d(Y)
        if Y.size == 0:
            return self.log_marginal(y[None, :])
        d = y.size
        mu_n, kappa_n, nu_n, T_n, _ = self._posterior(Y)
        df = nu_n - d + 1
        scale = T_n * (kappa_n + 1) / (kappa_n * df)
        r = y - mu_n
        maha = r @ np.linalg.solve(scale, r)
        return (gammaln((df + d) / 2.0) - gammaln(df / 2.0)
                - d / 2.0 * np.log(df * np.pi) - 0.5 * np.linalg.slogdet(scale)[1]
                - (df + d) / 2.0 * np.log1p(maha / df))


def crp_log_prior_next(z, alpha):
    """Eq 4: log prior that the next observation joins each existing state
    (in index order 0..K-1) or opens a new state (last entry)."""
    z = np.asarray(z, dtype=int).ravel()
    counts = np.bincount(z).astype(float) if z.size else np.zeros(0)
    return np.log(np.append(counts, alpha) / (z.size + alpha))


def crp_log_prob(z, alpha):
    """Eq 5: log P(partition z) = K log a + log G(a) - log G(N + a) + sum_k log G(n_k)."""
    z = np.asarray(z, dtype=int).ravel()
    if z.size == 0:
        return 0.0
    counts = np.bincount(z)
    counts = counts[counts > 0]
    return (counts.size * np.log(alpha) + gammaln(alpha) - gammaln(z.size + alpha)
            + gammaln(counts).sum())


def _states(Y, z):
    z = np.asarray(z, dtype=int).ravel()
    return [Y[z == k] for k in range(z.max() + 1)] if z.size else []


def log_partition_posterior(Y, z, alpha, obs):
    """Eq 6: unnormalised log P(z | Y) = log P(z) + sum_k log p(Y_k)."""
    Y = np.atleast_2d(Y)
    return crp_log_prob(z, alpha) + sum(obs.log_marginal(Yk) for Yk in _states(Y, z))


def partition_evidence_ratio(Y, z_a, z_b, alpha, obs):
    """Eq 7: log P(z_a | Y) - log P(z_b | Y). Positive favours z_a."""
    return (log_partition_posterior(Y, z_a, alpha, obs)
            - log_partition_posterior(Y, z_b, alpha, obs))


def state_posterior(y, Y, z, alpha, obs):
    """Eq 8: normalised log P(probe y belongs to state k | Y, z), for each
    existing state k = 0..K-1 followed by a new state."""
    Y = np.atleast_2d(Y)
    loglik = [obs.log_predictive(y, Yk) for Yk in _states(Y, z)]
    loglik.append(obs.log_predictive(y, Y[:0]))
    post = crp_log_prior_next(z, alpha) + np.array(loglik)
    return post - logsumexp(post)


def state_evidence_ratio(y, Y, z, alpha, obs, k=None):
    """Eq 11: log P(state k | y) - log P(new state | y). With k=None the most
    probable existing state is used."""
    post = state_posterior(y, Y, z, alpha, obs)
    return (post[:-1].max() if k is None else post[k]) - post[-1]


def hypothesis_posterior(y, Y, partitions, alpha, obs):
    """Joint log posterior over (partition, probe assignment) pairs, used for
    the morph analysis (Fig 7). Returns one row per partition, each holding
    log P(partition, probe in state k | y, Y) for its existing states then a
    new state. Partition weights come from Eq 6, assignments from Eq 8."""
    lps = np.array([log_partition_posterior(Y, z, alpha, obs) for z in partitions])
    lps -= logsumexp(lps)
    return [lp + state_posterior(y, Y, z, alpha, obs) for lp, z in zip(lps, partitions)]


ORIGINAL_BURN_PER_PAIR = 400


def two_context_observations(n_pairs, sd=0.3, distance=2.0, seed=1, burn_per_pair=0):
    """Training stream used for Figs 3, 4 and 7: alternating 1-D observations
    from context A (mean -distance/2) and context B (mean +distance/2).

    Draw order and RNG (legacy RandomState, one standard normal per
    observation) match the released code. That code also draws 4 x 100
    uniform numbers after every pair (random activity patterns for novel
    states, used only for its place-cell readout); pass
    burn_per_pair=ORIGINAL_BURN_PER_PAIR to skip the same draws so that
    seed 1 reproduces its samples exactly.
    Returns Y (2*n_pairs x 1), the one-state partition and the true
    two-state partition.
    """
    rng = np.random.RandomState(seed)
    Y = np.empty((2 * n_pairs, 1))
    for i in range(n_pairs):
        Y[2 * i, 0] = -distance / 2.0 + sd * rng.standard_normal()
        Y[2 * i + 1, 0] = distance / 2.0 + sd * rng.standard_normal()
        rng.random_sample(burn_per_pair)
    return Y, np.zeros(2 * n_pairs, dtype=int), np.tile([0, 1], n_pairs)


# ---------------------------------------------------------------- plotting
# Each function draws one figure from values computed in experiments.ipynb,
# saves it to `path` and returns it so the notebook can show it inline.
# matplotlib is imported inside the functions so that compare_original.py,
# which only needs the model, does not depend on it.

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"
STYLE = {
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "lines.linewidth": 2, "font.size": 9, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
}


def _axes(figsize):
    import matplotlib.pyplot as plt
    plt.rcParams.update(STYLE)
    return plt.subplots(figsize=figsize)


def plot_fig2(names, ours, band, orig, path):
    """Fig 2: bars for seed 1, the median and 5-95% range over seeds, and the
    released code's values (if given) as crosses."""
    lo, med, hi = np.percentile(band, [5, 50, 95], axis=0)
    x = np.arange(len(names))
    fig, ax = _axes((5.2, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    ax.bar(x, ours, width=0.6, color=BLUE, alpha=0.8, label="this code, seed 1")
    ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="o", color=INK, ms=4, capsize=4,
                label="%d seeds, median and 5-95%%" % len(band))
    if orig is not None:
        ax.plot(x, orig, "x", color=ORANGE, ms=9, mew=2, label="released code, seed 1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set(ylabel="state evidence ratio (Eq 11)")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path)
    return fig


def plot_fig3b(x, one, two, points, path):
    """Fig 3B: posterior predictive densities under one state and two states."""
    fig, ax = _axes((4.5, 3.4))
    ax.plot(x, one, color=ORANGE, label="1 state")
    ax.plot(x, two[0], color=BLUE, label="2 states")
    ax.plot(x, two[1], color=BLUE)
    ax.plot(points, np.zeros(len(points)), "o", ms=4, color=INK, label="observations", clip_on=False)
    ax.set(xlabel="feature value", ylabel="predictive density")
    ax.legend(loc="upper left")
    fig.savefig(path)
    return fig


def plot_fig3c(n, curve, band, orig, path):
    """Fig 3C: -partition evidence ratio against experience for seed 1, with
    the 5-95% band and median over seeds and the released code's values."""
    fig, ax = _axes((4.5, 3.4))
    ax.axhline(0, color=MUTED, lw=1)
    ax.fill_between(n, *np.percentile(band, [5, 95], axis=0), color=BLUE, alpha=0.15, lw=0,
                    label="%d seeds, 5-95%%" % len(band))
    ax.plot(n, np.median(band, axis=0), color=BLUE, lw=1, ls="--", label="%d seeds, median" % len(band))
    ax.plot(n, curve, color=BLUE, marker="o", ms=4, label="this code, seed 1")
    if orig is not None:
        ax.plot(n, orig, "x", color=ORANGE, ms=8, mew=2, label="released code, seed 1")
    ax.set(xlabel="number of experiences (pairs)", ylabel="log P(2 states) - log P(1 state)", ylim=(-25, 25))
    ax.text(n[-1], 22, "favours remapping", ha="right", va="top", color=MUTED)
    ax.text(n[-1], -22, "favours one map", ha="right", va="bottom", color=MUTED)
    ax.legend(loc="upper left")
    fig.savefig(path)
    return fig


def plot_fig7(probes, curves, bands, origs, path):
    """Fig 7B: morph evidence ratio after each training length. `curves`,
    `bands` and `origs` are dicts keyed by the number of training pairs; the
    first key is drawn in orange and the second in blue."""
    fig, ax = _axes((5.2, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    last = list(curves)[-1]
    for n, color in zip(curves, (ORANGE, BLUE)):
        band = bands[n]
        ax.fill_between(probes, *np.percentile(band, [5, 95], axis=0), color=color, alpha=0.15, lw=0,
                        label="%d seeds, 5-95%%" % len(band) if n == last else None)
        ax.plot(probes, np.median(band, axis=0), color=color, lw=1, ls="--",
                label="%d seeds, median" % len(band) if n == last else None)
        ax.plot(probes, curves[n], color=color, marker="o", ms=4, label="after %d pairs (seed 1)" % n)
        if origs is not None:
            ax.plot(probes, origs[n], "x", color=INK, ms=7, mew=1.5,
                    label="released code" if n == last else None)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["square\n(-1)", "-0.5", "0", "0.5", "circle\n(+1)"])
    ax.set(xlabel="morph probe", ylabel="log odds: same as square vs different")
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: ("seeds" not in labels[i], "released" in labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left")
    fig.savefig(path)
    return fig


def plot_fig9a(n, curves, path):
    """Fig 9A: the Fig 3C curve for several values of alpha (dict alpha -> curve)."""
    fig, ax = _axes((5.2, 3.6))
    ax.axhline(0, color=MUTED, lw=1)
    for a, color in zip(curves, (ORANGE, BLUE, INK)):
        ax.plot(n, curves[a], color=color, marker="o", ms=3, label="alpha = %.0e" % a)
    ax.set(xlabel="number of experiences (pairs)", ylabel="log P(2 states) - log P(1 state)",
           title="Fig 9A  Negative partition evidence ratio for three alphas", ylim=(-25, 25))
    ax.legend(loc="upper left")
    fig.savefig(path)
    return fig
