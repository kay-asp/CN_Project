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
`compare_original.py` checks the two against each other.
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
