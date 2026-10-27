"""Internal-consistency checks for sanders_model.py. Run with `pytest -q`."""
import numpy as np
from scipy import integrate, stats

from sanders_model import (NormalWishart, crp_log_prior_next, crp_log_prob,
                           hypothesis_posterior, state_posterior,
                           two_context_observations)


def test_marginal_matches_numerical_integration_1d():
    # 1-D normal-Wishart = normal-gamma: precision lam ~ Gamma(nu0/2, rate=T0/2),
    # mu | lam ~ N(mu0, 1/(kappa0 lam)). A proper prior keeps the integral finite.
    obs = NormalWishart(mu0=0.0, kappa0=1.0, nu0=3.0, t0=1.0)
    Y = np.array([[-0.4], [0.3], [0.9]])

    def integrand(mu, lam):
        prior = (stats.gamma.pdf(lam, a=1.5, scale=2.0)
                 * stats.norm.pdf(mu, 0.0, 1 / np.sqrt(lam)))
        return prior * np.prod(stats.norm.pdf(Y[:, 0], mu, 1 / np.sqrt(lam)))

    val, _ = integrate.dblquad(integrand, 0, 60, -15, 15)
    assert np.isclose(obs.log_marginal(Y), np.log(val), atol=1e-6)


def test_chain_rule_predictive_vs_marginal():
    # log p(y_1..y_n) must equal the sum of sequential posterior predictives
    rng = np.random.RandomState(0)
    for d in (1, 2):
        obs = NormalWishart(nu0=d + 1.0, t0=0.5)
        Y = rng.normal(size=(8, d))
        chain = sum(obs.log_predictive(Y[i], Y[:i]) for i in range(len(Y)))
        assert np.isclose(chain, obs.log_marginal(Y))
    # also holds for the paper's (improper-looking) hyperparameters in 1-D
    obs = NormalWishart()
    Y = rng.normal(size=(6, 1))
    chain = sum(obs.log_predictive(Y[i], Y[:i]) for i in range(len(Y)))
    assert np.isclose(chain, obs.log_marginal(Y))


def test_crp_partition_prob_equals_sequential_prior():
    alpha = 0.37
    z = np.array([0, 0, 1, 0, 2, 1, 1])
    seq = sum(crp_log_prior_next(z[:i], alpha)[z[i]] for i in range(len(z)))
    assert np.isclose(seq, crp_log_prob(z, alpha))


def test_crp_sums_to_one_over_all_partitions_of_three():
    alpha = 2.5
    parts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2]]
    assert np.isclose(sum(np.exp(crp_log_prob(z, alpha)) for z in parts), 1.0)


def test_posteriors_are_normalised():
    Y, z1, z2 = two_context_observations(4)
    obs = NormalWishart()
    assert np.isclose(np.exp(state_posterior([0.2], Y, z2, 0.1, obs)).sum(), 1.0)
    joint = np.concatenate(hypothesis_posterior([0.2], Y, [z1, z2], 0.1, obs))
    assert np.isclose(np.exp(joint).sum(), 1.0)
