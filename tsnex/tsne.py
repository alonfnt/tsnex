from typing import Callable
import jax
import jax.numpy as jnp

import pcax


def kl_divergence(p, q):
    return jnp.sum(p * jnp.log(p / q))


def euclidean_distance(x, y):
    return jnp.sum((x - y) ** 2, axis=-1)


def shannon_entopy(p, eps=1e-12):
    return -jnp.sum(p * jnp.log2(p + eps))


def perplexity_fun(p):
    return 2 ** shannon_entopy(p)


def _conditional_probability(distances, sigma):
    p = jnp.exp(-distances / (2 * sigma**2))
    p = p / jnp.sum(p)
    return p


def _joint_probability(distances, sigma):
    p = _conditional_probability(distances, sigma)
    p = (p + p.T) / (2 * p.shape[0])
    return p


def _binary_search_perplexity(distances, target, tol=1e-5, max_iter=200):

    sigma_min = 1e-20
    sigma_max = 1e20
    sigma = 1.0

    def cond_fun(val):
        (sigma, perplexity, i, sigma_min, sigma_max) = val
        return (jnp.abs(perplexity - target) > tol) & (i < max_iter)

    def body_fun(val):
        (sigma, perp, i, sigma_min, sigma_max) = val
        p = _conditional_probability(distances, sigma)
        perp = perplexity_fun(p)
        sigma = jnp.where(perp > target, (sigma + sigma_min) / 2, (sigma + sigma_max) / 2)
        sigma_min = jnp.where(perp > target, sigma_min, sigma)
        sigma_max = jnp.where(perp > target, sigma, sigma_max)
        return (sigma, perp, i + 1, sigma_min, sigma_max)

    p = _conditional_probability(distances, sigma)
    perplexity = perplexity_fun(p)
    init_val = (sigma, perplexity, 0, sigma_min, sigma_max)
    sigma = jax.lax.while_loop(cond_fun, body_fun, init_val)[0]
    return sigma


def transform(
    X: jax.Array,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 1e-3,
    init: str = "pca",
    seed: int = 0,
    n_iter: int = 1000,
    metric_fn: Callable = None,
    early_exageration: float = 12.0,
) -> jax.Array:
    """
    Transform X to a lower dimensional representation using T-distributed Stochastic Neighbor Embedding.

    Args:
        X: The input data.
        n_components: The number of components of the output.
        perplexity: The perplexity of the conditional probability distribution.
        learning_rate: The learning rate of the optimization algorithm.
        init: The initialization method. Can be "pca" or "random".
        seed: The seed for the random number generator.
        n_iter: The number of iterations.
        metric_fn: The metric function to use. By default, it uses the euclidean distance.
        early_exageration: The early exageration factor.

    Returns:
        The transformed data.
    """
    if init == "pca":
        state = pcax.fit(X, n_components)
        X_new = pcax.transform(state, X)
    elif init == "random":
        X_new = jax.random.normal(jax.random.key(seed), (X.shape[0], n_components))
    else:
        raise ValueError(f"Unknown init_method: {init}")

    if metric_fn is None:
        metric_fn = euclidean_distance
    metric_fn = jax.vmap(jax.vmap(metric_fn, in_axes=(0, None)), in_axes=(None, 0))

    # Compute the probability of neighbours on the original embedding.
    # The matrix needs to be symetrized in order to be used as joint probability.
    distances = metric_fn(X, X)
    sigma = _binary_search_perplexity(distances, perplexity)
    P = _joint_probability(distances, sigma)

    @jax.grad
    def loss_fn(x, P):
        distances = metric_fn(x, x)
        Q = jax.nn.softmax(-distances)
        return kl_divergence(P, Q)

    def train_step(x, _):
        grads = loss_fn(x, P)
        x_new = x - learning_rate * grads
        return x_new, None

    def train_step_early_exageration(x, _):
        grads = loss_fn(x, early_exageration * P)
        x_new = x - learning_rate * grads
        return x_new, None

    n_exageration = 250
    X_new, _ = jax.lax.scan(train_step_early_exageration, X_new, xs=None, length=n_exageration)

    X_new, _ = jax.lax.scan(train_step, X_new, xs=None, length=n_iter)

    return X_new
