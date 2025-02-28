from functools import partial
from typing import Callable
import jax
import jax.numpy as jnp

import pcax


def kl_divergence(p, q, *, eps=1e-12):
    return jnp.sum(p * jnp.log((p+eps) / (q+eps)))


def euclidean_distance(x, y):
    return jnp.sum((x - y) ** 2, axis=-1)


def shannon_entropy(p, *, eps=1e-12):
    return -jnp.sum(p * jnp.log2(p + eps))


def perplexity_fun(p):
    return 2 ** shannon_entropy(p)


def _conditional_probability(distances, sigma, *, eps=1e-12):
    p = jnp.exp(-distances / (2 * sigma**2))
    p = p.at[jnp.diag_indices(p.shape[0])].set(0)
    p = p / (jnp.sum(p, axis=1, keepdims=True) + eps)
    return p


def _joint_probability(distances, sigma):
    p = _conditional_probability(distances, sigma)
    p = (p + p.T) / (2 * p.shape[0])
    return p


def _binary_search_perplexity(distances, target, tol=1e-5, max_iter=200, sigma_min=1e-10, sigma_max=1e10):

    sigma = 1.0

    def cond_fun(val):
        (_, perplexity, i, _, _) = val
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

def all_to_all(f: Callable, batch_size: int | None = None) -> Callable:
    """
    Transforms a scalar function f(x, y) -> scalar into one that operates over two arrays.

    Args:
        f: A function f(x, y) -> scalar.
        batch_size: Optional batch size for batching.

    Returns:
        A function g(x, y) -> matrix, where g applies f to each pair (xi, yj).
    """
    def g(x, y):
        return jax.lax.map(
            lambda i: jax.vmap(f, in_axes=(None, 0))(i, y),
            x,
            batch_size=batch_size
        )
    return jax.jit(g)


@partial(
    jax.jit,
    static_argnames=[
        "n_components",
        "perplexity",
        "learning_rate",
        "init",
        "seed",
        "n_iter",
        "metric_fn",
        "early_exageration",
        "batch_size",
    ],
)
def transform(
    X: jax.Array,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 1e-3,
    init: str = "pca",
    seed: int = 0,
    n_iter: int = 1000,
    metric_fn: Callable = euclidean_distance,
    early_exageration: float = 12.0,
    batch_size: int | None = None,
) -> jax.Array:
    """
    Transforms X to a lower-dimensional representation using t-SNE.

    Args:
        X: The input data.
        n_components: The number of output components.
        perplexity: The perplexity of the distribution.
        learning_rate: The learning rate of the optimizer.
        init: The initialization method ("pca" or "random").
        seed: The random seed.
        n_iter: The number of optimization steps.
        metric_fn: The metric function (defaults to euclidean_distance).
        early_exageration: The early exaggeration factor.
        batch_size: Optional batch size.

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

    # Compute the probability of neighbours on the original embedding.
    # The matrix needs to be symetrized in order to be used as joint probability.
    batch_size = batch_size or len(X)
    _metric_fn = all_to_all(metric_fn, batch_size=batch_size)
    distances = _metric_fn(X, X)
    sigma = _binary_search_perplexity(distances, perplexity)
    P = _joint_probability(distances, sigma)

    @jax.grad
    def loss_fn(x, P):
        distances = _metric_fn(x, x)
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
