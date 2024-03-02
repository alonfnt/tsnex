import jax
import jax.numpy as jnp

import pcax


def kl_divergence(p, q):
    return jnp.sum(p * jnp.log(p / q))


def euclidean_distance(x, y):
    return jnp.sum((x - y) ** 2, axis=-1)


def probability_fn(x, y, sigma):
    return jnp.exp(-euclidean_distance(x, y) / (2 * sigma**2))


def transform(
    X,
    *,
    n_components=2,
    perplexity=30.0,
    learning_rate=1e-3,
    init="pca",
    seed=0,
    n_iter=1000,
    metric_fn=None,
):
    """
    Transform X to a lower dimensional representation using T-distributed Stochastic Neighbor Embedding.


    Args:

    Returns:
        - X_new: jax.Array, shape (n_samples, n_components)
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
    vmapped_prob = jax.vmap(
        jax.vmap(probability_fn, in_axes=(0, None, None)), in_axes=(None, 0, None)
    )

    P = vmapped_prob(X, X, perplexity)

    @jax.grad
    def loss_fn(x):
        distances = metric_fn(x, x)
        Q = jax.nn.softmax(-distances)
        return kl_divergence(P, Q)

    def train_step(x, _):
        grads = loss_fn(x)
        x_new = x - learning_rate * grads
        return x_new, None

    n_exageration = 250
    X_new, _ = jax.lax.scan(train_step, X_new, xs=None, length=n_iter + n_exageration)

    return X_new
