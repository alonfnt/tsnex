import jax
import jax.numpy as jnp
import pytest
import tsnex


@pytest.mark.parametrize(
    "p,q,expected",
    [
        (
            jnp.array([0.1, 0.9]),
            jnp.array([0.9, 0.1]),
            jnp.sum(jnp.array([0.1, 0.9]) * jnp.log(jnp.array([0.1, 0.9]) / jnp.array([0.9, 0.1]))),
        ),
        (jnp.array([0.5, 0.5]), jnp.array([0.5, 0.5]), 0),
    ],
)
def test_kl_divergence(p, q, expected):
    result = tsnex.tsne.kl_divergence(p, q)
    assert jnp.isclose(result, expected, atol=1e-6)


@pytest.mark.parametrize(
    "x,y,expected",
    [
        (jnp.array([0, 0]), jnp.array([1, 1]), 2),
        (jnp.array([1, 2, 3]), jnp.array([4, 5, 6]), 27),
    ],
)
def test_euclidean_distance(x, y, expected):
    result = tsnex.tsne.euclidean_distance(x, y)
    assert jnp.isclose(result, expected)



@pytest.mark.parametrize(
    "x, sigma, expected",
    [
        (
            jnp.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]),
            jnp.array([2.0, 2.0, 2.0]),
            jnp.array([
                [0.0, 0.82, 0.18],
                [0.5, 0.0, 0.5],
                [0.18, 0.82, 0.0],
            ]),
        ),
    ],
)
def test_probability_fn(x, sigma, expected):
    sum_x = jnp.sum(jnp.square(x), axis=1, keepdims=True)
    distances = sum_x + sum_x.T - 2 * jnp.dot(x, x.T)
    result = tsnex.tsne._conditional_probability(distances, sigma)
    assert jnp.allclose(result, expected, atol=1e-2)

@pytest.mark.parametrize("init_method", ["pca", "random"])
def test_transform_shape_and_locality(init_method):
    key = jax.random.key(0)
    X = jax.random.normal(key, shape=(100, 50))
    X_transformed = tsnex.transform(X, init=init_method, seed=42, n_iter=10)
    assert X_transformed.shape == (100, 2)


def test_transform_invalid_init():
    key = jax.random.key(0)
    with pytest.raises(ValueError):
        tsnex.transform(jax.random.uniform(key, shape=(10, 5)), init="invalid_init")
    
def test_nan_output():
    key = jax.random.key(0)
    X = jax.random.normal(key, shape=(100, 50))
    X_transformed = tsnex.transform(X, n_components=2)
    assert not jnp.isnan(X_transformed).any()
