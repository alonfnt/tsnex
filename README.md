# TSNEx

**TSNEx** is a lightweight, high-performance Python library for t-Distributed Stochastic Neighbor Embedding (t-SNE) built on top of JAX. Leveraging the power of JAX, `tsnex` offers JIT compilation, automatic differentiation, and hardware acceleration support to efficiently handle high-dimensional data for visualization and clustering tasks.

## Installation
Use the package manager [pip](https://pypi.org/project/tsnex/) to install `tsnex`.
```bash
pip install tsnex
```

## Usage
```python
import tsnex

# Generate some high-dimensional data
key = jax.random.key(0)
X = jax.random.normal(key, shape=(10_000, 50))

# Perform t-SNE dimensionality reduction
X_embedded = tsnex.transform(X, n_components=2)
```

## Contributing
We welcome contributions to **TSNEx**! Whether it's adding new features, improving documentation, or reporting issues, please feel free to make a pull request and/or open an issue.

## License
TSNEx is licensed under the MIT License. See the ![LICENSE](LICENSE) file for more details.

