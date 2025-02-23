# tsnex

**t-SNEx** is a high-performance Python library for t-SNE, built on JAX for fast, scalable dimensionality reduction.
It utilizes JIT compilation, automatic differentiation, and hardware acceleration to efficiently process high-dimensional data for visualization and clustering.

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

## Citation
If you use `tsnex` in your research and need to reference it, please cite it as follows:
```
@software{alonso_tsnex,
  author = {Alonso, Albert},
  title = {tsnex: Minimal t-distributed stochastic neighbor embedding (t-SNE) implementation in JAX},
  url = {https://github.com/alonfnt/tsnex},
  version = {0.0.2}
}
```

## License
TSNEx is licensed under the MIT License. See the ![LICENSE](LICENSE) file for more details.

