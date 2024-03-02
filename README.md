# tsnex: Minimal t-SNEx implementation in JAX

**tsnex** is a lightweight, high-performance Python library for t-Distributed Stochastic Neighbor Embedding (t-SNE) built on top of JAX. Leveraging the power of JAX, `tsnex` offers JIT compilation, automatic differentiation, and hardware acceleration support to efficiently handle high-dimensional data for visualization and clustering tasks.

| [**Usage**](#usage)
[**Installation**](#installation)
| [**Contributing**](#contributing)
| [**License**](#license)

## Usage<a id="usage"></a>
```python

    key = jax.random.key(0)
    X = jax.random.normal(key, shape=(100, 50))

    X_embedded = tsnex.transform(X, n_components=2)
```

## Installation<a id="installation"></a>
`tsnex` can be installed using [PyPI](https://pypi.org/project/tsnex/) via `pip`:
```
pip install tsnex
```
or from GitHub directly
```
pip install git+git://github.com/alonfnt/tsnex.git
```

Likewise, you can clone this repository and install it locally

```bash
git clone https://github.com/alonfnt/tsnex.git
cd tsnex
pip install -r requirements.txt
```

## Contributing<a id="contributing"></a>
We welcome contributions to **tsnex**! Whether it's adding new features, improving documentation, or reporting issues, please feel free to make a pull request or open an issue.

## License<a id="license"></a>
Bayex is licensed under the MIT License. See the ![LICENSE](LICENSE) file for more details.

