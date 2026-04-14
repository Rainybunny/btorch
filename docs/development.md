# Development

Install precommit hooks for auto formatting.

PR without precommit formatting will not be accepted!

```bash
pre-commit install --install-hooks
```

Highly recommended to use [jaxtyping](https://docs.kidger.site/jaxtyping/) to mark expected array shape,
see [good example of using jaxtyping](https://fullstackdeeplearning.com/blog/posts/rwkv-explainer)

When using AI coding assistants, pls use [`desloppify`](https://github.com/peteromallet/desloppify) to verify code quality before submitting PRs. Consider installing it as a pre-push git hook to catch issues early.

## Run the tests

```bash
ruff check .
pytest tests
python -m sphinx.cmd.build docs docs/_build/html
```
