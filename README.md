# geneML

## Basic Usage

```bash
pip install git+https://github.com/hexagonbio/geneML

geneml --help
```

## Development

### Setup

```bash
git clone https://github.com/hexagonbio/geneML
cd geneML

uv sync
```

### Usage

```bash
# Run commands through uv
uv run geneml --help
```

or

```bash
# Activate the env for the current terminal session
source .venv/bin/activate
geneml --help
```

### Linting

```bash
uv run ruff check --fix
```

### Release

CI is configured to automatically publish tags to PyPI and GitHub.

```bash
git tag v1.2.3
git push origin v1.2.3
```
