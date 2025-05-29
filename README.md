# geneML

## Basic Usage

```bash
pip add git+https://github.com/hexagonbio/geneML

geneml --help
```

## Development

Setup:

```bash
git clone https://github.com/hexagonbio/geneML
cd geneML

uv sync
```

Usage:

```bash
# Run commands through uv
uv run geneml --help

# Activate the env for the current terminal session
source .venv/bin/activate
geneml --help
```

Linting:

```bash
uv run ruff check --fix
```
