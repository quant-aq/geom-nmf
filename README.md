# geo-nmf

![Tests](https://github.com/quant-aq/geo-nmf/actions/workflows/tests.yml/badge.svg)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)

Coming soon!

# Getting Started


## Installation

Install the latest version directly from GitHub:

```sh
pip install git+https://github.com/quant-aq/geo-nmf.git
```

Install from a specific branch:

```sh
pip install git+https://github.com/quant-aq/geo-nmf.git@<branch-name>
```

Install from a specific tag:

```sh
pip install git+https://github.com/quant-aq/geo-nmf.git@<tag>
```

## Contributing / Development Setup

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments. Follow the steps below to get your local development environment set up.

### 1. Install Poetry

If you don't already have Poetry installed, choose the instructions for your operating system below.

**macOS**

The recommended way on macOS is via [Homebrew](https://brew.sh/). If you don't have Homebrew installed, install it first:

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Poetry:

```sh
brew install poetry
```

**Linux / Windows (WSL)**

```sh
curl -sSL https://install.python-poetry.org | python3 -
```

Then follow any instructions it prints to add Poetry to your `PATH`.

**Verify the installation**

```sh
poetry --version
```

### 2. Clone the repository

```sh
git clone https://github.com/quant-aq/geo-nmf.git
cd geo-nmf
```

### 3. Install dependencies

This installs both the main dependencies and the development dependencies (e.g. pytest, sphinx) into an isolated virtual environment managed by Poetry:

```sh
poetry install
```

To also install the optional coverage tools:

```sh
poetry install --with coverage
```

You only need to run this once (and again whenever dependencies change).

### 4. Activate the virtual environment

Poetry creates and manages a virtual environment for you. To start a shell inside it:

```sh
poetry shell
```

Your prompt will change to indicate you're inside the environment. Any commands you run (like `pytest` or `python`) will use the project's isolated dependencies, not your system Python.

To exit the environment when you're done:

```sh
exit
```

Alternatively, you can run a single command inside the environment without activating it, using `poetry run`:

```sh
poetry run pytest tests/
```

### 5. Adding dependencies

To add a new dependency:

```sh
# Runtime dependency
poetry add <package>

# Development-only dependency
poetry add --group dev <package>
```

### 6. Verify everything is working

Run the test suite to confirm your setup is correct:

```sh
poetry run pytest tests/
```

All tests should pass before you start making changes.

## Building Documentation

Instructions will eventually go here...

## Running Tests

Tests are built and run using [pytest](https://docs.pytest.org/en/7.1.x/). Test files should be named with the `test_` prefix (e.g. `test_basics.py`) and located in the `tests/` directory.

Run all tests:

```sh
pytest tests/
```

Common flags:

| Flag | Description |
|------|-------------|
| `-v` | Verbose output — shows each test name and pass/fail status |
| `-q` | Quiet output — minimal output, just a summary |
| `-x` | Stop after the first failure |
| `-s` | Disable output capture — allows `print()` statements to show in terminal |
| `-k "expression"` | Run only tests whose names match the expression (e.g. `-k "test_fit"`) |
| `--tb=short` | Show a shortened traceback on failures (`short`, `long`, `no`, `line`) |
| `--cov=geo_nmf` | Report test coverage for the package (requires `pytest-cov`) |

Examples:

```sh
# Run all tests verbosely
pytest -v tests/

# Stop on first failure with full traceback
pytest -x --tb=long tests/

# Run only tests matching a keyword
pytest -v -k "test_fit" tests/

# Run with coverage report
pytest --cov=geo_nmf --cov-report=term-missing tests/
```

&copy; 2026, QuantAQ, Inc. All rights reserved.