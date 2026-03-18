# geom-nmf

![Tests](https://github.com/quant-aq/geom-nmf/actions/workflows/tests.yml/badge.svg)
![Python Versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)

Coming soon!

# Getting Started

## Installation

Install the latest version directly from GitHub:

```sh
pip install git+https://github.com/quant-aq/geom-nmf.git
```

Install from a specific branch:

```sh
pip install git+https://github.com/quant-aq/geom-nmf.git@<branch-name>
```

Install from a specific tag:

```sh
pip install git+https://github.com/quant-aq/geom-nmf.git@<tag>
```

## Contributing

We welcome contributions! If you'd like to fix a bug, add a feature, or improve documentation, follow the steps below. Don't worry if you're new to this — it's a straightforward process once you've done it a couple of times.

This project uses [Poetry](https://python-poetry.org/) to manage dependencies and virtual environments.

### 1. Fork the repository

A "fork" is your own personal copy of the project on GitHub. To create one, click the **Fork** button in the top-right corner of the [geom-nmf repository](https://github.com/quant-aq/geom-nmf) page.

### 2. Clone your fork

Download your fork to your local machine:

```sh
git clone https://github.com/<your-github-username>/geom-nmf.git
cd geom-nmf
```

### 3. Install Poetry

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

### 4. Install dependencies

This installs both the main dependencies and the development dependencies (e.g. pytest, sphinx) into an isolated virtual environment managed by Poetry:

```sh
poetry install
```

To also install the optional coverage tools:

```sh
poetry install --with coverage
```

You only need to run this once (and again whenever dependencies change).

To activate the virtual environment:

```sh
poetry shell
```

Your prompt will change to indicate you're inside the environment. To exit when you're done:

```sh
exit
```

Alternatively, prefix any command with `poetry run` to run it inside the environment without activating it:

```sh
poetry run pytest tests/
```

### 5. Create a branch

Never make changes directly on `main`. Instead, create a new branch with a short descriptive name:

```sh
git checkout -b my-feature-branch
```

### 6. Make your changes

Edit files, write code, fix bugs — whatever your contribution involves. Keep your changes focused on one thing per branch.

### 7. Run the tests

Before submitting, make sure all tests pass:

```sh
poetry run pytest tests/
```

If you've added new functionality, add a corresponding test in the `tests/` directory.

Common flags:

| Flag | Description |
|------|-------------|
| `-v` | Verbose output — shows each test name and pass/fail status |
| `-q` | Quiet output — minimal output, just a summary |
| `-x` | Stop after the first failure |
| `-s` | Disable output capture — allows `print()` statements to show in terminal |
| `-k "expression"` | Run only tests whose names match the expression (e.g. `-k "test_fit"`) |
| `--tb=short` | Show a shortened traceback on failures (`short`, `long`, `no`, `line`) |
| `--cov=geom_nmf` | Report test coverage for the package (requires `pytest-cov`) |

Examples:

```sh
# Run all tests verbosely
pytest -v tests/

# Stop on first failure with full traceback
pytest -x --tb=long tests/

# Run only tests matching a keyword
pytest -v -k "test_fit" tests/

# Run with coverage report
pytest --cov=geom_nmf --cov-report=term-missing tests/
```

### 8. Commit and push your changes

Stage and commit your changes with a clear message describing what you did:

```sh
git add <file(s) you changed>
git commit -m "Brief description of your change"
git push origin my-feature-branch
```

### 9. Open a Pull Request

Go to your fork on GitHub. You should see a prompt to open a Pull Request — click it. Fill in a short description of what your changes do and why. A maintainer will review your PR and may leave comments or request changes before merging.

### Tips

- Keep pull requests small and focused — one feature or fix per PR is much easier to review
- Check that your branch is up to date with `main` before opening a PR:
  ```sh
  git fetch origin
  git rebase origin/main
  ```
- To add a new dependency: `poetry add <package>` (or `poetry add --group dev <package>` for dev-only)
- If you're unsure about a change, open an issue first to discuss it before writing code

## Building Documentation

Instructions will eventually go here...

&copy; 2026, QuantAQ, Inc. All rights reserved.
