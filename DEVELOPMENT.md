# Development Guide

## Prerequisites

- Python 3.10+
- uv (Python package manager)
- Docker (for containerized development)

## Development Setup

### 1. Install uv

```bash
# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Setup Development Environment

```bash
# Clone the repository
git clone <repository-url>
cd femtotransformers

# Install the package in development mode with all extras
make install-dev

# Setup pre-commit hooks
make setup-dev
```

## Development Workflow

### Code Quality

We use ruff and mypy for code quality:

```bash
# Format code
make format

# Lint code (with auto-fix)
make lint

# Type checking
make type-check

# All quality checks
make quality
```

### Testing

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run example tests
make test-examples
```

### Quick Development Fixes

```bash
# Quick fix for modified files only
make fixup

# Full style and formatting
make style
```

## Project Structure

femtotransformers/
├── src/femtotransformers/          # Main package
├── tests/                          # Test files
├── examples/                       # Example scripts
├── utils/                          # Development utilities
├── docker/                         # Docker configurations
├── pyproject.toml                  # Project configuration
├── Makefile                        # Development commands
└── .pre-commit-config.yaml         # Pre-commit hooks

## Configuration

All project configuration is centralized in `pyproject.toml`:

- **Build system**: Hatchling
- **Dependencies**: Modern versions with Python 3.12 support
- **Code quality**: Ruff for linting/formatting, mypy for type checking
- **Testing**: pytest with coverage
- **Tool configurations**: All tools configured in pyproject.toml

## Docker Development

Build and run the development container:

```bash
# Build the Docker image
make docker-build

# Run the container
make docker-run
```

The Docker image is based on the latest NVIDIA PyTorch container (`25.06-py3`) and includes:

- Python 3.10+
- PyTorch 2.5+
- All project dependencies
- Development tools (ruff, mypy, pytest)

## CI/CD

The project uses GitHub Actions for continuous integration:

- **Linting**: Runs ruff check and format verification
- **Type checking**: Runs mypy on the source code
- **Testing**: Runs pytest with coverage reporting
- **Building**: Creates distribution packages

## Dependency Management

Dependencies are managed through `pyproject.toml`:

```bash
# Update dependencies
make deps-update

# Sync environment with lock file
make deps-sync

# Update dependency table
make deps-table-update
```

## Migration from Old Setup

This project has been migrated from:

- `setup.py` → `pyproject.toml`
- `requirements.txt` → `pyproject.toml` dependencies
- `pytest.ini` → `pyproject.toml` pytest configuration
- `pip` → `uv` for package management
- Mixed tooling → `ruff` + `mypy` for all code quality

## Best Practices

1. **Use pre-commit hooks**: Run `make setup-dev` to install
2. **Test before committing**: Run `make quality` and `make test`
3. **Update dependencies**: Regularly run `make deps-update`
4. **Follow type hints**: Use mypy-compatible type annotations
5. **Format code**: Use `make format` before committing
