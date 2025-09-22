# Contributing to TIDE-Lite

Thank you for your interest in contributing to TIDE-Lite! This document provides guidelines and conventions for contributing to the project.

## Code of Conduct

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what is best for the project and community

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/DynamicEmbeddings.git
   cd DynamicEmbeddings
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set up pre-commit hooks** (optional but recommended)
   ```bash
   pre-commit install
   ```

## Code Standards

### Python Version
- Minimum Python 3.10
- Use modern Python features (type hints, dataclasses, etc.)

### Code Style
- Follow PEP 8 with 100-character line limit
- Use Black for formatting
- Use isort for import sorting
- Use ruff for linting

### Type Hints
- **Required** for all function signatures
- Use `from typing import` for type annotations
- Example:
  ```python
  from typing import Optional, List, Dict, Any
  
  def process_data(
      inputs: List[str],
      config: Optional[Dict[str, Any]] = None
  ) -> torch.Tensor:
      """Process input data."""
      ...
  ```

### Docstrings
- Use Google-style docstrings
- Required for all public functions, classes, and modules
- Example:
  ```python
  def train_model(
      model: nn.Module,
      dataloader: DataLoader,
      config: TIDEConfig
  ) -> Dict[str, float]:
      """Train the model for one epoch.
      
      Args:
          model: Model to train
          dataloader: Training data loader
          config: Training configuration
          
      Returns:
          Dictionary of training metrics
          
      Raises:
          ValueError: If config is invalid
      """
      ...
  ```

### Logging
- Use logging module, **never print()**
- Get logger at module level:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```

## Commit Style

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes

### Examples
```bash
# Feature
git commit -m "feat(models): add temporal consistency loss"

# Bug fix
git commit -m "fix(data): handle missing timestamps in STS-B"

# Documentation
git commit -m "docs(readme): add Colab quickstart guide"

# Refactoring
git commit -m "refactor(utils): simplify config validation logic"
```

### Scope
Common scopes:
- `models`: Model architecture changes
- `data`: Data loading/preprocessing
- `train`: Training logic
- `eval`: Evaluation metrics
- `utils`: Utility functions
- `cli`: Command-line interface
- `config`: Configuration management
- `tests`: Test files

## Testing Conventions

### Test Structure
```
tests/
├── unit/           # Unit tests
│   ├── test_models.py
│   ├── test_data.py
│   └── test_utils.py
├── integration/    # Integration tests
│   ├── test_training.py
│   └── test_evaluation.py
└── fixtures/       # Test data and fixtures
```

### Writing Tests
- Use pytest for all tests
- Test file names: `test_<module>.py`
- Test function names: `test_<functionality>`
- Use fixtures for common setup
- Aim for >80% code coverage

### Example Test
```python
import pytest
import torch
from tide_lite.models import TIDELite

@pytest.fixture
def model():
    """Create a test model."""
    return TIDELite(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        time_mlp_hidden=64
    )

def test_forward_pass(model):
    """Test model forward pass."""
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    timestamps = torch.tensor([0.0, 86400.0])
    
    output = model(input_ids, attention_mask, timestamps)
    
    assert output.shape == (batch_size, 384)
    assert not torch.isnan(output).any()
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tide_lite --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_models.py

# Run with verbose output
pytest -v
```

## Code Review Process

### Before Submitting PR
1. **Run formatters**
   ```bash
   black src/ tests/
   isort src/ tests/
   ```

2. **Run linters**
   ```bash
   ruff check src/ tests/
   mypy src/
   ```

3. **Run tests**
   ```bash
   pytest
   ```

4. **Update documentation** if needed

### Pull Request Guidelines
1. **One feature per PR** - Keep PRs focused
2. **Clear description** - Explain what and why
3. **Link issues** - Reference related issues
4. **Add tests** - New features need tests
5. **Update docs** - Keep documentation current

### Review Checklist
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Type hints are present
- [ ] No print statements (use logging)
- [ ] Commit messages follow convention

## Project Structure

Maintain single responsibility principle:

```
src/tide_lite/
├── data/       # Data loading ONLY
├── models/     # Model architectures ONLY
├── train/      # Training logic ONLY
├── eval/       # Evaluation metrics ONLY
├── cli/        # CLI interfaces ONLY
├── utils/      # Shared utilities
└── plots/      # Visualization tools
```

## Documentation

### Code Documentation
- Docstrings for all public APIs
- Type hints for clarity
- Comments for complex logic

### User Documentation
- Update README.md for new features
- Add examples to notebooks/
- Update CLI help text

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`

## Getting Help

- Open an issue for bugs or features
- Discussion forum for questions
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
