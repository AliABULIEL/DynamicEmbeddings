# Changelog

All notable changes to TIDE-Lite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation and setup guides
- Complete pipeline orchestrator CLI (`tide`)
- Polished README with architecture diagram
- Local and Colab how-to guides with copy-paste commands
- Memory-efficient training options for limited resources

### Changed
- Improved developer UX with clearer command structure
- Enhanced error messages and logging
- Optimized default hyperparameters for better out-of-box performance

### Fixed
- Memory leaks in evaluation scripts
- Batch size handling for edge cases

## [0.3.0] - 2024-01-20

### Added
- Unified CLI orchestrator (`tide`) for complete pipeline management
- Report generation with automatic visualizations
- Ablation study automation
- Baseline comparison framework
- Support for multiple evaluation metrics
- Dry-run mode for testing without execution
- Google Colab notebooks with interactive demos

### Changed
- Restructured CLI to use subcommands (train, eval-stsb, eval-quora, etc.)
- Improved configuration system with YAML support
- Enhanced logging with structured output
- Optimized data loading for faster training

### Fixed
- GPU memory management issues
- Checkpoint loading compatibility
- Mixed precision training stability

## [0.2.0] - 2024-01-15

### Added
- Temporal evaluation module (TimeQA-lite)
- Quora duplicate pairs retrieval evaluation  
- STS-B benchmark integration
- Automatic baseline comparisons
- Results aggregation across evaluations
- Performance plotting utilities

### Changed
- Model architecture to use sigmoid gating
- Time encoding to sinusoidal positional encoding
- Default hyperparameters based on ablation studies
- Training loop for better stability

### Fixed
- Gradient accumulation bugs
- Evaluation metric calculations
- Memory leaks in retrieval evaluation

## [0.1.0] - 2024-01-10

### Added
- Initial TIDE-Lite model implementation
- Basic training script with configurable parameters
- Frozen encoder architecture with temporal MLP
- Temporal and preservation loss functions
- STS-B evaluation script
- Configuration management system
- Basic logging utilities

### Changed
- Initial release

## [0.0.1] - 2024-01-05

### Added
- Project structure and setup
- Basic requirements file
- Initial README
- Git repository initialization

## Tagging Instructions

### Creating a New Release

1. **Update Version Numbers**
   ```bash
   # Update version in setup.py
   sed -i 's/version=".*"/version="0.3.0"/' setup.py
   
   # Update version in __init__.py  
   sed -i 's/__version__ = ".*"/__version__ = "0.3.0"/' src/tide_lite/__init__.py
   ```

2. **Update CHANGELOG**
   - Move items from `[Unreleased]` to new version section
   - Add release date
   - Add comparison link at bottom

3. **Commit Changes**
   ```bash
   git add -A
   git commit -m "chore: release v0.3.0"
   ```

4. **Create and Push Tag**
   ```bash
   # Create annotated tag
   git tag -a v0.3.0 -m "Release version 0.3.0"
   
   # Push tag to remote
   git push origin v0.3.0
   
   # Or push all tags
   git push --tags
   ```

5. **Create GitHub Release**
   ```bash
   # Using GitHub CLI
   gh release create v0.3.0 \
     --title "TIDE-Lite v0.3.0" \
     --notes "See CHANGELOG.md for details" \
     --prerelease  # Remove for stable release
   ```

### Version Numbering Convention

- **MAJOR** version (1.0.0): Incompatible API changes
- **MINOR** version (0.1.0): New functionality, backwards compatible
- **PATCH** version (0.0.1): Bug fixes, backwards compatible

### Pre-release Tags

- Alpha: `v0.3.0-alpha.1`
- Beta: `v0.3.0-beta.1`
- Release Candidate: `v0.3.0-rc.1`

### Hotfix Process

```bash
# Create hotfix branch from tag
git checkout -b hotfix/0.3.1 v0.3.0

# Make fixes
# ...

# Update version to 0.3.1
# Update CHANGELOG

# Merge and tag
git checkout main
git merge --no-ff hotfix/0.3.1
git tag -a v0.3.1 -m "Hotfix version 0.3.1"
git push origin main --tags
```

### Checking Tags

```bash
# List all tags
git tag -l

# Show specific tag details
git show v0.3.0

# Delete local tag
git tag -d v0.3.0

# Delete remote tag
git push origin --delete v0.3.0
```

## Links

- [Releases](https://github.com/yourusername/TIDE-Lite/releases)
- [Commits](https://github.com/yourusername/TIDE-Lite/commits/main)
- [Issues](https://github.com/yourusername/TIDE-Lite/issues)

---

[Unreleased]: https://github.com/yourusername/TIDE-Lite/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yourusername/TIDE-Lite/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yourusername/TIDE-Lite/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yourusername/TIDE-Lite/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/yourusername/TIDE-Lite/releases/tag/v0.0.1
