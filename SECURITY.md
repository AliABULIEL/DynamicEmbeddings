# Security Policy

## Supported Versions

We release security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security
vulnerability, please follow these steps:

1. **Do NOT** open a public issue
2. Email aliab@example.com with details:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

3. Allow up to 48 hours for initial response
4. Work with maintainers on coordinated disclosure

## Security Considerations

### Data Privacy
- This project processes academic abstracts (public domain)
- No personal data or PII is collected or stored
- Model embeddings may encode semantic information from training data

### Model Security
- LoRA adapters are small (~1-5MB) and can be inspected
- Base model is from Hugging Face's official sentence-transformers
- No remote model loading or arbitrary code execution

### Dependencies
- All dependencies are pinned with exact versions
- Regular security audits via `pip-audit` or Dependabot
- FAISS is used for similarity search (CPU version is default)

### Reproducibility
- Fixed random seeds for deterministic runs
- Environment dumps include all package versions and commit SHAs
- No telemetry or external logging

## Best Practices

When using this project:
- Run in isolated virtual environments
- Review data sources before processing
- Use GPU hardware with appropriate security controls
- Keep dependencies updated via `pip install --upgrade -r requirements.txt`

## Acknowledgments

We appreciate responsible disclosure and will credit security researchers
(with permission) in release notes.
