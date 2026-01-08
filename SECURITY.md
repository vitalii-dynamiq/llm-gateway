# Security Policy

## Supported Versions

We release patches for security vulnerabilities in the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by emailing the maintainers directly or by using GitHub's private vulnerability reporting feature.

When reporting a vulnerability, please include:

1. **Description**: A clear description of the vulnerability
2. **Impact**: The potential security impact
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Affected Versions**: Which versions are affected
5. **Suggested Fix**: If you have a fix in mind, please share it

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution**: Depends on complexity, typically within 30 days

## Security Best Practices

When using arcllm:

### API Key Management

- **Never commit API keys** to version control
- Use environment variables for API keys:
  ```bash
  export OPENAI_API_KEY="sk-..."
  export ANTHROPIC_API_KEY="sk-ant-..."
  ```
- Use secrets management in CI/CD (GitHub Secrets, etc.)
- Rotate API keys regularly

### Input Validation

- Validate and sanitize user inputs before passing to LLM APIs
- Be cautious with user-provided content in system prompts
- Consider content filtering for production applications

### Network Security

- arcllm uses HTTPS for all provider communications
- TLS certificate verification is enabled by default
- Consider network-level controls for production deployments

### Logging

- arcllm does not log API keys or sensitive content by default
- Be careful when enabling debug logging in production
- Do not log full request/response bodies in production

### Dependency Security

arcllm has **zero runtime dependencies** by design, which significantly reduces supply chain risk. For development dependencies, we:

- Use pinned versions in `pyproject.toml`
- Regularly update to patched versions
- Review dependency changes in PRs

## Security Features

### Built-in Protections

1. **TLS/SSL**: All HTTPS connections use system CA certificates
2. **No External Dependencies**: Zero runtime dependencies = smaller attack surface
3. **Input Sanitization**: Request bodies are properly encoded
4. **Connection Pooling**: Secure connection reuse with proper cleanup

### What We Don't Do

- We don't store or cache API keys beyond the request lifecycle
- We don't send telemetry or analytics
- We don't make requests to any domains other than the configured provider endpoints

## Acknowledgments

We appreciate security researchers who responsibly disclose vulnerabilities. Contributors who report valid security issues will be acknowledged (with permission) in release notes.

## Contact

For security-related questions that don't involve vulnerabilities, please open a GitHub issue.
