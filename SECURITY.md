# 🔒 Security Policy

<p align="left">
  <b>English</b> | <a href="SECURITY.ja.md">日本語 (Japanese)</a>
</p>

## Supported Versions

The ONIWA project is actively developed. Security patches and fixes are provided for the following versions:

| Version | Supported |
| :--- | :--- |
| `main` branch (latest commit) | :white_check_mark: Supported |
| < 0.1.0 (older commits) | :x: Not supported |

---

## Reporting a Vulnerability

If you discover a potential security vulnerability (e.g., memory unsafety, information disclosure, or denial-of-service risks), **please do not open a public issue**. Instead, report it privately using one of the following channels:

1. **GitHub Private Vulnerability Reporting (Recommended)**:
   - Navigate to the repository's [Security tab](https://github.com/KazutoMakino/oniwa/security) and click **"Report a vulnerability"** to create a confidential draft advisory.
2. **Direct Contact**:
   - Contact the project maintainer via the email address or contact info listed on their GitHub profile.

### Information to Include
- Affected crate or component (e.g., `oniwa-lm`, `oniwa-pipeline`, CLI binaries, etc.)
- Clear description of the vulnerability with reproduction steps (PoC code or command sequences)
- Potential impact assessment (DoS, panic, unexpected resource consumption, etc.)

We will acknowledge receipt promptly, review the findings, and coordinate a fix and public advisory.
