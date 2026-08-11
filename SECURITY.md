# Security Policy

## Supported versions

Voxint is pre-alpha. Only the latest release (and `main`) receives security fixes.

## Reporting a vulnerability

Please report vulnerabilities privately via GitHub Security Advisories on this repository
("Report a vulnerability"). Do not open a public issue for security reports. You should receive
an acknowledgement within 7 days.

## Deployment posture

Voxint is designed for **single-machine, self-hosted** deployments:

- The API and review UI bind to `127.0.0.1` by default and require a single-user credential.
  Never expose them directly to the internet — put a reverse proxy with TLS and your own auth in
  front if you need remote access.
- Media routes stream files from `MEDIA_ROOT` only, behind the same auth, with path traversal
  guarded server-side.
- All credentials (database, Redis, LLM API keys, Hugging Face token) come from environment
  variables — nothing is baked into images or committed to the repository.
