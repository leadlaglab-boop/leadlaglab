# Security Policy

_Lead/Lag Lab — last updated 2026-10-03_

---

## Secrets management

- **No secrets in git.** `.env` is gitignored. gitleaks runs as a pre-commit hook and in CI.
- **Local:** copy `.env.example` → `.env`; fill in values. Never commit `.env`.
- **CI/CD:** all secrets live in GitHub Actions repository secrets. The workflow reads them as environment variables; they are never printed to logs.
- **Rotation:** if a key is ever committed accidentally, rotate it immediately and treat the leaked value as compromised.

## GitHub Actions security

- Least-privilege permissions: `contents: read` by default; `contents: write` only for the data-archive commit job.
- All third-party Actions pinned by commit SHA (not mutable tag).
- `GITHUB_TOKEN` permissions scoped per job.
- Dependabot enabled for pip and npm; weekly schedule.
- `pip-audit` and `npm audit` run in CI on every push; failures block merge.

## Branch protection (main)

- Require PR review before merge (enable in GitHub repo settings).
- Require status checks to pass (lint, type-check, test, leakage-test).
- No force-push to `main`.
- Enable GitHub secret scanning (Settings → Security → Secret scanning).

## Cloudflare Pages security headers

Defined in `site/public/_headers`. Applied to all responses:

```
/*
  Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: strict-origin-when-cross-origin
  Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https://static.cloudflareinsights.com; connect-src 'self' https://cloudflareinsights.com; font-src 'self'; frame-ancestors 'none'
```

Note: `'unsafe-inline'` for styles is acceptable for the initial build. If a strict CSP is achievable without it (e.g., via CSS hashing), remove it in M7.

## Analytics and privacy

- **Cloudflare Web Analytics** — no cookies, no personal data, no consent banner required.
- **No third-party trackers, pixels, or tag managers.**
- No server-side user data stored (static site; no backend).
- localStorage used only for watchlist (user-controlled; no PII).

## Dependency scanning

```
# Run locally before any PR
pip-audit
npm audit
```

Both run automatically in CI (`.github/workflows/security.yml`).

## Responsible disclosure

If you find a security issue, email `leadlaglab@gmail.com`. We will respond within 72 hours.
