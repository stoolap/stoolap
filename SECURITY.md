# Security Policy

## Supported versions

Security fixes are released for the latest minor version on the `main` branch. Older releases do not receive backported fixes.

| Version | Supported |
|---------|-----------|
| 0.4.x   | Yes       |
| < 0.4   | No        |

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through [GitHub Security Advisories](https://github.com/stoolap/stoolap/security/advisories/new) for this repository. Include a description of the issue, the affected version or commit, and steps or a minimal SQL script that reproduces it.

You will receive an acknowledgement within 7 days. Once the report is confirmed, a fix is prepared on a private branch, released in a new version, and the advisory is published with credit to the reporter unless they prefer otherwise.

## Scope

Stoolap is an embedded database library. It runs inside the calling process with that process's privileges and has no network listener, authentication or access control of its own. The following are in scope:

- Memory safety defects reachable through the public Rust, C or driver APIs, including through valid or malformed SQL.
- Data corruption or silent data loss, including on crash, on WAL replay or on volume reload.
- Panics reachable from SQL or the API that abort the host process (the release profile uses `panic = "abort"`).
- Vulnerabilities in dependencies, which are checked continuously with `cargo audit` in CI.

The following are out of scope: denial of service through queries that legitimately consume large amounts of memory or time on the caller's own data, and issues in applications that embed Stoolap.

## Dependency auditing

The `Cargo.lock` is checked against the [RustSec advisory database](https://rustsec.org/) on every push and pull request, and once a day on a schedule, by the `Security Audit` workflow. A new advisory affecting a locked dependency fails the workflow until the dependency is updated or the advisory is triaged and ignored in `.cargo/audit.toml` with a reason.
