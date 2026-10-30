# Security Policy

Hyundai Motor Group takes the security of our software products and services seriously, including all source code repositories within our GitHub organizations.

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub Issues, Discussions, or Pull Requests.

To report a security vulnerability, please email:

- psirt@hyundai.com

## Known limitations for deployers

The web app's login is a name picker **without a password** and is meant for trusted internal networks only; it is
not access control. Uploads are not size- or type-restricted. Do not expose the server to untrusted networks without a reverse proxy
that adds authentication and limits.
