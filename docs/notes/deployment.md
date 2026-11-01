# Deployment and operations

Code: `infra/`, `.github/workflows/ci.yml`, `backend/app/metrics.py`, `backend/scripts/seed.py`.

**Nothing under `infra/` has been run, and neither has the CI workflow.** Docker was not available when these were written. Test them first. Application metrics and the seed script were run locally.

## Key decisions

**1. Multi-stage Docker builds with non-root users.** The final image contains no build tools, which keeps it smaller and removes things an attacker could use. Running as a normal user limits what a compromised process can do. The frontend uses an unprivileged Nginx image for the same reason.

**2. Migrations run as a one-shot service.** The API and workers wait for `migrate` to finish, so no process meets a schema it does not expect, and two API replicas can never race to migrate.

**3. Qdrant in production, even though Chroma is the development default.** The API and Celery worker are separate processes, and Chroma's local mode expects one owner. This is the reason for the `VectorStore` abstraction: switching needs one environment variable, and no code.

## More design points

- **Why is `/metrics` blocked at Nginx?** It reveals traffic details. Prometheus reaches it directly inside the Docker network.
- **Metric labels use route templates.** A label per document id would create unlimited time series and overload Prometheus.
- **Why `proxy_buffering off`?** Nginx would otherwise hold back the streamed answer until it finished.
- **Why `--proxy-headers`?** Without it every request appears to come from Nginx, and per-IP login limits would apply to all users together.
- **One certificate problem.** Nginx will not start without a certificate, and Let's Encrypt needs a running web server. `first-certificate.sh` breaks the cycle with a temporary certificate.
- **Seed password from the environment.** A demo password in the repository is a working credential the moment someone deploys it. Seeding also refuses to run when `ENVIRONMENT=production`.

## Known limits

- One server means one point of failure.
- Backups stay on the same disk until someone copies them away.
- The Qdrant snapshot stays in Qdrant's volume, and only the response is saved in `/backups`.
- Prometheus has no alert rules yet.
- Uploaded files are not included in the backup.
- Chat has no per-user rate limit, only Nginx's limit per IP address, so one account could run up a hosted LLM bill.
- When Redis is down, each failed cache call waits about 4 seconds before carrying on, so the app stays up but gets slower.
