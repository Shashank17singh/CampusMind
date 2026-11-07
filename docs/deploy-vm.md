# Deploying to one cloud VM

This puts CampusMind on a single server using the production compose file. **These steps have not been run.** Docker was not available where the files were written, so expect to fix small things on your first deployment.

## Pick a server

You need at least 2 CPUs, **4 GB of memory** and 40 GB of disk for the app with a hosted LLM. Running Ollama on the same server needs 16 GB or more, and answers will still be slow without a GPU.

| Option | Notes |
|---|---|
| Oracle Cloud Always Free | ARM instance with up to 4 CPUs and 24 GB. Free, and the best fit if you can get capacity |
| GitHub Student Developer Pack | Includes cloud credits. Check the current offers |
| Azure for Students | Credit without a card, usually enough for a small VM for a semester |

Free offers change, so check the provider's current terms.

## Steps

You need a domain name pointing at the server's public IP.

1. Create an Ubuntu 24.04 VM. Open ports 22, 80 and 443 and nothing else.

2. Install Docker:

   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker "$USER" && newgrp docker
   ```

3. Get the code and configure it:

   ```bash
   git clone <your repository> campusmind && cd campusmind
   cp .env.example .env
   ```

   In `.env`, set:

   | Variable | Value |
   |---|---|
   | `DOMAIN` | Your domain |
   | `SECRET_KEY` | Output of `openssl rand -hex 32` |
   | `POSTGRES_PASSWORD` | Output of `openssl rand -hex 16` |
   | `GRAFANA_PASSWORD` | A password of your choice |
   | `LLM_PROVIDER` and `GROQ_API_KEY` or `ANTHROPIC_API_KEY` | Your LLM. The production stack has no Ollama, so pick a hosted one |
   | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` | Optional, for Google sign-in |
   | `ALLOWED_EMAIL_DOMAIN` | Your college's domain |
   | `SMTP_*` | A mail provider. Without this, no verification email is sent, so nobody can log in |
   | `COOKIE_SECURE` | `true` |

4. Build, and get a certificate:

   ```bash
   docker compose -f infra/compose/docker-compose.prod.yml build
   DOMAIN=campus.example.org EMAIL=you@example.org sh infra/scripts/first-certificate.sh
   ```

5. Start everything:

   ```bash
   docker compose -f infra/compose/docker-compose.prod.yml up -d
   docker compose -f infra/compose/docker-compose.prod.yml ps
   ```

6. Open `https://<your domain>`, register, then make yourself an admin:

   ```bash
   docker compose -f infra/compose/docker-compose.prod.yml exec postgres \
     psql -U campusmind -c "UPDATE users SET role='admin' WHERE email='you@college.edu';"
   ```

## Operating it

- **Grafana:** `ssh -L 3000:localhost:3000 user@server`, then open <http://localhost:3000>.
- **Logs:** `docker compose -f infra/compose/docker-compose.prod.yml logs -f backend worker`
- **Backups** go to the `backups` volume daily. Copy them off the server, and test a restore once:

  ```bash
  pg_restore -h postgres -U campusmind -d campusmind --clean postgres-<stamp>.dump
  ```

- **Updating:** `git pull`, then rebuild with `up -d --build`. Migrations run automatically.
