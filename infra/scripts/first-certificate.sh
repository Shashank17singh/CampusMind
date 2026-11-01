#!/bin/sh
# Gets the first HTTPS certificate. Nginx will not start without one, and Let's Encrypt
# needs Nginx running, so this starts it with a temporary certificate first.
# Usage: DOMAIN=campus.example.org EMAIL=you@example.org sh infra/scripts/first-certificate.sh
set -eu

compose="docker compose -f infra/compose/docker-compose.prod.yml"
live="/etc/letsencrypt/live/$DOMAIN"

$compose run --rm --no-deps --entrypoint sh certbot -c \
    "mkdir -p $live && openssl req -x509 -nodes -newkey rsa:2048 -days 1 \
     -keyout $live/privkey.pem -out $live/fullchain.pem -subj /CN=localhost"

$compose up -d nginx

$compose run --rm --no-deps --entrypoint sh certbot -c "rm -rf $live /etc/letsencrypt/archive/$DOMAIN /etc/letsencrypt/renewal/$DOMAIN.conf"
$compose run --rm --no-deps certbot certonly --webroot -w /var/www/certbot \
    -d "$DOMAIN" --email "$EMAIL" --agree-tos --no-eff-email

$compose exec nginx nginx -s reload
