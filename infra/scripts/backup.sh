#!/bin/sh
# Runs once a day inside the backup container: a PostgreSQL dump and a Qdrant snapshot.
# Copy /backups off the server too. A backup on the same disk does not survive losing it.
set -u

while true; do
    stamp=$(date +%Y%m%d-%H%M%S)

    if pg_dump -h postgres -U campusmind -Fc campusmind > "/backups/postgres-$stamp.dump"; then
        echo "Wrote postgres-$stamp.dump"
    else
        echo "PostgreSQL backup FAILED" >&2
        rm -f "/backups/postgres-$stamp.dump"
    fi

    # Qdrant writes the snapshot into its own volume, next to the collection.
    if wget -q -O - --post-data='' \
        "http://qdrant:6333/collections/campusmind_chunks/snapshots" > "/backups/qdrant-$stamp.json"; then
        echo "Requested Qdrant snapshot"
    else
        echo "Qdrant snapshot FAILED" >&2
    fi

    find /backups -type f -mtime "+${KEEP_DAYS:-14}" -delete
    sleep 86400
done
