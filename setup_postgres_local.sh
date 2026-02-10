#!/usr/bin/env bash
set -euo pipefail

export PG14BIN=/usr/lib/postgresql/14/bin

# create and grant ownership to postgres
sudo mkdir -p /data/pgdata
sudo chown -R postgres:postgres /data/pgdata
sudo chmod 700 /data/pgdata

# init as postgres
sudo -u postgres "$PG14BIN/initdb" -D /data/pgdata -E UTF8 --locale=C

# start as postgres
sudo -u postgres "$PG14BIN/pg_ctl" -D /data/pgdata -l /data/pgdata/pg.log start

psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -c "CREATE ROLE letta LOGIN PASSWORD 'letta';"
psql -h 127.0.0.1 -p 5432 -U postgres -d postgres -c "CREATE DATABASE letta OWNER letta;"
psql -h 127.0.0.1 -p 5432 -U postgres -d letta -c "CREATE EXTENSION IF NOT EXISTS vector;"
