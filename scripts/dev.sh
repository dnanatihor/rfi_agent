#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — add LLM_API_KEY and GOOGLE_CLIENT_* before Q&A/Drive."
fi
docker compose up -d
echo "Postgres is on localhost:5433"
echo "Backend:  poetry install && poetry run uvicorn rfi_agent.main:app --reload --reload-dir src --host 127.0.0.1 --port 8000"
echo "Frontend: cd web && npm install && npm run dev"
