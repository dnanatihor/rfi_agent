# Deploy / local test

Run the RFI agent on one machine: Postgres (Docker), FastAPI on port **8000**, Next.js on port **3000**. Then walk the checklist in §8.

On startup the API creates missing tables from the models. It does not alter tables that already exist.

## Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+ (2.x is fine)
- Node.js 20+ (24 is fine)
- Docker with Compose
- An LLM key (OpenAI-compatible, Anthropic, or AWS Bedrock)
- Optional: Google Cloud OAuth client if you will index Drive files

## 1. Environment

From the repo root:

```bash
cp .env.example .env
```

Edit `.env`. Do not commit it.

Minimum to test Q&A: `LLM_API_KEY` (or `OPENAI_API_KEY` in the shell) plus embeddings. Drive can stay empty.

### Models

Pick one chat provider. Embeddings cannot use Anthropic (no embedding API).

**OpenAI or any OpenAI-compatible gateway** (default):

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...
# LLM_BASE_URL=https://openrouter.ai/api/v1   # optional

EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

`OPENAI_API_KEY` in the shell is used if `LLM_API_KEY` is empty.

**Anthropic:**

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
LLM_API_KEY=sk-ant-...
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=sk-...          # still need an embedding key
```

**Bedrock:**

```bash
LLM_PROVIDER=bedrock
LLM_MODEL=anthropic.claude-sonnet-4-20250514-v1:0
EMBEDDING_PROVIDER=bedrock
EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
EMBEDDING_DIM=1024
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Pin exact model ids. Do not use `*-latest`.

Leave `RERANK_PROVIDER` empty for the first test (hybrid RRF only). Optional later: `jina`, `cohere`, or `openai_compatible`.

### Google Drive (optional)

Skip this until the URL/upload path works.

1. Create a GCP project and enable [Google Drive API](https://console.cloud.google.com/apis/library/drive.googleapis.com).
2. [OAuth consent](https://console.cloud.google.com/auth/audience): External, **Testing**, add your Google account as a test user.
3. [Data Access](https://console.cloud.google.com/auth/scopes): `https://www.googleapis.com/auth/drive.readonly`.
4. [Create client](https://console.cloud.google.com/auth/clients): **Web application**.
   - Origins: `http://localhost:3000`, `http://localhost:8000`
   - Redirect: `http://localhost:8000/oauth/google/callback`
5. Copy the id and secret into `.env` immediately (Google shows the secret once):

```bash
GOOGLE_CLIENT_ID=....apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=....
GOOGLE_REDIRECT_URI=http://localhost:8000/oauth/google/callback
FRONTEND_ORIGIN=http://localhost:3000
```

## 2. Postgres

```bash
docker compose up -d
docker compose ps
```

Wait until `postgres` is healthy. Connection string in `.env`:

```
DATABASE_URL=postgresql+psycopg://rfi:rfi@127.0.0.1:5433/rfi
```

Host port is **5433** so it does not clash with a local Postgres on 5432.

Stop: `docker compose down`. Data volume `rfi_pgdata` is kept unless you add `-v`.

## 3. Backend

Poetry creates `.venv` in the repo (`poetry.toml`).

```bash
poetry install
poetry run uvicorn rfi_agent.main:app --reload --reload-dir src --host 127.0.0.1 --port 8000
```

Check:

```bash
curl -s http://127.0.0.1:8000/health
```

You want `"ok": true` and `"llm_configured": true`. `"embeddings_configured": true` is required to index. `"google_configured": true` only after Drive keys are set.

The `vector` extension and any missing tables are created on startup. Existing tables are not altered.

## 4. Frontend

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The UI proxies `/backend/*` to port 8000.

The status line under the heading should show `LLM ready`, not `LLM key missing`. If it stays on **Starting…**, the API is not up.

## 5. One-shot helper

`scripts/dev.sh` copies `.env` if missing and starts Postgres. It does not start uvicorn or Next.js.

```bash
./scripts/dev.sh
```

Then run the backend and frontend commands from steps 3 and 4.

## 6. Automated tests

These do **not** call a live LLM or Postgres:

```bash
poetry install
poetry run pytest -q
```

Expect 28 passing. Coverage includes chunking, SSRF, RRF, sitemap expansion, citation quote matching, questionnaire CSV parse, export draft labeling, and project-scoped retrieval.

Eval prompts for a later live golden set: `tests/fixtures/eval_questions.json` (20 questions). They are not executed by `pytest`.

## 7. Sample questionnaire (for the UI test)

Save as `sample-rfi.csv`:

```csv
Question,Category,Guidance
Do you encrypt data at rest?,Security,Yes/No
How is access to production systems authenticated?,Security,Short paragraph
What is the backup recovery objective?,Operations,Yes/No or time window
```

Three rows is under the 20-question confirmation threshold, so drafting starts after upload.

## 8. Manual test (do this in the browser)

Use a **small** site first. A large docs site can be thousands of pages and will take time and embedding cost.

### A. Organization and project

1. Open [http://localhost:3000](http://localhost:3000).
2. Save the organization name, then continue.
3. Create a project (name and customer) or open one from **Your projects**.
4. The workspace order is project index, sources, questionnaire, then chats.

### B. Index a URL

1. Paste a small HTTPS docs URL (or a single page / sitemap) → **Index**.
2. Source status moves `pending` → `fetching` → `indexing` → `ready` (or `partial` if some pages fail).
3. Chunk count becomes > 0.
4. Ask: a question the page actually answers. You should get citations with URLs/quotes.
5. Edit the answer text if you want, then **Approve into memory**.
6. **Reject** on a throwaway answer and confirm status becomes `rejected`.

### C. Memory reuse

1. Click **New chat**. The project index stays. Approved memory stays with the organization.
2. Ask the same (or very similar) question.
3. Origin should be `memory` if similarity ≥ `MEMORY_ACCEPT` (default 0.88).

### D. Questionnaire + export

1. Open a chat. Index documents on the project if you have not already.
2. Upload `sample-rfi.csv`.
3. Status goes `parsed` → `running` → `complete` (or `partial` if the model/budget fails).
4. The table fills with per-question status and draft text.
5. Download JSON and XLSX with **Include unreviewed drafts** unchecked: only approved/rejected rows (often empty if you have not reviewed batch answers).
6. Check **Include unreviewed drafts** and download again: drafts appear and are labeled `draft` in XLSX.

### E. Detach / reindex

1. On a project with a ready source, click **Detach**. Status becomes `detached`; later asks must not cite that source.
2. **Reindex** on a live source: status returns to `pending` then `ready` again.

### F. Drive (only if `.env` has Google keys)

1. **Connect Google Drive**, finish OAuth, land back on `http://localhost:3000/?drive=connected`.
2. Paste a Drive file or folder URL → **Index**.
3. Health should show Drive connected; source becomes `ready`.

### G. Health snapshot

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

Useful fields: `llm_configured`, `embeddings_configured`, `rerank_configured`, `google_configured`, `drive_connected`.

## 9. Use it (short)

1. Save the organization, then create or open a project.
2. Paste a site, sitemap, or document HTTPS URL and click **Index**. Every website URL is expanded the same way: robots sitemap, then `/sitemap.xml`, then `/sitemap-pages.xml`.
3. Or connect Drive and paste a file or folder URL, or upload a file.
4. Ask in a chat, then edit, approve, or reject. A trace is shown on the answer it produced.
5. Upload a questionnaire (CSV / XLSX / DOCX). More than 20 questions asks for confirmation. Runs are resumable.
6. Export JSON or XLSX. Drafts are omitted unless you opt in.
7. **New chat** starts another thread on the same project index. It does not delete documents. Approved memory stays with the organization.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| UI stuck on “Starting…” | Backend not running; `curl http://127.0.0.1:8000/health` |
| Ask has nothing to cite | Index a source on this project first, or approve an answer into memory |
| LLM key missing | `.env` `LLM_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) and **restart uvicorn** |
| Indexing never leaves `pending` | Embeddings key missing; watch the uvicorn terminal |
| Large site “forever” | Expected for hundreds of pages; watch chunk count, or use a single-page URL |
| `parse_failed:image_only_pdf` | Scanned PDF; OCR is out of scope |
| Drive client not set | `GOOGLE_CLIENT_*` in `.env`, restart backend |
| `redirect_uri_mismatch` | Client redirect must be exactly `http://localhost:8000/oauth/google/callback` |
| Drive login blocked | Consent screen in Testing; your account listed as a test user |
| Postgres connection refused | `docker compose ps`; port 5433 |
| Schema / missing column errors | Tables are created on startup and are not altered later. Recreate the database after a model change |
| Port 8000/3000 in use | Stop the old process or change the port (update `FRONTEND_ORIGIN` and Next rewrites if you move 8000) |
| Poetry not found | Install from https://python-poetry.org/docs/#installation then retry `poetry install` |
| pytest collection errors | `poetry install` from repo root; then `poetry run pytest -q` |

## Layout

```
README.md              What it is and how to start
LICENSE                Apache-2.0
docker-compose.yml     Postgres + pgvector
.env.example           Template for secrets (copy to .env)
pyproject.toml         Poetry project + dependencies
poetry.lock            Locked Python versions
poetry.toml            In-project `.venv`
src/rfi_agent/         FastAPI, LangGraph, ingest
web/                   Next.js UI
tests/                 Offline harness tests
docs/DEPLOY.md         Local setup and troubleshooting
```
