# RFI Agent

Local draft assistant for RFIs, RFPs, and security questionnaires. It answers from documents you index and from answers you have already approved. Every answer cites its sources, or it says the evidence is not there.

This is a single-machine app: Postgres in Docker, a FastAPI backend, and a Next.js UI. It does not send drafts to a customer or a portal.

## How it is organized

1. **Organization.** Each time you open the app, save the organization first. Approved answers belong to the organization and are shared by its projects.
2. **Projects.** A project is one customer RFI. It has one document index.
3. **Chats.** A chat is a question thread inside a project. Every chat in that project asks against the same index. Starting a new chat does not delete indexed documents.

Index a website, a Google Drive file or folder, or an upload. Then ask a question, review the draft, approve it into memory, or reject it. You can also upload a questionnaire (CSV, XLSX, or DOCX) and export reviewed answers as JSON or XLSX.

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) 1.8+
- Node.js 20+
- Docker with Compose
- An API key for an OpenAI-compatible chat model, Anthropic, or Amazon Bedrock
- Embeddings from an OpenAI-compatible API or Bedrock (Anthropic has no embeddings API)

## Run it locally

```bash
cp .env.example .env
# Put your LLM and embedding keys in .env. Do not commit .env.

docker compose up -d
poetry install
poetry run uvicorn rfi_agent.main:app --reload --reload-dir src --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The UI sends `/backend/*` to the API on port 8000.

`./scripts/dev.sh` creates `.env` if it is missing and starts Postgres. It does not start the API or the UI.

Postgres listens on **127.0.0.1:5433** with the local development user and password `rfi` / `rfi`. Change that password before you expose the database port.

On startup the API creates the `vector` extension and the tables from the SQLAlchemy models. It does not alter an existing database. Adding a column means editing the models and recreating the database.

Full setup, Drive OAuth, and troubleshooting: [docs/DEPLOY.md](docs/DEPLOY.md).

## Providers

Set these in `.env`. Pin exact model ids.

| Role | Providers |
| --- | --- |
| Chat | `openai_compatible`, `anthropic`, `bedrock` |
| Embeddings | `openai_compatible`, `bedrock` |
| Optional rerank | `jina`, `cohere`, `openai_compatible` |

Leave `RERANK_PROVIDER` empty to use hybrid retrieval only.

Any public HTTPS website URL is expanded the same way: `robots.txt` sitemap, then `/sitemap.xml`, then `/sitemap-pages.xml`. Child sitemaps are followed. Drive links stay a separate source type.

## Tests

These do not call a live model or Postgres:

```bash
poetry run pytest -q
```

## License

[Apache-2.0](LICENSE)
