# GuardRAG

> **Chat with your PDF — without guessing.**

GuardRAG is a retrieval-augmented generation (RAG) service for PDFs with four differentiating ideas:

### Content-Addressed Ingestion

> Uploading the same PDF twice must never re-chunk or re-embed it.

Every upload is hashed with SHA-256. If the hash already exists, GuardRAG returns the existing
document (`"status": "already_indexed"`) with zero extraction, chunking, or embedding work.

### Evidence Guard

> Relevant context does not always mean sufficient evidence.

Before any final LLM call, an **evidence sufficiency check** inspects the retrieved chunks. If they cannot
support an answer, GuardRAG abstains instead of hallucinating.

### Semantic Cache

> Semantically similar questions can reuse previous answers and avoid unnecessary LLM calls.

Questions are embedded and matched against previously answered questions **for the same document,
the same query mode, and the same reader selection** using pgvector cosine similarity. On a hit, the
cached answer is returned with **zero** final LLM calls. An exhaustive answer is never served to a
count question (and vice versa).

### Query-Routed Strategies

> "How many times …", "list every …", "summarize chapter 3", "compare A and B" are not top-k questions.

A deterministic router classifies each question and dispatches to a matching strategy: exact
page-text search and counting (LLM-free), document-wide exhaustive gathering with coverage
reporting, hierarchical whole-document summaries, chapter page-range extraction, balanced
comparison evidence, multi-hop decomposition, metadata, and bibliography parsing. The normal
factual path is untouched — same three LLM calls, same latency.

---

## Screenshots

**1 · Upload** — drop a PDF; SHA-256 dedup, tables/metadata/OCR extraction, local embedding:

![GuardRAG upload screen with drag-and-drop PDF dropzone](docs/screenshots/upload.png)

**2 · Reader + chat** — full-width pdf.js reader with page sidebar; the floating chat opens
with suggested questions and never takes space from the page:

![PDF reader showing a research paper with the floating chat panel and suggested questions](docs/screenshots/chat-suggestions.png)

**3 · Grounded answer** — the chat answers questions about the document that is open in the
reader (evidence-checked; answers never come from outside the PDF):

![Chat answering a question about k-fold cross validation deep inside a 28-page paper](docs/screenshots/answered-question.png)

## Architecture

```text
                 ┌──────────────────────────────┐
                 │   Next.js UI (React / TS)    │
                 └──────────────┬───────────────┘
                                │ REST
                 ┌──────────────▼───────────────┐
                 │        FastAPI backend       │
                 │  POST /api/upload            │
                 │  POST /api/chat              │
                 │  GET  /health                │
                 └──────────────┬───────────────┘
                                │
     PDF ─► PyMuPDF ─► tables/metadata/OCR ─► chunking ─► sentence-transformers (CPU, local)
                                │
        ┌───────────────────────▼────────────────────────┐
        │            PostgreSQL + pgvector               │
        │  documents │ document_pages (exact page text)   │
        │  document_chunks (Vector 384) │ document_summaries
        │  semantic_cache (Vector 384, mode+context keys) │
        └───────────────────────┬────────────────────────┘
                                │
   Chat request flow:
   question ─► deterministic router ─► strategy dispatch
        │            NORMAL: embed ─► cache lookup (mode+context) ─► HIT? ─► cached
        │                       │ miss
        │                       ▼
        │          grounded passage ∪ raw question (half the top-K each) ─► evidence guard
        │                       │ insufficient        │ sufficient
        │                       ▼                     ▼
        │                  abstain + reason     final LLM answer ─► cache
        │
        └─► COUNT / LOCATION / EXACT / PRESENCE: exact page-text search (LLM-free)
            GLOBAL: cached hierarchical summary (map-reduce, bounded prompts)
            CHAPTER: page-range extraction, bounded map-reduce for long sections
            COMPARISON / MULTI_HOP / METADATA / REFERENCE / DEFINITION: specialized paths
```

## Features

- **PDF ingestion** — PyMuPDF text extraction with page numbers preserved
- **Tables as markdown** — detected tables are converted to markdown rows with structure intact
- **Scanned-page OCR** — image-only pages are recovered through tesseract (`eng+ben`), with an
  upload warning when pages still lack usable text; truly image-only PDFs are rejected clearly
- **Running header/footer stripping** — repeated furniture lines are removed conservatively
- **PDF metadata** — title/author/… captured at upload and answerable via `METADATA` questions
- **SHA-256 dedup** — identical PDFs reuse the existing document and embeddings
- **Chunking** — simple word-based sliding window with overlap
- **Local embeddings** — `BAAI/bge-small-en-v1.5` (CPU only, loaded once at startup)
- **Vector search in SQL** — pgvector cosine similarity inside PostgreSQL (HNSW index), scoped
  per document
- **Query routing** — deterministic classification into NORMAL / EXHAUSTIVE / COUNT / EXACT /
  PRESENCE / LOCATION / GLOBAL / CHAPTER / COMPARISON / MULTI_HOP / METADATA / REFERENCE /
  DEFINITION, with an LLM fallback only for low-confidence inputs. Summary questions are
  typo-tolerant ("samary of thi paper" still routes to GLOBAL), document-scoped list nouns
  ("contribution of this paper") trigger a whole-document sweep, and "where is X mentioned"
  answers with locations instead of a yes/no
- **Dual retrieval (NORMAL)** — the grounded passage and the raw question each get half the
  top-K, so a passage that is off-target (or built from an opening text that never mentions
  the topic) cannot crowd out the question's own topic words; the passage prompt also forbids
  "the document does not provide…" phrasing that would embed into arbitrary text
- **Exact counting & location** — word-boundary counts over overlap-free page text (never
  double-counted across overlapping chunks), page-by-page snippets, LLM-free
- **Exhaustive answers with coverage** — term search ∪ broadened retrieval ∪ neighbor chunks,
  capped prompt budget, and an explicit `completeness: complete|partial` field
- **Hierarchical whole-document summaries** — map-reduce over ≤30-page parts, cached per
  document; a 1000-page book is summarized once in bounded prompts, never in one giant call
- **Chapter page ranges** — section headings resolve to page ranges; long sections use the same
  bounded map-reduce
- **Evidence Sufficiency Guard** — structured LLM verdict, refuses to guess
- **Prompt-injection hardening** — document text is framed as untrusted data in every system
  prompt (answer, evidence, retrieval-passage, summaries); the document can never sit in the
  system role
- **Semantic LLM cache** — document-scoped, mode-scoped, selection-scoped, threshold-gated
  answer reuse; legacy cache rows only serve plain NORMAL questions
- **Source attribution** — every answer reports the pages it came from, as clickable chips that
  navigate the reader
- **Strategy & coverage chips** — the chat shows the query mode and whether coverage was
  complete or partial; upload warnings surface as a toast
- **Reader UI** — full-width pdf.js reader (canvas + selectable text layer) with page sidebar,
  zoom, fit-width and a floating chat panel that never takes space from the page
- **Reader-aware answers** — optional `current_page` / `selected_text` context shape the answer
  prompt only (never retrieval, never the evidence verdict); `selected_text` also participates in
  the cache key so answers computed under one selection are never reused for another

## Tech stack

| Layer     | Tech |
|-----------|------|
| Backend   | Python, FastAPI, Pydantic, SQLAlchemy 2 |
| Database  | PostgreSQL + pgvector |
| Embeddings| sentence-transformers (CPU) |
| LLM       | Any OpenAI-compatible chat completions API |
| Frontend  | Next.js, React, TypeScript, Tailwind CSS |

## Why pgvector

- One database for documents, vectors, and cache — no extra vector service to operate
- Similarity search runs **inside SQL** (`ORDER BY embedding <=> $1`), never loading vectors into Python
- Document isolation is a simple `WHERE document_id = $1`
- Same threshold arithmetic (`1 - distance`) used for retrieval and cache matching

## Setup

### 1. PostgreSQL + pgvector

Use your existing PostgreSQL installation (pgvector ≥ 0.5 required):

```bash
# Ubuntu/Debian
sudo apt install postgresql-18-pgvector
```

Create a dedicated database (do not touch other databases):

```bash
psql -U postgres -c 'CREATE DATABASE guardrag;'
psql -U postgres -d guardrag -c 'CREATE EXTENSION IF NOT EXISTS vector;'
```

The backend also runs `CREATE EXTENSION IF NOT EXISTS vector` and creates its tables on startup,
so step 2 is optional if your role is allowed to create extensions.

### 2. Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit DATABASE_URL + LLM_API_KEY
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev                 # http://localhost:3000
```

## Environment variables

`backend/.env.example`:

```env
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/guardrag

LLM_API_KEY=
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=https://api.openai.com/v1

EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

TOP_K=4

CACHE_SIMILARITY_THRESHOLD=0.90
```

`LLM_BASE_URL` accepts any OpenAI-compatible endpoint (OpenAI, OpenRouter, vLLM, Ollama, …).

## API

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /api/settings` · `POST /api/settings`

Runtime LLM configuration (persisted to `backend/.env`, applied immediately):

```json
{ "llm_model": "gpt-4o-mini", "llm_base_url": "https://api.openai.com/v1", "llm_api_key_masked": "sk-proj…WZ0A" }
```

POST accepts any subset of `llm_api_key`, `llm_model`, `llm_base_url` — switch between OpenAI,
Grok (xAI) or any OpenAI-compatible endpoint without restarting. The key is never returned in full.

### `POST /api/upload`

`multipart/form-data` with a `file` field (PDF, ≤ 20 MB).

First upload of a file:

```json
{ "document_id": "8f0a…", "filename": "paper.pdf", "pages": 10, "status": "ready", "reused": false, "warning": null }
```

Same PDF (identical SHA-256) uploaded again — existing document and embeddings are reused,
nothing is re-chunked or re-embedded:

```json
{ "document_id": "8f0a…", "filename": "paper.pdf", "pages": 10, "status": "already_indexed", "reused": true }
```

When extraction found problems (pages without a usable text layer, OCR recovery), `warning`
carries a human-readable note:

```json
{ "document_id": "8f0a…", "fWhat evidence does this question require?”ilename": "scan.pdf", "pages": 40, "status": "ready", "reused": false,
  "warning": "38 of 40 pages have no usable text layer (possibly scanned images); answers may be incomplete. OCR recovered text from 38 page(s)." }
```

### `POST /api/chat`

```json
{
  "document_id": "8f0a…",
  "question": "What is the main contribution of this paper?",
  "current_page": 3,
  "selected_text": "optional highlighted passage (≤ 4000 chars)"
}
```

`current_page` and `selected_text` are optional reader hints. They only shape how the final
answer prompt is interpreted — never which chunks are retrieved, never the evidence verdict.
`selected_text` additionally participates in the cache key (answers computed under one
selection are never served for a different selection); `current_page` never affects the cache.

Answered:

```json
{
  "status": "answered",
  "answer": "…",
  "cache_hit": false,
  "evidence_sufficient": true,
  "sources": [{ "page": 3, "chunk_id": 12 }],
  "query_mode": "NORMAL",
  "completeness": null
}
```

`query_mode` reports which strategy produced the answer. `completeness` is set by
document-wide strategies: `"complete"` when the whole text was covered, `"partial"` when the
search had to stop early (or a side of a comparison had no evidence).

```json
{ "status": "answered", "answer": "Occurrences of 'graph': 5\n…", "cache_hit": false,
  "evidence_sufficient": true, "sources": [{ "page": 1, "chunk_id": 0 }], "query_mode": "COUNT",
  "completeness": "complete" }
```

Cache hit:

```json
{ "status": "answered", "answer": "…", "cache_hit": true, "evidence_sufficient": true, "sources": [], "query_mode": "NORMAL" }
```

Insufficient evidence:

```json
{
  "status": "insufficient_evidence",
  "answer": null,
  "cache_hit": false,
  "evidence_sufficient": false,
  "reason": "The uploaded PDF does not provide enough information to answer this question.",
  "query_mode": "PRESENCE"
}
```

## Example usage

1. Start PostgreSQL, the backend, and the frontend.
2. Drop a PDF on the upload screen → indexing finishes → the reader opens full-width
   (a warning toast appears if pages needed OCR or lacked text).
3. Select a passage (or not), open the floating chat, ask *"What problem does this paper
   address?"* → answered with clickable `p. 1` / `p. 2` source chips that scroll the reader.
4. Ask the same question again → `⚡ Reused previous answer`
   (semantic cache hit, **no final LLM call**).
5. Ask *"What was the author's phone number?"* → `⚠ Not enough evidence`
   (**no hallucinated answer, no final LLM call**).
6. Ask *"How many times does the word graph appear?"* → exact page-text count with per-page
   breakdown, `count` chip, no LLM call.
7. Ask *"List all contributions of this paper"* → `exhaustive` chip with `full document` or
   `partial coverage` depending on whether the whole text was searched.
8. Ask *"Give a complete overview of this document"* → hierarchical summary built once from
   bounded page windows and cached for later global questions.
9. Ask *"samary of thi paper"* (sic) → typo-tolerant routing still produces the full-document
   summary; ask *"contribution of this paper"* → `exhaustive` sweep with per-item citations.
10. Re-upload the same file → `Already indexed — reusing existing embeddings`, `document_id`
    unchanged, chunk count unchanged.

## Project structure

```text
guardrag/
├── backend/
│   ├── app/
│   │   ├── main.py        # FastAPI endpoints
│   │   ├── config.py      # env settings (incl. OCR options)
│   │   ├── database.py    # engine + init (extension, tables, additive migrations, HNSW)
│   │   ├── models.py      # Document, DocumentPage, DocumentChunk, DocumentSummary, SemanticCache
│   │   ├── schemas.py     # Pydantic contracts
│   │   ├── pdf.py         # validation + extraction (tables, OCR, metadata, warnings)
│   │   ├── chunking.py    # overlapping word chunks
│   │   ├── embeddings.py  # single-load CPU embedding model
│   │   ├── retriever.py   # pgvector cosine top-K
│   │   ├── routing.py     # deterministic query router + format instructions
│   │   ├── search.py      # exact page-text search, counting, structure helpers
│   │   ├── summaries.py   # hierarchical map-reduce document summaries
│   │   ├── cache.py       # semantic cache (mode + selection isolated)
│   │   ├── evidence.py    # evidence sufficiency guard
│   │   ├── llm.py         # OpenAI-compatible client + JSON mode
│   │   └── rag.py         # decision flow + strategy dispatch
│   ├── tests/             # 115 tests (unit + guardrag_test integration)
│   ├── requirements.txt
│   ├── pyproject.toml
│   └── .env.example
├── frontend/              # Next.js UI
│   ├── components/
│   │   ├── UploadScreen.tsx   # dropzone + indexing steps
│   │   ├── Reader.tsx         # pdf.js canvas + text layer, toolbar, page sidebar
│   │   └── FloatingChat.tsx   # FAB + fixed overlay panel, mode/coverage chips
│   ├── lib/                   # typed API client
│   └── public/pdf.worker.min.mjs   # copied from pdfjs-dist on postinstall
├── docs/screenshots/      # README images
└── README.md
```

## Tests

```bash
cd backend
pytest -q
```

115 tests, all LLM traffic mocked:

- **Unit** — PDF validation, chunking, cache hit/miss/threshold, evidence verdicts, query
  routing for every mode (including typo-tolerant summary routing and doc-scoped exhaustive),
  dual-retrieval merge (passage + question sides), exact search/counting/structure helpers,
  extraction (tables → markdown, OCR recovery, metadata, warnings, furniture stripping)
- **Integration** (real PostgreSQL, isolated `guardrag_test` database; auto-skipped when
  PostgreSQL is unavailable) — strategy dispatch end-to-end: exhaustive completeness (every
  item reaches the answer prompt), LLM-free counting, absence without false positives, chapter
  page ranges, balanced comparison prompts, cache isolation by mode and by selection,
  prompt-injection containment (document text never enters a system prompt), reference
  parsing, legacy documents without page rows, and **1000-page boundedness** (summary and
  exhaustive prompts never exceed their budgets)
- **API** — upload stores pages/metadata/chunks with warning + dedup, chat accepts the
  legacy two-field body and returns `query_mode` / `completeness`

## Future improvements

- Hybrid (BM25 + vector) retrieval
- Persistent conversation history
- Streaming answers
- Citation-map resolution for numbered in-text citations
- Per-user documents and access control
# GuardRag
