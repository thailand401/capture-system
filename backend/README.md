# Capture System Backend

Backend for an Android application that collects traffic-sign capture events. Images are
stored **temporarily** in Supabase Storage; metadata and AI predictions are stored
**permanently** in Postgres. A background worker polls for newly uploaded events, runs an
(currently stubbed) AI pipeline, saves the prediction, and deletes the temporary image.

## Tech stack

Python 3.12 · FastAPI · SQLAlchemy 2.x (async) · Alembic · Pydantic v2 · Supabase
(Postgres + Storage) · uv · OpenCV · Ultralytics YOLO (interface only) · PaddleOCR
(interface only)

## Project layout

```
backend/
    app/
        api/          # FastAPI routers + request-scoped dependencies
        core/          # config, logging, exceptions
        db/            # SQLAlchemy engine/session, declarative base
        models/        # ORM models (capture_event, prediction, processing_log)
        schemas/        # Pydantic request/response models
        services/       # business logic (kept out of routes)
        repositories/   # data-access layer, one class per aggregate
        storage/        # Supabase Storage backend (+ swappable interface)
        ai/             # AI pipeline interfaces, stub implementations, orchestrator
        workers/        # background polling worker + per-event processor
        utils/          # cross-cutting helpers (retry policy)
    migrations/         # Alembic migrations
    tests/              # pytest test suite
```

## Setup

1. Install [uv](https://docs.astral.sh/uv/).
2. Install dependencies:
   ```
   uv sync
   ```
3. Copy `.env.example` to `.env` and fill in real values (a `.env` is already present in
   this workspace for local development — never commit it).
4. Apply database migrations:
   ```
   uv run alembic upgrade head
   ```
5. Run the API (the background worker starts automatically in the same process):
   ```
   uv run uvicorn app.main:app --reload
   ```

## Supabase setup

- Create a **private** Storage bucket named `traffic-sign-temp`.
- `SUPABASE_KEY` must be the **service_role** key (server-side only — bypasses RLS to
  manage the bucket and tables). Never embed it in the Android app or any client.
- `DATABASE_URL` uses the async driver: `postgresql+asyncpg://...` (note the `+asyncpg`).

## API

| Method | Path            | Description                                   |
|--------|-----------------|------------------------------------------------|
| POST   | `/events`       | Multipart upload: `metadata` (JSON), `image`, `thumbnail`. Returns `{id, status}`. |
| GET    | `/events/{id}`  | Status + predictions for one event (`id` is the public UUID). |
| GET    | `/events`       | Paginated list (`limit`, `offset`, optional `status` filter). |
| GET    | `/health`       | Liveness probe.                                |

## Background worker

Runs as an asyncio task inside the FastAPI process (see `app/workers/runner.py`),
polling every `WORKER_INTERVAL_SECONDS`. Each cycle: claims `NEW` events
(`SELECT ... FOR UPDATE SKIP LOCKED`, so it stays safe even if scaled to multiple
instances), downloads the image to `LOCAL_CACHE`, runs the AI pipeline, saves a
`prediction` row, deletes the image + thumbnail from Storage, and marks the event
`DONE`. Any failure marks the event `ERROR` and is recorded in `processing_log`; the
worker loop itself never crashes. Storage operations retry with exponential backoff
(`WORKER_MAX_RETRIES` / `WORKER_RETRY_BACKOFF_SECONDS`).

## AI pipeline

`app/ai/interfaces.py` defines replaceable stages: `Detector`, `ShapeValidator`,
`ColorValidator`, `OCR`, `Classifier`, `ValidationEngine`, orchestrated by `Pipeline`.
`app/ai/stubs.py` provides placeholder implementations so the system runs end-to-end
today. To integrate real models later, implement a stage (e.g. a YOLO-backed
`Detector` reading `YOLO_MODEL`) and wire it in `app/ai/factory.py` — no other code
needs to change.

## Testing

```
uv sync --extra dev
uv run pytest
```

Tests run against an isolated in-memory SQLite database and fake storage/AI doubles —
no live Supabase project or network access is required.

## Security notes

- Never store image binaries in Postgres — only Storage paths.
- The public API exposes each event's `uuid` as `id`; the internal bigint primary key
  is never returned, to avoid sequential-ID enumeration.
- Uploaded `image`/`thumbnail` parts are validated for content-type (`image/*`) and a
  maximum size (`MAX_UPLOAD_SIZE_BYTES`) before being accepted.
