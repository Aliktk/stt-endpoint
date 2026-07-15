# Design Decisions

This document covers the engineering choices behind the pipeline (Part 1) and
the system-design questions (Part 2). Where a Part 2 answer is already
demonstrated by the running code, that link is called out.

## Part 1 — pipeline choices

**API-first with a local fallback.** ElevenLabs is the primary backend, Deepgram
secondary, and faster-whisper the final local fallback. Cloud APIs give the best
accuracy and need no local GPU; the local model guarantees the service still
works with no keys and no network, and keeps data on the machine. The fallback
order is configuration, not code (`PROVIDER_ORDER`), so it can be reordered per
deployment.

**One normalized result.** Every provider returns the same
`TranscriptionResult` (`language`, `text`, `segments[]`, `provider`,
`duration`). Callers never branch on which backend answered. ElevenLabs exposes
only word-level timing, so its provider groups words into segments on natural
pauses; Deepgram and whisper expose segments/utterances natively.

**Timestamps per segment.** Segment-level timing is the clean default for
captions and navigation. Word-level timing is available from the underlying
engines and could be surfaced later without changing the pipeline shape.

**Formats.** Decoding is delegated to ffmpeg (via faster-whisper and pydub),
which covers WAV, MP3, M4A, FLAC, OGG, WEBM, and more, and normalizes to 16 kHz
mono internally. We validate existence, extension, and size up front so bad
input fails fast with a clear error instead of deep inside a decoder.

**Long files.** Split on silence into offset-tagged chunks, transcribe each, and
merge with corrected timestamps. This bounds memory, makes per-chunk retries
possible, and is the same idea that scales to background workers in production.

## Part 2 — system design

### How would you handle concurrent uploads?

Keep transcription out of the request path — it is CPU/GPU-bound and slow. The
API accepts the upload, writes the audio to storage, creates a job record, and
returns `202 Accepted` with a `job_id`. A pool of background workers
(Celery / RQ / Huey / Dramatiq) consumes a queue and transcribes. Concurrency is
then two independent dials: API replicas for upload throughput, and worker count
for transcription throughput, sized to available CPU/GPU.

*In this demo:* the provider abstraction, validation, and chunking that a worker
would run already exist as `TranscriptionService`; only the queue/worker plumbing
is deferred.

### How would you store audio and transcripts?

Audio is large binary data → object storage (S3 / GCS / Azure Blob), referenced
by key. Transcripts and metadata → PostgreSQL: job id, status, provider,
language, duration, timestamps, and the audio key. Segment lists fit naturally in
a `JSONB` column for easy retrieval; if full-text search over transcripts is
needed, index the text column or push it to a search engine.

### How do you retry or recover failed transcriptions?

Each job carries a status: `queued → processing → completed | failed`. On failure
the worker logs the error with context, increments a retry counter, and the queue
re-runs it with exponential backoff, capped at ~3 attempts. After that the job is
marked `failed` but the original audio is retained so it can be replayed manually.

*In this demo:* the provider chain is the first line of recovery — a failing
provider is caught and the next is tried before the request is ever considered
failed.

### How would you expose this as an API?

FastAPI. `POST /v1/transcribe` to submit (returning `202 + job_id` in the async
design; the demo returns the result synchronously), `GET /v1/jobs/{id}` to poll
status and fetch the transcript. Cross-cutting concerns: API-key/JWT auth, rate
limiting, request size limits and format validation, consistent error envelopes,
and `/v1` versioning. The demo already implements versioning, validation, the
error envelope, and correct status codes (400 / 422 / 502).

## Trade-offs & what I'd add for production

- **Async job pipeline** — queue + workers + `202`/polling (or webhooks), as above.
- **Persistence** — Postgres + object storage instead of temp files.
- **Observability** — structured logs, request IDs, metrics (latency, provider
  hit/fallback rates, failure counts), tracing.
- **Security** — auth, rate limiting, per-tenant quotas, signed upload URLs.
- **Cost controls** — route by file length/language, cache repeat inputs by
  content hash, prefer the local model for cheap/bulk work.
- **Word-level output & diarization** — surface word timings and speaker labels
  where the backend supports them.
