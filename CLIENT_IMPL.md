# Rust Client Rewrite

## Top-level modules to recreate/decouple
- config: structs for API key, base URL, retry/backoff, pagination policy, per-endpoint fields; loadable from env/file
- http_client: thin wrapper over reqwest with request builder, auth header injection, redirect handling, retry/backoff, JSON body extraction
- pagination: FSM + helpers to append page/offset/cursor params and track cursor/has_more/max_pages; shared by crawlers
- models: typed structs for each API item (e.g., StationRecord) with serde derives; newtype IDs to avoid stringly usage
- storage: repository trait + sqlite implementation; wraps connection pool/transactions and exposes ingest/upsert ops
- json_handlers: pure functions that map raw JSON objects into `PageBatch<T>` and enforce field/alias mapping
- crawler: generic over item type + handlers; orchestrates fetch -> parse -> append -> storage write; owns metadata/config
- runner: queue/executor for seeds -> endpoints; handles stop, idle callbacks, concurrency; built on tokio tasks or async channels
- cli/bin: small entrypoint wiring config, concrete models/handlers, DB path, and spawning runner
- telemetry: logging/span helpers, metrics counters for requests/retries/errors/pages processed

## Decoupling notes
- Keep http_client stateless; pass config/context into methods instead of storing mutable state
- Define traits for storage and parsing so crawler/runner are agnostic of DB and payload shapes
- Keep pagination policy data-driven (config structs) to avoid per-API branching in crawler core
- Model construction/validation happens in json_handlers; crawler only understands `PageBatch<T>`
- Runner operates on trait objects or generics over Seed->Vec<Endpoint> so scheduler logic is reusable across clients
- Separate error types per layer but convert to a shared error enum for public API

## Quick mapping from C++ pieces to Rust
- `http.*` -> http_client module using reqwest + url; let reqwest handle TLS
- `crawler_support.*` -> config + pagination modules with builder helpers for URLs
- `crawler.*` -> crawler module with generic handlers and retry logic; runner module for background worker/queue
- `database.*` -> storage module with a `Storage` trait and sqlite-backed impl using rusqlite/sqlx
- `climate.cpp` specifics -> models + handlers submodules; CLI wires base_url/api_key/db path and kicks off runner

## Remaining generic scaffolding needed
- storage: implement a `Storage` trait using sqlx/sqlite (pooled connections, migrations, upsert helpers, retries/backoff on busy) plus a thin repository layer to keep crawlers DB-agnostic
- endpoint model: shared `Endpoint` struct + builders for path/query/body and per-endpoint retry/ratelimit overrides to replace the C++ Endpoint/builder glue
- job wiring: a generic seed->endpoint adapter that constructs `Job` instances for the runner so specific clients only provide seeds and parsers
- config I/O: loader/validator for `ClientConfig` from file/env, and constructors for `HttpClient`, `Pager`, `Runner` derived from config
- telemetry: tracing initialization helpers (JSON/plain), metrics counters/timers for requests/retries/pages/items, and hooks for runner/crawler to report errors/idle
- rate limiting: reusable governor/tower limiter wrapper integrated with `HttpClient` to enforce per-endpoint or global RPM/burst
