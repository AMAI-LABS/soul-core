# Changelog

All notable changes to `soul-core` will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.1] - 2026-02-09

### Added
- `snapshot` module — generic `SnapshotLog<T>` for versioned state persistence
  - Append-only JSONL via `VirtualFs`, works on native and WASM
  - `VersionedSnapshot<T>` wraps any serializable type with version, timestamp, label
  - `VersionInfo` for lightweight history queries
  - Rollback preserves full audit trail (appends old version as new entry)
  - 16 tests covering planner snapshots, custom structs, rollback chains

### Fixed
- WASM-compatible `async_trait(?Send)` on all trait definitions

## [0.10.0] - 2026-02-09

### Added
- Same as 0.10.1 (published before snapshot module was integrated)

## [0.9.0] - 2026-02-09

### Added
- Planner retry and resume for crash recovery
  - `PlanTask` fields: `attempt`, `max_retries`, `last_error`, `output` (all `serde(default)`)
  - `can_retry()` method on `PlanTask`
  - `Planner` methods: `fail_with_error()`, `retry()`, `retry_all()`, `resume()`, `checkpoint()`, `set_max_retries()`
  - `resume()` resets InProgress tasks to Pending with attempt increment (handles laptop close / process kill)
  - `render()` shows `(attempt N/M)` for failed tasks with retries remaining
  - 19 new tests (81 total planner)

## [0.8.0] - 2026-02-09

### Added
- Planner clarifying questions for complex plans
  - `QuestionId` type, `PlanQuestion` struct with question text, options, answer, related tasks
  - `Planner` methods: `add_question()`, `add_question_with_options()`, `link_question_to_task()`, `answer_question()`, `get_question()`, `all_questions()`, `open_questions()`, `has_open_questions()`, `is_ready_to_execute()`
  - `PlannerSnapshot` extended with `questions` and `next_question_id` (serde(default) for backward compat)
  - `render()` shows open questions with `?` prefix and options in brackets
  - 14 new tests

## [0.7.2] - 2026-02-09

### Changed
- Fixed Cargo.toml authors to `"AMAI Labs <team@amai.net>"`

## [0.7.1] - 2026-02-09

### Changed
- Added authors field to Cargo.toml

## [0.7.0] - 2026-02-09

### Added
- OAuth provider support for Anthropic (auto-detected from `sk-ant-oat` prefix)
  - Bearer token auth with `anthropic-beta: oauth-2025-04-20` header
  - Automatic tool name remap (greek_nature combos) to bypass semantic filter
- `ToolRemap` struct for bidirectional tool name mapping (outbound + inbound)
- `ProxyConfig` for routing provider requests through transparent proxy (WASM browser support)
  - `ProxyConfig::new()` with separate `/anthropic` and `/openai` path prefixes
  - `ProxyConfig::passthrough()` for direct stand-in proxies (e.g. mock-llm-service)

### Changed
- Providers (`AnthropicProvider`, `OpenAIProvider`, `ProviderRegistry`) ungated from `native` feature — now available on all targets including WASM

## [0.6.0] - 2026-02-09

### Added
- `planner` module — task graph with dependencies, status tracking, timing, display rendering
  - `Planner`, `PlanTask`, `TaskStatus`, `PlannerSnapshot` types
  - DAG dependency tracking with cycle detection
  - `render()` for visual task display
  - 48 tests
- `MemoryFsSnapshot` — serialize/restore in-memory filesystem state
- WASM feature gates and multi-platform CI

## [0.5.0] - 2026-02-09

### Added
- `vfs` module — `VirtualFs` trait with `NativeFs` (OS) and `MemoryFs` (WASM/tests)
- `vexec` module — `VirtualExecutor` trait with `NativeExecutor`, `MockExecutor`, `NoopExecutor`
- `permission` module — rule-based permission gate with glob patterns and risk levels
- `cost` module — per-turn token counting, model-specific pricing, budget enforcement
- `mcp` module — Model Context Protocol client with JSON-RPC transport (stdio + HTTP)
- `skill` module — config-driven tool definitions from `.skill`/`.md` files with YAML frontmatter
- `executor` module — pluggable tool execution backends (shell, HTTP, MCP, LLM, direct)
- `session` module — JSONL session persistence with lane serialization
- `memory` module — hierarchical memory (MEMORY.md + topic files, bootstrap files)
- `soullog` module — multi-sink structured logging

## [0.3.0] - 2026-02-09

### Added
- `semantic_recursion` module — context social graph, TF-IDF search, symlinks, semantic fragmentation
- `rlm` module — Recursive Language Model engine (document search DSL)
- CI pipeline with clippy, fmt, test, doc checks

## [0.2.0] - 2026-02-09

### Added
- `agent` module — steerable agent loop with tool execution, compaction triggers, interruption
- `provider` module — multi-provider LLM abstraction (Anthropic, OpenAI) with SSE streaming
- `context` module — token-aware context window management with compaction
- `tool` module — async tool trait and registry
- `hook` module — three-tier hook pipeline (modifying, void, persist)
- `subagent` module — subagent spawner for parallel task delegation
- Core types: `Message`, `Role`, `ContentBlock`, `ToolDefinition`, `AgentConfig`

## [0.1.0] - 2026-02-09

### Added
- Initial release — async agentic runtime for Rust
- Core type definitions and error types
