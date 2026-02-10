# soul-core

Async agentic runtime for Rust. Steerable agent loops, multi-provider LLM abstraction, semantic context management, virtual filesystem, and WASM-ready architecture.

[![Crates.io](https://img.shields.io/crates/v/soul-core.svg)](https://crates.io/crates/soul-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What is this?

`soul-core` is the engine that powers autonomous agent loops. It provides everything needed to build LLM-powered agents that can reason, use tools, manage context, persist sessions, and run on native OS or in a browser via WASM.

```rust
use soul_core::agent::{AgentLoop, RunOptions};
use soul_core::provider::anthropic::AnthropicProvider;
use soul_core::tool::ToolRegistry;
use soul_core::types::*;

// Configure
let model = ModelInfo::new("claude-sonnet-4-5-20250929", ProviderKind::Anthropic);
let config = AgentConfig::new(model, "You are a helpful assistant.");
let provider = AnthropicProvider::new();

// Build
let tools = ToolRegistry::new();
let mut agent = AgentLoop::new(Arc::new(provider), tools, config);

// Run
let (event_tx, event_rx) = mpsc::unbounded_channel();
let (steering_tx, steering_rx) = mpsc::unbounded_channel();
let options = RunOptions {
    session_id: "session-1".into(),
    initial_messages: vec![Message::user("Hello")],
};
let messages = agent.run(options, event_tx, steering_rx).await?;
```

## Architecture

| Module | Purpose |
|--------|---------|
| `agent` | Steerable agent loop with tool execution, compaction triggers, interruption |
| `provider` | Multi-provider LLM abstraction (Anthropic, OpenAI) with SSE streaming |
| `context` | Token-aware context window management with compaction and circuit breaker |
| `tool` | Async tool trait and registry |
| `hook` | Three-tier hook pipeline: modifying, void, persist |
| `permission` | Rule-based permission gate with glob patterns and risk levels |
| `cost` | Token/USD cost tracking with budget enforcement |
| `mcp` | Model Context Protocol client with JSON-RPC transport |
| `skill` | Config-driven tool definitions loaded from `.skill`/`.md` files |
| `executor` | Pluggable tool execution backends (shell, HTTP, MCP, LLM) |
| `session` | JSONL session persistence with lane serialization |
| `memory` | Hierarchical memory (MEMORY.md + topic files, bootstrap files) |
| `vfs` | Virtual filesystem: MemoryFs (WASM/tests), NativeFs (OS) |
| `vexec` | Virtual executor: MockExecutor (WASM/tests), NativeExecutor (OS) |
| `planner` | Task graph with dependencies, status tracking, timing, and display rendering |
| `snapshot` | Generic versioned snapshot log — append-only JSONL history with rollback |
| `soullog` | Multi-sink structured logging |
| `subagent` | Subagent spawner for parallel task delegation |
| `semantic_recursion` | Context graph, TF-IDF search, symlinks, semantic fragmentation |
| `rlm` | Recursive Language Model engine (document search DSL) |
| `types` | Core types: Message, Role, ContentBlock, ToolDefinition, AgentConfig |
| `error` | Error types with thiserror |

## Key Features

### Agent Loop
The core `AgentLoop` runs a think-act-observe cycle: send messages to an LLM, execute any tool calls, feed results back, repeat until the LLM responds without tool calls or a budget/turn limit is hit.

- Real-time event streaming via `mpsc` channels
- Mid-loop steering (inject messages, interrupt)
- Automatic context compaction when approaching token limits
- Optional cost tracking and budget enforcement

### Virtual Filesystem
All storage (sessions, memory, skills) goes through `VirtualFs`, a trait with two implementations:

- **`NativeFs`** — real OS filesystem via `tokio::fs` (behind `native` feature)
- **`MemoryFs`** — in-memory BTreeMap (works everywhere, including WASM)

Custom implementations can target IndexedDB, S3, or anything else.

### Virtual Executor
Shell commands go through `VirtualExecutor`:

- **`NativeExecutor`** — real subprocesses via `tokio::process` (behind `native` feature)
- **`MockExecutor`** — canned responses for testing
- **`NoopExecutor`** — returns error (for WASM environments)

### Skills
Define tools as `.skill` or `.md` files with YAML frontmatter:

```markdown
---
name: search_codebase
description: Search for a pattern
input_schema:
  type: object
  properties:
    pattern:
      type: string
  required: [pattern]
execution:
  type: shell
  command_template: "rg '{{pattern}}' --json"
  timeout_secs: 30
---
Search the codebase using ripgrep.
```

Load and register as tools:
```rust
let loader = SkillLoader::new(fs, "skills");
loader.load_all().await?;
loader.register_all_as_tools(&mut registry, executor);
```

### Hook Pipeline
Three tiers of hooks modify agent behavior without changing the loop:

- **Modifying hooks** — run sequentially, can alter or cancel operations (permission checks, prompt injection)
- **Void hooks** — run in parallel, fire-and-forget (logging, metrics)
- **Persist hooks** — synchronous transforms on the hot path (redaction, storage)

### Semantic Context Management
The `semantic_recursion` module implements a social graph for LLM context:

- Graph nodes for every piece of context (requests, responses, tool calls, compactions)
- Symlinks replace messages with 6-char hash references that LLMs can resolve
- Semantic fragmentation splits long messages into clusters via agglomerative clustering on TF-IDF embeddings
- Token-budgeted retrieval combines keyword relevance, vector similarity, and graph neighbor expansion

### Permission System
Rule-based permission gate with glob patterns:

```rust
let mut manager = PermissionManager::new();
manager
    .add_rule(PermissionRule::allow("read_*"))
    .add_rule(PermissionRule::deny("rm_*").with_reason("Destructive"))
    .classify_risk("write_*", RiskLevel::Execution);
```

### Cost Tracking
Per-turn token counting with model-specific pricing and budget enforcement:

```rust
let tracker = CostTracker::new("session-1");
let enforcer = BudgetEnforcer::new(BudgetPolicy {
    hard_limit_usd: Some(5.0),
    ..Default::default()
});
let agent = AgentLoop::new(provider, tools, config)
    .with_cost_tracker(tracker)
    .with_budget(enforcer);
```

### MCP Client
Connect to Model Context Protocol servers and expose their tools:

```rust
let transport = StdioTransport::new("npx", &["-y", "@modelcontextprotocol/server-filesystem"]);
let mut client = McpClient::new(Box::new(transport));
client.initialize().await?;
client.register_tools(&mut registry).await?;
```

### OAuth Provider
The Anthropic provider supports both API key and OAuth token auth, auto-detected from the key prefix:

```rust
use soul_core::provider::AnthropicProvider;
use soul_core::types::AuthProfile;

// API key auth (default)
let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-api03-...");

// OAuth token auth (auto-detected from sk-ant-oat prefix)
let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-oat01-...");
// Automatically adds: Bearer auth, beta headers, metadata.user_id, tool name remap
```

OAuth tokens trigger automatic tool name remapping (greek_nature combos) to bypass Anthropic's semantic tool name filter.

### Proxy Support (WASM / Browser)
All providers work in WASM browser environments via a transparent proxy:

```rust
use soul_core::provider::{ProxyConfig, AnthropicProvider, OpenAIProvider};

// Route through a transparent proxy (e.g. for WASM browser agents)
let proxy = ProxyConfig::new("https://your-app.example.com/api");
let anthropic = AnthropicProvider::with_base_url(proxy.anthropic_url());
let openai = OpenAIProvider::with_base_url(proxy.openai_url());

// Or passthrough mode (proxy is a direct stand-in for one API)
let proxy = ProxyConfig::passthrough("http://localhost:8081");
let anthropic = AnthropicProvider::with_base_url(proxy.anthropic_url());
```

## WASM Support

The library compiles to WebAssembly. Use `--no-default-features --features wasm`:

```toml
[dependencies]
soul-core = { version = "0.7", default-features = false, features = ["wasm"] }
```

In WASM mode:
- `MemoryFs` replaces `NativeFs`
- `NoopExecutor` replaces `NativeExecutor`
- LLM providers work via configurable `base_url` pointing to a transparent proxy
- `reqwest` uses fetch API
- `uuid` uses `js` feature for browser crypto

## Installation

```toml
# Native (default) — full OS support
[dependencies]
soul-core = "0.7"

# WASM — browser target
[dependencies]
soul-core = { version = "0.7", default-features = false, features = ["wasm"] }

# Minimal — no runtime, no filesystem
[dependencies]
soul-core = { version = "0.7", default-features = false }
```

## Testing

```bash
cargo test                           # 648 tests (628 unit + 14 integration + 6 doc)
cargo test --no-default-features     # verify non-native builds
cargo clippy -- -D warnings          # lint
cargo fmt --check                    # format check
cargo doc --no-deps                  # build docs
```

## License

MIT
