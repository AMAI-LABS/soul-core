//! # soul-core
//!
//! Async agentic runtime for Rust — the engine that powers autonomous agent loops
//! with steerable execution, multi-provider LLM abstraction, and semantic context management.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use soul_core::semantic_recursion::{SemanticContextEngine, RetrievalQuery};
//!
//! let mut engine = SemanticContextEngine::new();
//!
//! // Ingest a conversation
//! let req = engine.ingest_user_request("Explain Rust async");
//! let resp = engine.ingest_llm_response("Rust async uses tokio...", req, Some("claude"));
//!
//! // Symlink for compact references
//! let hash = engine.symlink_node(req).unwrap();
//! // hash is a 6-char hex like "A3F2B1"
//!
//! // Fragment long messages into semantic clusters
//! let fragments = engine.symlink_fragmented(resp);
//! // Each cluster gets its own symlink hash
//!
//! // Retrieve relevant context with token budget
//! let window = engine.retrieve(&RetrievalQuery {
//!     text: "async programming".into(),
//!     max_tokens: 4000,
//!     max_results: 5,
//!     include_graph_neighbors: true,
//! });
//! ```
//!
//! ## Architecture
//!
//! The crate is organized into these core modules:
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`types`] | Core types: `Message`, `Role`, `ContentBlock`, `ToolDefinition`, `AgentConfig` |
//! | [`provider`] | Multi-provider LLM abstraction (Anthropic, OpenAI) with SSE streaming |
//! | [`agent`] | Steerable agent loop with tool execution, compaction triggers, interruption |
//! | [`context`] | Token-aware context window management with compaction and circuit breaker |
//! | [`session`] | JSONL-based session persistence with lane serialization |
//! | [`tool`] | Async tool trait and registry for agent tool execution |
//! | [`hook`] | Three-tier hook pipeline: modifying (sequential), void (parallel), persist (hot path) |
//! | [`subagent`] | Subagent spawner for parallel task delegation |
//! | [`memory`] | Hierarchical memory (MEMORY.md + topic files, bootstrap files) |
//! | [`error`] | Error types with thiserror: Provider, RateLimited, Auth, ContextOverflow, etc. |
//! | [`rlm`] | Recursive Language Model engine — custom DSL for document search (arxiv 2512.24601) |
//! | [`semantic_recursion`] | Context social graph, TF-IDF vector search, symlinks, semantic fragmentation |
//!
//! ## Context Management: The Core Innovation
//!
//! The [`semantic_recursion`] module implements a "social graph for LLM context":
//!
//! - **Graph nodes** represent every piece of context: user requests, LLM responses,
//!   tool calls, external data, compaction summaries
//! - **Graph edges** encode relationships: RespondsTo, FollowsInSequence, CompactedInto,
//!   TriggeredTool, ProvidesData, DerivedFrom
//! - **Nothing is ever deleted** — compaction deactivates nodes but preserves them with
//!   CompactedInto edges. `full_ancestry()` traces through any depth back to originals
//! - **Symlinks** replace messages with 6-char hash references (`[A3F2B1]: summary`).
//!   LLMs can call `resolve_symlink` tool to expand any hash
//! - **Semantic fragmentation** splits messages into clusters via agglomerative clustering
//!   on TF-IDF embeddings. Close sentences group together, distant content gets separate symlinks
//! - **Token-budgeted retrieval** combines keyword relevance, vector similarity, and graph
//!   neighbor expansion to craft optimal context windows per task

pub mod agent;
pub mod context;
pub mod cost;
pub mod error;
pub mod executor;
pub mod hook;
pub mod mcp;
pub mod memory;
pub mod permission;
pub mod provider;
pub mod rlm;
pub mod semantic_recursion;
pub mod session;
pub mod skill;
pub mod soullog;
pub mod subagent;
pub mod tool;
pub mod types;
pub mod vexec;
pub mod vfs;

pub use error::SoulError;
pub use types::*;
