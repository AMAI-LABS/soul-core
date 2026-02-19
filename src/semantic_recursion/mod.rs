//! # Semantic Recursion
//!
//! Context social graph for LLM session management — the core context management
//! engine that treats every piece of conversation as a node in a relationship graph.
//!
//! ## Design Philosophy
//!
//! Traditional LLM context management truncates or drops old messages when the window fills.
//! Semantic recursion takes a different approach: **nothing is ever lost**. Instead:
//!
//! 1. **Compaction** deactivates old nodes but preserves them with `CompactedInto` edges
//! 2. **Symlinks** replace verbose content with 6-char hash references (`[A3F2B1]: summary`)
//! 3. **Fragmentation** splits messages into semantic clusters — close content groups together,
//!    distant content gets separate symlinks
//! 4. **Retrieval** combines keyword scoring, TF-IDF vector search, and graph neighbor expansion
//!    to craft optimal context windows per task
//!
//! ## Core Components
//!
//! - [`SemanticContextEngine`] — The main engine combining all components
//! - [`ContextGraph`] — Social graph with typed nodes and edges
//! - [`VectorStore`] — In-memory TF-IDF vector store for semantic search
//! - [`Tokenizer`] — Pure-Rust whitespace tokenizer with vocabulary management
//! - [`SymlinkStore`] — Maps 6-char hashes to full content
//! - [`fragment_message`] — Splits messages into semantic clusters via agglomerative clustering
//!
//! ## Symlink + Fragmentation Flow
//!
//! ```text
//! Long message with multiple topics
//!     │
//!     ▼
//! split_sentences() → ["Rust async...", "Python data...", "Rust traits..."]
//!     │
//!     ▼
//! TF-IDF embeddings per sentence
//!     │
//!     ▼
//! Agglomerative clustering (cosine similarity)
//!     │
//!     ├── Cluster 0: ["Rust async...", "Rust traits..."]  → [A3F2B1]: Rust discussion
//!     └── Cluster 1: ["Python data..."]                   → [C4D5E6]: Python mention
//! ```
//!
//! ## Tools for LLM
//!
//! Three tools are provided for the LLM to interact with symlinks:
//!
//! - [`ResolveSymlinkTool`] — Expand hash references back to full content
//! - [`SearchSymlinksTool`] — Search symlinks by keyword
//! - [`ListSymlinksTool`] — List all active symlinks with token savings

mod context_engine;
pub mod fragment;
mod graph;
pub mod symlink;
pub mod symlink_tool;
mod tokenizer;
mod vector_store;

pub use context_engine::{
    ContextWindow, EngineStats, RetrievalQuery, SemanticContextEngine, SymlinkedContextWindow,
};
pub use fragment::{fragment_message, split_sentences, Fragment, FragmentConfig, FragmentResult};
pub use graph::{ContextGraph, EdgeKind, GraphEdge, GraphNode, NodeId, NodeKind};
pub use symlink::{auto_summary, ResolveResult, Symlink, SymlinkStore};
pub use symlink_tool::{ListSymlinksTool, ResolveSymlinkTool, SearchSymlinksTool};
pub use tokenizer::{Token, TokenizedText, Tokenizer};
pub use vector_store::{Embedding, SearchResult, VectorStore};
