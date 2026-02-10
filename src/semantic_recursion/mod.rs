//! # Semantic Recursion
//!
//! Context social graph for LLM session management.
//!
//! - Tokenization + in-memory vector store for semantic search
//! - Relationship graph between context pieces, requests, and responses
//! - Growing memory that never loses anything — compaction builds graph edges, not deletions
//! - Refined message crafting per task based on relevance scoring
//! - **Symlinks**: compact 6-char hash references to full content
//! - **Fragmentation**: k-neighbors clustering splits messages into semantic fragments,
//!   close clusters become one symlink, distant content gets separate symlinks

mod tokenizer;
mod vector_store;
mod graph;
mod context_engine;
pub mod symlink;
pub mod symlink_tool;
pub mod fragment;

pub use tokenizer::{Tokenizer, Token, TokenizedText};
pub use vector_store::{VectorStore, Embedding, SearchResult};
pub use graph::{ContextGraph, NodeId, NodeKind, EdgeKind, GraphNode, GraphEdge};
pub use context_engine::{SemanticContextEngine, ContextWindow, RetrievalQuery, EngineStats};
pub use symlink::{SymlinkStore, Symlink, ResolveResult, auto_summary};
pub use symlink_tool::{ResolveSymlinkTool, SearchSymlinksTool, ListSymlinksTool};
pub use fragment::{Fragment, FragmentResult, FragmentConfig, fragment_message, split_sentences};
