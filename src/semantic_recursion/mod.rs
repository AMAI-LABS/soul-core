//! # Semantic Recursion
//!
//! Context social graph for LLM session management.
//!
//! - Tokenization + in-memory vector store for semantic search
//! - Relationship graph between context pieces, requests, and responses
//! - Growing memory that never loses anything — compaction builds graph edges, not deletions
//! - Refined message crafting per task based on relevance scoring

mod tokenizer;
mod vector_store;
mod graph;
mod context_engine;

pub use tokenizer::{Tokenizer, Token, TokenizedText};
pub use vector_store::{VectorStore, Embedding, SearchResult};
pub use graph::{ContextGraph, NodeId, NodeKind, EdgeKind, GraphNode, GraphEdge};
pub use context_engine::{SemanticContextEngine, ContextWindow, RetrievalQuery};
