//! # soul-core
//!
//! Async agentic runtime for Rust.
//!
//! Implements steerable agent loops, multi-provider LLM abstraction,
//! context management with compaction, session persistence, memory hierarchy,
//! tool execution pipelines, hook systems, and subagent orchestration.

pub mod types;
pub mod provider;
pub mod agent;
pub mod context;
pub mod session;
pub mod tool;
pub mod hook;
pub mod subagent;
pub mod memory;
pub mod error;

pub use types::*;
pub use error::SoulError;
