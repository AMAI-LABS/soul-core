//! # Recursive Language Model (RLM)
//!
//! Implementation of the RLM paper (arXiv:2512.24601) in Rust.
//!
//! Instead of Python REPL, uses a custom mini DSL for document operations.
//! The LLM writes DSL commands to examine, chunk, query sub-LLMs,
//! and aggregate results over arbitrarily large contexts.

mod dsl;
mod environment;
mod engine;

pub use dsl::{DslCommand, DslParser, DslError};
pub use environment::{RlmEnvironment, Variable};
pub use engine::{RlmEngine, RlmConfig, RlmResult, RlmIteration};
