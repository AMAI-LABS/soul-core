//! # Recursive Language Model (RLM)
//!
//! Implementation of the Recursive Language Model paper ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601))
//! in pure Rust with a custom mini DSL for document operations.
//!
//! ## Key Idea
//!
//! The paper shows that LLMs can extend their effective context window 100x by treating
//! the prompt as an external environment variable and using a REPL loop to query, chunk,
//! and aggregate results over arbitrarily large documents.
//!
//! Instead of Python REPL (as in the paper), soul-core provides a custom Rust DSL that
//! the LLM writes inside ` ```rlm` ``` blocks.
//!
//! ## DSL Commands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `QUERY target "prompt"` | Ask a sub-LLM a question about a variable |
//! | `QUERY_BATCH target source "prompt"` | Query each item in a list |
//! | `CHUNK target source BY_LINES n` | Split text into chunks by line count |
//! | `CHUNK target source BY_CHARS n` | Split text into chunks by character count |
//! | `SLICE target source start end` | Extract a substring |
//! | `LEN target source` | Get length (chars for text, items for list) |
//! | `JOIN target source "separator"` | Join list items into text |
//! | `GET target source` | Copy a variable |
//! | `CONCAT target a b` | Concatenate two texts |
//! | `INDEX target source n` | Get nth item from list |
//! | `MAP target source "template"` | Transform each list item via sub-LLM |
//! | `FILTER target source "condition"` | Filter list items via sub-LLM |
//! | `PRINT var_name` | Print a variable's contents |
//! | `SHOW_VARS` | Show all variables and their types |
//! | `FINAL var_name` | Return a variable as the final answer |
//! | `FINAL_TEXT "text"` | Return literal text as the final answer |
//!
//! ## Components
//!
//! - [`DslParser`] — Parses DSL commands from text, extracts ` ```rlm` ``` blocks from LLM responses
//! - [`RlmEnvironment`] — Variable store that executes DSL commands (text/list/number types)
//! - [`RlmEngine`] — Completion loop: LLM generates DSL → parse → execute → feedback → iterate

mod dsl;
mod engine;
mod environment;

pub use dsl::{DslCommand, DslError, DslParser};
pub use engine::{RlmConfig, RlmEngine, RlmIteration, RlmResult};
pub use environment::{RlmEnvironment, Variable};
