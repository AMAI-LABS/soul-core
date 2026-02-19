//! # Recursive Language Model (RLM) — Infinite Context Engine
//!
//! Inspired by the Recursive Language Model paper ([arXiv:2512.24601](https://arxiv.org/abs/2512.24601)),
//! extended to serve as the **context-as-document** memory layer for autonomous agents.
//!
//! ## Core Design Principle
//!
//! **The agent's own conversation history IS the document.**
//!
//! RLM is NOT a one-off document analysis tool. It is the agent's memory management system
//! that enables infinite context — the conversation grows forever without destructive
//! compaction. Every turn, every tool result, every subagent's output becomes part of a
//! growing external document that the agent navigates via a structured REPL.
//!
//! ```text
//! Traditional Agent:  context window fills → truncate/compact → information lost
//! RLM Agent:          context grows forever → externalized as document → REPL navigates it
//! ```
//!
//! ## How It Works
//!
//! 1. **Each turn**, the full conversation history is serialized into a structured text
//!    document (the "context"). This includes user messages, LLM responses, tool calls,
//!    tool results, and subagent outputs.
//!
//! 2. **The LLM** receives only a compact window: the system prompt, recent messages,
//!    and context metadata (size, structure). It does NOT receive the full history directly.
//!
//! 3. **To access its past**, the LLM writes DSL commands in ` ```rlm` ``` blocks:
//!    QUERY to ask questions about history, CHUNK to split it, MAP to process sections,
//!    SLICE to view specific regions.
//!
//! 4. **Nothing is ever lost.** The external document only grows. The agent can always
//!    navigate back to any prior turn, tool result, or decision point.
//!
//! ## DSL Commands
//!
//! | Command | Description |
//! |---------|-------------|
//! | `QUERY target "prompt"` | Ask a sub-LLM a question about a variable |
//! | `QUERY_BATCH target source "prompt"` | Query each item in a list |
//! | `CHUNK target source BY_LINES n` | Split text into chunks by line count |
//! | `CHUNK target source BY_CHARS n` | Split text into chunks by character count |
//! | `CHUNK target source BY_REGEX "pattern"` | Split on regex boundaries |
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
//! - [`RlmEngine`] — Orchestrates the context-as-document loop: serialize history → LLM
//!   generates DSL → parse → execute → feedback → iterate
//!
//! ## Context-as-Document vs SemanticContextEngine
//!
//! Both solve infinite context, through different mechanisms:
//!
//! | Aspect | RLM | SemanticContextEngine |
//! |--------|-----|----------------------|
//! | Model | External document + DSL REPL | Knowledge graph + symlink compression |
//! | Navigation | LLM writes QUERY/CHUNK/MAP | Vector search + graph neighbors |
//! | Compaction | None needed — document grows | Deactivation + symlink refs |
//! | LLM agency | High — LLM controls exploration | Low — engine auto-selects |
//! | Token cost | Higher (REPL iterations) | Lower (compressed references) |
//!
//! Both can be benchmarked via [`AgentLoop`](crate::agent::AgentLoop)'s `ContextStrategy`.

mod dsl;
mod engine;
mod environment;
mod recall_tool;

pub use dsl::{DslCommand, DslError, DslParser};
pub use engine::{RlmConfig, RlmEngine, RlmIteration, RlmResult};
pub use environment::{RlmEnvironment, Variable};
pub use recall_tool::{RecallTool, SharedContextDoc};
