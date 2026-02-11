//! RLM Environment — manages variables, executes DSL commands

use std::collections::HashMap;

use regex::Regex;

use crate::error::{SoulError, SoulResult};

use super::dsl::DslCommand;

/// A variable in the RLM environment
#[derive(Debug, Clone)]
pub enum Variable {
    /// Single text value
    Text(String),
    /// List of text values (chunks, results, etc.)
    List(Vec<String>),
    /// Numeric value
    Number(usize),
}

impl Variable {
    pub fn as_text(&self) -> String {
        match self {
            Variable::Text(s) => s.clone(),
            Variable::List(v) => v.join("\n"),
            Variable::Number(n) => n.to_string(),
        }
    }

    pub fn as_list(&self) -> Vec<String> {
        match self {
            Variable::Text(s) => vec![s.clone()],
            Variable::List(v) => v.clone(),
            Variable::Number(n) => vec![n.to_string()],
        }
    }

    pub fn type_name(&self) -> &str {
        match self {
            Variable::Text(_) => "text",
            Variable::List(_) => "list",
            Variable::Number(_) => "number",
        }
    }
}

/// Execution result from a single DSL command
#[derive(Debug, Clone)]
pub enum ExecResult {
    /// Command executed, produced output
    Output(String),
    /// Command produced no visible output
    Silent,
    /// Command requests an LLM query
    QueryRequest {
        target: String,
        prompt: String,
        context: String,
    },
    /// Command requests batched LLM queries
    QueryBatchRequest {
        target: String,
        prompts: Vec<String>,
        contexts: Vec<String>,
    },
    /// Command requests map over list with LLM
    MapRequest {
        target: String,
        items: Vec<String>,
        prompt_template: String,
    },
    /// Command requests filter with LLM
    FilterRequest {
        target: String,
        items: Vec<String>,
        condition: String,
    },
    /// Final answer found
    FinalAnswer(String),
    /// Show variables
    VarList(String),
}

/// The RLM environment — holds variables and executes DSL commands
pub struct RlmEnvironment {
    variables: HashMap<String, Variable>,
    output_log: Vec<String>,
}

impl RlmEnvironment {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            output_log: Vec::new(),
        }
    }

    /// Load context into the environment
    pub fn load_context(&mut self, context: String) {
        self.variables
            .insert("context".into(), Variable::Text(context));
    }

    /// Load context as a list of documents
    pub fn load_context_list(&mut self, documents: Vec<String>) {
        self.variables
            .insert("context".into(), Variable::List(documents));
    }

    /// Set a variable
    pub fn set_var(&mut self, name: &str, value: Variable) {
        self.variables.insert(name.to_string(), value);
    }

    /// Get a variable
    pub fn get_var(&self, name: &str) -> Option<&Variable> {
        self.variables.get(name)
    }

    /// Get context metadata (for system prompt — symbolic handle, not the actual data)
    pub fn context_metadata(&self) -> String {
        match self.variables.get("context") {
            Some(Variable::Text(s)) => {
                let len = s.len();
                let lines = s.lines().count();
                let preview = &s[..s.len().min(200)];
                format!("Context: text, {len} chars, {lines} lines\nPreview: {preview}...")
            }
            Some(Variable::List(v)) => {
                let count = v.len();
                let total_chars: usize = v.iter().map(|s| s.len()).sum();
                let chunk_sizes: Vec<usize> = v.iter().map(|s| s.len()).collect();
                let preview = if count > 10 {
                    format!("{:?}... [{} others]", &chunk_sizes[..10], count - 10)
                } else {
                    format!("{chunk_sizes:?}")
                };
                format!(
                    "Context: list of {count} items, {total_chars} total chars\nChunk sizes: {preview}"
                )
            }
            Some(Variable::Number(n)) => format!("Context: number = {n}"),
            None => "Context: not loaded".into(),
        }
    }

    /// Execute a DSL command that doesn't require LLM calls
    pub fn execute(&mut self, command: &DslCommand) -> SoulResult<ExecResult> {
        match command {
            DslCommand::ChunkByLines {
                target,
                source,
                lines_per_chunk,
            } => {
                let text = self.require_text(source)?;
                let lines: Vec<&str> = text.lines().collect();
                let chunks: Vec<String> = lines
                    .chunks(*lines_per_chunk)
                    .map(|chunk| chunk.join("\n"))
                    .collect();
                let count = chunks.len();
                self.variables
                    .insert(target.clone(), Variable::List(chunks));
                Ok(ExecResult::Output(format!(
                    "Chunked into {count} parts by {lines_per_chunk} lines each"
                )))
            }
            DslCommand::ChunkByChars {
                target,
                source,
                chars_per_chunk,
            } => {
                let text = self.require_text(source)?;
                let mut chunks = Vec::new();
                let mut start = 0;
                while start < text.len() {
                    let end = (start + chars_per_chunk).min(text.len());
                    // Try to break at a newline boundary
                    let actual_end = if end < text.len() {
                        text[start..end]
                            .rfind('\n')
                            .map(|pos| start + pos + 1)
                            .unwrap_or(end)
                    } else {
                        end
                    };
                    chunks.push(text[start..actual_end].to_string());
                    start = actual_end;
                }
                let count = chunks.len();
                self.variables
                    .insert(target.clone(), Variable::List(chunks));
                Ok(ExecResult::Output(format!(
                    "Chunked into {count} parts by ~{chars_per_chunk} chars"
                )))
            }
            DslCommand::ChunkByRegex {
                target,
                source,
                pattern,
            } => {
                let text = self.require_text(source)?;
                let re = Regex::new(pattern.as_str()).map_err(|e| {
                    SoulError::Other(anyhow::anyhow!("Invalid regex pattern \"{pattern}\": {e}"))
                })?;
                // Split on regex — each non-empty segment becomes a chunk
                let chunks: Vec<String> = re
                    .split(&text)
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.to_string())
                    .collect();
                let count = chunks.len();
                self.variables
                    .insert(target.clone(), Variable::List(chunks));
                Ok(ExecResult::Output(format!(
                    "Split into {count} sections by regex \"{pattern}\""
                )))
            }
            DslCommand::Slice {
                target,
                source,
                start,
                end,
            } => {
                let text = self.require_text(source)?;
                let actual_start = (*start).min(text.len());
                let actual_end = (*end).min(text.len());
                let slice = text[actual_start..actual_end].to_string();
                let len = slice.len();
                self.variables.insert(target.clone(), Variable::Text(slice));
                Ok(ExecResult::Output(format!(
                    "Sliced [{start}..{end}], {len} chars"
                )))
            }
            DslCommand::Len { target, source } => {
                let var = self.require_var(source)?;
                let len = match var {
                    Variable::Text(s) => s.len(),
                    Variable::List(v) => v.len(),
                    Variable::Number(n) => *n,
                };
                self.variables.insert(target.clone(), Variable::Number(len));
                Ok(ExecResult::Output(format!("{source} length = {len}")))
            }
            DslCommand::Join {
                target,
                source,
                separator,
            } => {
                let list = self.require_list(source)?;
                let sep = separator.replace("\\n", "\n").replace("\\t", "\t");
                let joined = list.join(&sep);
                let len = joined.len();
                self.variables
                    .insert(target.clone(), Variable::Text(joined));
                Ok(ExecResult::Output(format!(
                    "Joined {source} into {len} chars"
                )))
            }
            DslCommand::Get { target, source } => {
                let var = self.require_var(source)?.clone();
                self.variables.insert(target.clone(), var);
                Ok(ExecResult::Silent)
            }
            DslCommand::Concat {
                target,
                left,
                right,
            } => {
                let l = self.require_text(left)?;
                let r = self.require_text(right)?;
                let combined = format!("{l}{r}");
                let len = combined.len();
                self.variables
                    .insert(target.clone(), Variable::Text(combined));
                Ok(ExecResult::Output(format!("Concatenated into {len} chars")))
            }
            DslCommand::Index {
                target,
                source,
                index,
            } => {
                let list = self.require_list(source)?;
                if *index >= list.len() {
                    return Err(SoulError::Other(anyhow::anyhow!(
                        "Index {index} out of bounds for {source} (len={})",
                        list.len()
                    )));
                }
                let item = list[*index].clone();
                self.variables.insert(target.clone(), Variable::Text(item));
                Ok(ExecResult::Output(format!(
                    "Got item [{index}] from {source}"
                )))
            }
            DslCommand::Print { var_name } => {
                let var = self.require_var(var_name)?;
                let output = match var {
                    Variable::Text(s) => {
                        if s.len() > 500 {
                            format!("{} ... [{} chars total]", &s[..500], s.len())
                        } else {
                            s.clone()
                        }
                    }
                    Variable::List(v) => {
                        format!(
                            "List[{}]: {:?}",
                            v.len(),
                            v.iter()
                                .map(|s| {
                                    if s.len() > 100 {
                                        format!("{}...", &s[..100])
                                    } else {
                                        s.clone()
                                    }
                                })
                                .collect::<Vec<_>>()
                        )
                    }
                    Variable::Number(n) => n.to_string(),
                };
                self.output_log.push(output.clone());
                Ok(ExecResult::Output(output))
            }
            DslCommand::ShowVars => {
                let vars: Vec<String> = self
                    .variables
                    .iter()
                    .map(|(k, v)| format!("  {k}: {}", v.type_name()))
                    .collect();
                let output = if vars.is_empty() {
                    "No variables.".to_string()
                } else {
                    format!("Variables:\n{}", vars.join("\n"))
                };
                Ok(ExecResult::VarList(output))
            }
            DslCommand::Final { var_name } => {
                let var = self.require_var(var_name)?;
                Ok(ExecResult::FinalAnswer(var.as_text()))
            }
            DslCommand::FinalText { text } => Ok(ExecResult::FinalAnswer(text.clone())),
            // LLM-requiring commands return requests
            DslCommand::Query {
                target,
                prompt,
                context_var,
            } => {
                let ctx = self.require_text(context_var)?;
                Ok(ExecResult::QueryRequest {
                    target: target.clone(),
                    prompt: prompt.clone(),
                    context: ctx,
                })
            }
            DslCommand::QueryBatch {
                target,
                prompts,
                context_vars,
            } => {
                let contexts: Vec<String> = context_vars
                    .iter()
                    .map(|v| self.require_text(v).unwrap_or_default())
                    .collect();
                // If single prompt + single context_var that's a list, expand
                if prompts.len() == 1 && context_vars.len() == 1 {
                    if let Some(Variable::List(items)) = self.variables.get(&context_vars[0]) {
                        let expanded_prompts: Vec<String> = items
                            .iter()
                            .map(|item| format!("{}\n\n{item}", prompts[0]))
                            .collect();
                        return Ok(ExecResult::QueryBatchRequest {
                            target: target.clone(),
                            prompts: expanded_prompts,
                            contexts: items.clone(),
                        });
                    }
                }
                Ok(ExecResult::QueryBatchRequest {
                    target: target.clone(),
                    prompts: prompts.clone(),
                    contexts,
                })
            }
            DslCommand::Map {
                target,
                source,
                prompt_template,
            } => {
                let items = self.require_list(source)?;
                Ok(ExecResult::MapRequest {
                    target: target.clone(),
                    items,
                    prompt_template: prompt_template.clone(),
                })
            }
            DslCommand::Filter {
                target,
                source,
                condition,
            } => {
                let items = self.require_list(source)?;
                Ok(ExecResult::FilterRequest {
                    target: target.clone(),
                    items,
                    condition: condition.clone(),
                })
            }
        }
    }

    /// List all variable names
    pub fn var_names(&self) -> Vec<String> {
        self.variables.keys().cloned().collect()
    }

    /// Get output log
    pub fn output_log(&self) -> &[String] {
        &self.output_log
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn require_var(&self, name: &str) -> SoulResult<&Variable> {
        self.variables.get(name).ok_or_else(|| {
            let available: Vec<_> = self.variables.keys().collect();
            SoulError::Other(anyhow::anyhow!(
                "Variable '{name}' not found. Available: {available:?}"
            ))
        })
    }

    fn require_text(&self, name: &str) -> SoulResult<String> {
        let var = self.require_var(name)?;
        Ok(var.as_text())
    }

    fn require_list(&self, name: &str) -> SoulResult<Vec<String>> {
        let var = self.require_var(name)?;
        Ok(var.as_list())
    }
}

impl Default for RlmEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_text_context() {
        let mut env = RlmEnvironment::new();
        env.load_context("Hello world, this is a test document.".into());
        let meta = env.context_metadata();
        assert!(meta.contains("text"));
        assert!(meta.contains("37 chars"));
    }

    #[test]
    fn load_list_context() {
        let mut env = RlmEnvironment::new();
        env.load_context_list(vec!["doc1".into(), "doc2".into(), "doc3".into()]);
        let meta = env.context_metadata();
        assert!(meta.contains("3 items"));
    }

    #[test]
    fn chunk_by_lines() {
        let mut env = RlmEnvironment::new();
        let text = (0..10)
            .map(|i| format!("Line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        env.load_context(text);

        let cmd = DslCommand::ChunkByLines {
            target: "chunks".into(),
            source: "context".into(),
            lines_per_chunk: 3,
        };
        let result = env.execute(&cmd).unwrap();
        assert!(matches!(result, ExecResult::Output(_)));

        let chunks = env.get_var("chunks").unwrap();
        assert_eq!(chunks.as_list().len(), 4); // 10 lines / 3 = 4 chunks (3,3,3,1)
    }

    #[test]
    fn chunk_by_chars() {
        let mut env = RlmEnvironment::new();
        env.load_context("a".repeat(1000));

        let cmd = DslCommand::ChunkByChars {
            target: "chunks".into(),
            source: "context".into(),
            chars_per_chunk: 300,
        };
        env.execute(&cmd).unwrap();

        let chunks = env.get_var("chunks").unwrap();
        assert!(chunks.as_list().len() >= 3);
    }

    #[test]
    fn slice_text() {
        let mut env = RlmEnvironment::new();
        env.load_context("Hello World!".into());

        let cmd = DslCommand::Slice {
            target: "part".into(),
            source: "context".into(),
            start: 0,
            end: 5,
        };
        env.execute(&cmd).unwrap();

        let part = env.get_var("part").unwrap();
        assert_eq!(part.as_text(), "Hello");
    }

    #[test]
    fn len_text() {
        let mut env = RlmEnvironment::new();
        env.load_context("Hello".into());

        let cmd = DslCommand::Len {
            target: "size".into(),
            source: "context".into(),
        };
        env.execute(&cmd).unwrap();

        let size = env.get_var("size").unwrap();
        assert!(matches!(size, Variable::Number(5)));
    }

    #[test]
    fn len_list() {
        let mut env = RlmEnvironment::new();
        env.set_var(
            "items",
            Variable::List(vec!["a".into(), "b".into(), "c".into()]),
        );

        let cmd = DslCommand::Len {
            target: "count".into(),
            source: "items".into(),
        };
        env.execute(&cmd).unwrap();

        assert!(matches!(env.get_var("count").unwrap(), Variable::Number(3)));
    }

    #[test]
    fn join_list() {
        let mut env = RlmEnvironment::new();
        env.set_var(
            "items",
            Variable::List(vec!["hello".into(), "world".into()]),
        );

        let cmd = DslCommand::Join {
            target: "text".into(),
            source: "items".into(),
            separator: " ".into(),
        };
        env.execute(&cmd).unwrap();

        assert_eq!(env.get_var("text").unwrap().as_text(), "hello world");
    }

    #[test]
    fn get_copies_var() {
        let mut env = RlmEnvironment::new();
        env.set_var("original", Variable::Text("data".into()));

        let cmd = DslCommand::Get {
            target: "copy".into(),
            source: "original".into(),
        };
        env.execute(&cmd).unwrap();

        assert_eq!(env.get_var("copy").unwrap().as_text(), "data");
    }

    #[test]
    fn concat_texts() {
        let mut env = RlmEnvironment::new();
        env.set_var("a", Variable::Text("hello ".into()));
        env.set_var("b", Variable::Text("world".into()));

        let cmd = DslCommand::Concat {
            target: "c".into(),
            left: "a".into(),
            right: "b".into(),
        };
        env.execute(&cmd).unwrap();

        assert_eq!(env.get_var("c").unwrap().as_text(), "hello world");
    }

    #[test]
    fn index_list() {
        let mut env = RlmEnvironment::new();
        env.set_var(
            "items",
            Variable::List(vec!["a".into(), "b".into(), "c".into()]),
        );

        let cmd = DslCommand::Index {
            target: "second".into(),
            source: "items".into(),
            index: 1,
        };
        env.execute(&cmd).unwrap();

        assert_eq!(env.get_var("second").unwrap().as_text(), "b");
    }

    #[test]
    fn index_out_of_bounds() {
        let mut env = RlmEnvironment::new();
        env.set_var("items", Variable::List(vec!["a".into()]));

        let cmd = DslCommand::Index {
            target: "x".into(),
            source: "items".into(),
            index: 5,
        };
        assert!(env.execute(&cmd).is_err());
    }

    #[test]
    fn print_truncates_long() {
        let mut env = RlmEnvironment::new();
        env.set_var("big", Variable::Text("x".repeat(1000)));

        let cmd = DslCommand::Print {
            var_name: "big".into(),
        };
        let result = env.execute(&cmd).unwrap();
        if let ExecResult::Output(s) = result {
            assert!(s.contains("chars total"));
            assert!(s.len() < 600);
        }
    }

    #[test]
    fn show_vars() {
        let mut env = RlmEnvironment::new();
        env.set_var("x", Variable::Number(42));
        env.set_var("y", Variable::Text("hello".into()));

        let cmd = DslCommand::ShowVars;
        let result = env.execute(&cmd).unwrap();
        if let ExecResult::VarList(s) = result {
            assert!(s.contains("x: number"));
            assert!(s.contains("y: text"));
        }
    }

    #[test]
    fn final_answer() {
        let mut env = RlmEnvironment::new();
        env.set_var("answer", Variable::Text("42".into()));

        let result = env
            .execute(&DslCommand::Final {
                var_name: "answer".into(),
            })
            .unwrap();
        assert!(matches!(result, ExecResult::FinalAnswer(s) if s == "42"));
    }

    #[test]
    fn final_text() {
        let mut env = RlmEnvironment::new();
        let result = env
            .execute(&DslCommand::FinalText {
                text: "direct answer".into(),
            })
            .unwrap();
        assert!(matches!(result, ExecResult::FinalAnswer(s) if s == "direct answer"));
    }

    #[test]
    fn query_returns_request() {
        let mut env = RlmEnvironment::new();
        env.load_context("document text".into());

        let cmd = DslCommand::Query {
            target: "result".into(),
            prompt: "summarize".into(),
            context_var: "context".into(),
        };
        let result = env.execute(&cmd).unwrap();
        assert!(matches!(result, ExecResult::QueryRequest { .. }));
    }

    #[test]
    fn map_returns_request() {
        let mut env = RlmEnvironment::new();
        env.set_var("chunks", Variable::List(vec!["a".into(), "b".into()]));

        let cmd = DslCommand::Map {
            target: "summaries".into(),
            source: "chunks".into(),
            prompt_template: "summarize: {item}".into(),
        };
        let result = env.execute(&cmd).unwrap();
        assert!(matches!(result, ExecResult::MapRequest { items, .. } if items.len() == 2));
    }

    #[test]
    fn missing_var_error() {
        let mut env = RlmEnvironment::new();
        let cmd = DslCommand::Print {
            var_name: "nonexistent".into(),
        };
        assert!(env.execute(&cmd).is_err());
    }
}
