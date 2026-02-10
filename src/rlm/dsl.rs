//! Mini DSL for LLMs to operate on documents.
//!
//! Commands the LLM can emit inside ```rlm``` code blocks:
//!
//! ```text
//! LET result = QUERY "summarize this" WITH chunk
//! LET chunks = CHUNK context BY_LINES 100
//! LET chunks = CHUNK context BY_CHARS 50000
//! LET chunks = CHUNK context BY_REGEX "^## "
//! LET result = QUERY_BATCH ["prompt1", "prompt2"] WITH [chunk1, chunk2]
//! LET subset = SLICE context 0 1000
//! LET count = LEN context
//! LET joined = JOIN results "\n"
//! LET text = GET var_name
//! PRINT var_name
//! SHOW_VARS
//! FINAL var_name
//! FINAL_TEXT "inline answer"
//! ```

use std::fmt;

/// A parsed DSL command
#[derive(Debug, Clone, PartialEq)]
pub enum DslCommand {
    /// LET var = QUERY "prompt" WITH context_var
    Query {
        target: String,
        prompt: String,
        context_var: String,
    },
    /// LET var = QUERY_BATCH [prompts...] WITH [vars...]
    QueryBatch {
        target: String,
        prompts: Vec<String>,
        context_vars: Vec<String>,
    },
    /// LET var = CHUNK source BY_LINES n
    ChunkByLines {
        target: String,
        source: String,
        lines_per_chunk: usize,
    },
    /// LET var = CHUNK source BY_CHARS n
    ChunkByChars {
        target: String,
        source: String,
        chars_per_chunk: usize,
    },
    /// LET var = CHUNK source BY_REGEX "pattern"
    ChunkByRegex {
        target: String,
        source: String,
        pattern: String,
    },
    /// LET var = SLICE source start end
    Slice {
        target: String,
        source: String,
        start: usize,
        end: usize,
    },
    /// LET var = LEN source
    Len {
        target: String,
        source: String,
    },
    /// LET var = JOIN source separator
    Join {
        target: String,
        source: String,
        separator: String,
    },
    /// LET var = GET var_name (alias / copy)
    Get {
        target: String,
        source: String,
    },
    /// LET var = CONCAT a b
    Concat {
        target: String,
        left: String,
        right: String,
    },
    /// LET var = INDEX source idx
    Index {
        target: String,
        source: String,
        index: usize,
    },
    /// LET var = MAP source "prompt template with {item}"
    Map {
        target: String,
        source: String,
        prompt_template: String,
    },
    /// LET var = FILTER source "condition prompt"
    Filter {
        target: String,
        source: String,
        condition: String,
    },
    /// PRINT var
    Print { var_name: String },
    /// SHOW_VARS
    ShowVars,
    /// FINAL var_name
    Final { var_name: String },
    /// FINAL_TEXT "inline text"
    FinalText { text: String },
}

#[derive(Debug, Clone)]
pub struct DslError {
    pub line: usize,
    pub message: String,
    pub source_line: String,
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Line {}: {} ({})", self.line, self.message, self.source_line)
    }
}

impl std::error::Error for DslError {}

pub struct DslParser;

impl DslParser {
    /// Parse a block of DSL code (extracted from ```rlm``` blocks)
    pub fn parse(input: &str) -> Result<Vec<DslCommand>, DslError> {
        let mut commands = Vec::new();
        for (line_num, line) in input.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
                continue;
            }
            match Self::parse_line(trimmed) {
                Ok(cmd) => commands.push(cmd),
                Err(msg) => {
                    return Err(DslError {
                        line: line_num + 1,
                        message: msg,
                        source_line: trimmed.to_string(),
                    });
                }
            }
        }
        Ok(commands)
    }

    /// Extract ```rlm``` code blocks from LLM response text
    pub fn extract_blocks(text: &str) -> Vec<String> {
        let mut blocks = Vec::new();
        let mut in_block = false;
        let mut current = String::new();

        for line in text.lines() {
            if line.trim().starts_with("```rlm") {
                in_block = true;
                current.clear();
            } else if in_block && line.trim() == "```" {
                in_block = false;
                if !current.trim().is_empty() {
                    blocks.push(current.trim().to_string());
                }
            } else if in_block {
                current.push_str(line);
                current.push('\n');
            }
        }
        blocks
    }

    fn parse_line(line: &str) -> Result<DslCommand, String> {
        let tokens: Vec<&str> = Self::tokenize(line);
        if tokens.is_empty() {
            return Err("Empty line".into());
        }

        match tokens[0].to_uppercase().as_str() {
            "LET" => Self::parse_let(&tokens),
            "PRINT" => {
                if tokens.len() < 2 {
                    return Err("PRINT requires a variable name".into());
                }
                Ok(DslCommand::Print {
                    var_name: tokens[1].to_string(),
                })
            }
            "SHOW_VARS" => Ok(DslCommand::ShowVars),
            "FINAL" => {
                if tokens.len() < 2 {
                    return Err("FINAL requires a variable name".into());
                }
                Ok(DslCommand::Final {
                    var_name: tokens[1].to_string(),
                })
            }
            "FINAL_TEXT" => {
                let text = Self::extract_quoted_from(line, "FINAL_TEXT")?;
                Ok(DslCommand::FinalText { text })
            }
            _ => Err(format!("Unknown command: {}", tokens[0])),
        }
    }

    fn parse_let(tokens: &[&str]) -> Result<DslCommand, String> {
        // LET target = OPERATION ...
        if tokens.len() < 4 || tokens[2] != "=" {
            return Err("LET syntax: LET <var> = <operation> ...".into());
        }
        let target = tokens[1].to_string();
        let operation = tokens[3].to_uppercase();

        match operation.as_str() {
            "QUERY" => {
                // LET target = QUERY "prompt" WITH var
                let full_line = tokens[4..].join(" ");
                let (prompt, rest) = Self::split_quoted_and_rest(&full_line)?;
                let rest_tokens: Vec<&str> = rest.split_whitespace().collect();
                if rest_tokens.is_empty() || rest_tokens[0].to_uppercase() != "WITH" {
                    return Err("QUERY syntax: QUERY \"prompt\" WITH <var>".into());
                }
                let context_var = rest_tokens.get(1).ok_or("QUERY missing context variable")?;
                Ok(DslCommand::Query {
                    target,
                    prompt,
                    context_var: context_var.to_string(),
                })
            }
            "QUERY_BATCH" => {
                // Simplified: LET target = QUERY_BATCH "prompt" WITH chunks_var
                let full_line = tokens[4..].join(" ");
                let (prompt, rest) = Self::split_quoted_and_rest(&full_line)?;
                let rest_tokens: Vec<&str> = rest.split_whitespace().collect();
                if rest_tokens.is_empty() || rest_tokens[0].to_uppercase() != "WITH" {
                    return Err("QUERY_BATCH syntax: QUERY_BATCH \"prompt\" WITH <chunks_var>".into());
                }
                let context_var = rest_tokens.get(1).ok_or("QUERY_BATCH missing context variable")?;
                Ok(DslCommand::QueryBatch {
                    target,
                    prompts: vec![prompt],
                    context_vars: vec![context_var.to_string()],
                })
            }
            "CHUNK" => {
                // LET target = CHUNK source BY_LINES|BY_CHARS|BY_REGEX ...
                if tokens.len() < 7 {
                    return Err("CHUNK syntax: CHUNK <source> BY_LINES|BY_CHARS|BY_REGEX <value>".into());
                }
                let source = tokens[4].to_string();
                let strategy = tokens[5].to_uppercase();
                match strategy.as_str() {
                    "BY_LINES" => {
                        let n: usize = tokens[6]
                            .parse()
                            .map_err(|_| "BY_LINES requires a number")?;
                        Ok(DslCommand::ChunkByLines {
                            target,
                            source,
                            lines_per_chunk: n,
                        })
                    }
                    "BY_CHARS" => {
                        let n: usize = tokens[6]
                            .parse()
                            .map_err(|_| "BY_CHARS requires a number")?;
                        Ok(DslCommand::ChunkByChars {
                            target,
                            source,
                            chars_per_chunk: n,
                        })
                    }
                    "BY_REGEX" => {
                        let pattern = Self::extract_quoted_at(tokens, 6)?;
                        Ok(DslCommand::ChunkByRegex {
                            target,
                            source,
                            pattern,
                        })
                    }
                    _ => Err(format!("Unknown chunk strategy: {strategy}")),
                }
            }
            "SLICE" => {
                // LET target = SLICE source start end
                if tokens.len() < 7 {
                    return Err("SLICE syntax: SLICE <source> <start> <end>".into());
                }
                let source = tokens[4].to_string();
                let start: usize = tokens[5].parse().map_err(|_| "SLICE start must be a number")?;
                let end: usize = tokens[6].parse().map_err(|_| "SLICE end must be a number")?;
                Ok(DslCommand::Slice { target, source, start, end })
            }
            "LEN" => {
                if tokens.len() < 5 {
                    return Err("LEN syntax: LEN <source>".into());
                }
                Ok(DslCommand::Len {
                    target,
                    source: tokens[4].to_string(),
                })
            }
            "JOIN" => {
                // LET target = JOIN source "separator"
                if tokens.len() < 6 {
                    return Err("JOIN syntax: JOIN <source> \"separator\"".into());
                }
                let source = tokens[4].to_string();
                let separator = Self::extract_quoted_at(tokens, 5)?;
                Ok(DslCommand::Join { target, source, separator })
            }
            "GET" => {
                if tokens.len() < 5 {
                    return Err("GET syntax: GET <source>".into());
                }
                Ok(DslCommand::Get {
                    target,
                    source: tokens[4].to_string(),
                })
            }
            "CONCAT" => {
                if tokens.len() < 6 {
                    return Err("CONCAT syntax: CONCAT <left> <right>".into());
                }
                Ok(DslCommand::Concat {
                    target,
                    left: tokens[4].to_string(),
                    right: tokens[5].to_string(),
                })
            }
            "INDEX" => {
                if tokens.len() < 6 {
                    return Err("INDEX syntax: INDEX <source> <index>".into());
                }
                let index: usize = tokens[5].parse().map_err(|_| "INDEX requires a number")?;
                Ok(DslCommand::Index {
                    target,
                    source: tokens[4].to_string(),
                    index,
                })
            }
            "MAP" => {
                // LET target = MAP source "prompt template"
                if tokens.len() < 6 {
                    return Err("MAP syntax: MAP <source> \"prompt template\"".into());
                }
                let source = tokens[4].to_string();
                let prompt_template = Self::extract_quoted_at(tokens, 5)?;
                Ok(DslCommand::Map { target, source, prompt_template })
            }
            "FILTER" => {
                if tokens.len() < 6 {
                    return Err("FILTER syntax: FILTER <source> \"condition\"".into());
                }
                let source = tokens[4].to_string();
                let condition = Self::extract_quoted_at(tokens, 5)?;
                Ok(DslCommand::Filter { target, source, condition })
            }
            _ => Err(format!("Unknown operation: {operation}")),
        }
    }

    /// Tokenize respecting quoted strings
    fn tokenize(line: &str) -> Vec<&str> {
        let mut tokens = Vec::new();
        let mut chars = line.char_indices().peekable();
        let mut token_start: Option<usize> = None;

        while let Some(&(i, c)) = chars.peek() {
            if c == '"' {
                // Quoted string — find matching close
                let start = i;
                chars.next(); // consume opening quote
                while let Some(&(_, ch)) = chars.peek() {
                    chars.next();
                    if ch == '"' {
                        break;
                    }
                }
                let end = chars.peek().map(|&(i, _)| i).unwrap_or(line.len());
                tokens.push(&line[start..end]);
                token_start = None;
            } else if c.is_whitespace() {
                if let Some(start) = token_start {
                    tokens.push(&line[start..i]);
                    token_start = None;
                }
                chars.next();
            } else {
                if token_start.is_none() {
                    token_start = Some(i);
                }
                chars.next();
            }
        }
        if let Some(start) = token_start {
            tokens.push(&line[start..]);
        }
        tokens
    }

    fn extract_quoted_at(tokens: &[&str], index: usize) -> Result<String, String> {
        let token = tokens.get(index).ok_or("Missing quoted string")?;
        if token.starts_with('"') && token.ends_with('"') && token.len() >= 2 {
            Ok(token[1..token.len() - 1].to_string())
        } else {
            // Rejoin remaining tokens and try to extract
            let rest = tokens[index..].join(" ");
            if rest.starts_with('"') {
                if let Some(end) = rest[1..].find('"') {
                    return Ok(rest[1..end + 1].to_string());
                }
            }
            Err(format!("Expected quoted string at position {index}, got: {token}"))
        }
    }

    fn extract_quoted_from(line: &str, prefix: &str) -> Result<String, String> {
        let rest = line.strip_prefix(prefix).unwrap_or(line).trim();
        if rest.starts_with('"') {
            if let Some(end) = rest[1..].find('"') {
                return Ok(rest[1..end + 1].to_string());
            }
        }
        Err("Expected quoted string".into())
    }

    fn split_quoted_and_rest(input: &str) -> Result<(String, String), String> {
        let trimmed = input.trim();
        if !trimmed.starts_with('"') {
            return Err("Expected quoted string".into());
        }
        if let Some(end) = trimmed[1..].find('"') {
            let quoted = trimmed[1..end + 1].to_string();
            let rest = trimmed[end + 2..].trim().to_string();
            Ok((quoted, rest))
        } else {
            Err("Unterminated quoted string".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Parser Tests ───────────────────────────────────────────────────

    #[test]
    fn parse_query() {
        let cmds = DslParser::parse(r#"LET result = QUERY "summarize this document" WITH context"#).unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(&cmds[0], DslCommand::Query { target, prompt, context_var }
            if target == "result" && prompt == "summarize this document" && context_var == "context"));
    }

    #[test]
    fn parse_chunk_by_lines() {
        let cmds = DslParser::parse("LET chunks = CHUNK context BY_LINES 100").unwrap();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(&cmds[0], DslCommand::ChunkByLines { target, source, lines_per_chunk }
            if target == "chunks" && source == "context" && *lines_per_chunk == 100));
    }

    #[test]
    fn parse_chunk_by_chars() {
        let cmds = DslParser::parse("LET chunks = CHUNK context BY_CHARS 50000").unwrap();
        assert!(matches!(&cmds[0], DslCommand::ChunkByChars { chars_per_chunk, .. } if *chars_per_chunk == 50000));
    }

    #[test]
    fn parse_chunk_by_regex() {
        let cmds = DslParser::parse(r#"LET sections = CHUNK context BY_REGEX "^## ""#).unwrap();
        assert!(matches!(&cmds[0], DslCommand::ChunkByRegex { pattern, .. } if pattern == "^## "));
    }

    #[test]
    fn parse_slice() {
        let cmds = DslParser::parse("LET preview = SLICE context 0 1000").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Slice { start, end, .. } if *start == 0 && *end == 1000));
    }

    #[test]
    fn parse_len() {
        let cmds = DslParser::parse("LET size = LEN context").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Len { target, source } if target == "size" && source == "context"));
    }

    #[test]
    fn parse_join() {
        let cmds = DslParser::parse(r#"LET combined = JOIN results "\n""#).unwrap();
        assert!(matches!(&cmds[0], DslCommand::Join { separator, .. } if separator == "\\n"));
    }

    #[test]
    fn parse_get() {
        let cmds = DslParser::parse("LET copy = GET original").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Get { target, source } if target == "copy" && source == "original"));
    }

    #[test]
    fn parse_concat() {
        let cmds = DslParser::parse("LET merged = CONCAT part1 part2").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Concat { left, right, .. } if left == "part1" && right == "part2"));
    }

    #[test]
    fn parse_index() {
        let cmds = DslParser::parse("LET first = INDEX chunks 0").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Index { index, .. } if *index == 0));
    }

    #[test]
    fn parse_map() {
        let cmds = DslParser::parse(r#"LET summaries = MAP chunks "summarize: {item}""#).unwrap();
        assert!(matches!(&cmds[0], DslCommand::Map { prompt_template, .. } if prompt_template == "summarize: {item}"));
    }

    #[test]
    fn parse_filter() {
        let cmds = DslParser::parse(r#"LET relevant = FILTER chunks "is this about rust?""#).unwrap();
        assert!(matches!(&cmds[0], DslCommand::Filter { condition, .. } if condition == "is this about rust?"));
    }

    #[test]
    fn parse_print() {
        let cmds = DslParser::parse("PRINT result").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Print { var_name } if var_name == "result"));
    }

    #[test]
    fn parse_show_vars() {
        let cmds = DslParser::parse("SHOW_VARS").unwrap();
        assert!(matches!(&cmds[0], DslCommand::ShowVars));
    }

    #[test]
    fn parse_final() {
        let cmds = DslParser::parse("FINAL answer").unwrap();
        assert!(matches!(&cmds[0], DslCommand::Final { var_name } if var_name == "answer"));
    }

    #[test]
    fn parse_final_text() {
        let cmds = DslParser::parse(r#"FINAL_TEXT "The answer is 42""#).unwrap();
        assert!(matches!(&cmds[0], DslCommand::FinalText { text } if text == "The answer is 42"));
    }

    #[test]
    fn parse_comments_skipped() {
        let input = "# This is a comment\n// Also a comment\nLET x = LEN context";
        let cmds = DslParser::parse(input).unwrap();
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn parse_empty_lines_skipped() {
        let input = "\n\n  \n  LET x = LEN context\n\n";
        let cmds = DslParser::parse(input).unwrap();
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn parse_multi_line_program() {
        let program = r#"
LET chunks = CHUNK context BY_LINES 50
LET size = LEN chunks
PRINT size
LET result = QUERY "summarize all" WITH context
FINAL result
"#;
        let cmds = DslParser::parse(program).unwrap();
        assert_eq!(cmds.len(), 5);
    }

    #[test]
    fn parse_error_unknown_command() {
        let result = DslParser::parse("FOOBAR stuff");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("Unknown command"));
    }

    #[test]
    fn parse_error_bad_let() {
        let result = DslParser::parse("LET x QUERY stuff");
        assert!(result.is_err());
    }

    // ─── Block Extraction Tests ─────────────────────────────────────────

    #[test]
    fn extract_single_block() {
        let text = "Let me analyze this.\n```rlm\nLET x = LEN context\nPRINT x\n```\nDone.";
        let blocks = DslParser::extract_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("LET x"));
    }

    #[test]
    fn extract_multiple_blocks() {
        let text = "First:\n```rlm\nLET a = LEN context\n```\nThen:\n```rlm\nFINAL a\n```";
        let blocks = DslParser::extract_blocks(text);
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn extract_no_blocks() {
        let text = "Just regular text without any code blocks.";
        let blocks = DslParser::extract_blocks(text);
        assert!(blocks.is_empty());
    }

    #[test]
    fn extract_ignores_non_rlm_blocks() {
        let text = "```python\nprint('hello')\n```\n```rlm\nLET x = LEN context\n```";
        let blocks = DslParser::extract_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("LET x"));
    }

    // ─── Tokenizer Tests ────────────────────────────────────────────────

    #[test]
    fn tokenize_simple() {
        let tokens = DslParser::tokenize("LET x = LEN context");
        assert_eq!(tokens, vec!["LET", "x", "=", "LEN", "context"]);
    }

    #[test]
    fn tokenize_with_quotes() {
        let tokens = DslParser::tokenize(r#"LET r = QUERY "hello world" WITH ctx"#);
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[4], "\"hello world\"");
    }
}
