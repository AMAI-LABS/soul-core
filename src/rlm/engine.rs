//! RLM Engine — orchestrates the recursive language model loop

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::provider::Provider;
use crate::types::*;

use super::dsl::DslParser;
use super::environment::{ExecResult, RlmEnvironment, Variable};

/// Configuration for an RLM session
#[derive(Debug, Clone)]
pub struct RlmConfig {
    pub model: ModelInfo,
    pub max_iterations: usize,
    pub max_depth: usize,
    pub system_prompt_override: Option<String>,
}

impl RlmConfig {
    pub fn new(model: ModelInfo) -> Self {
        Self {
            model,
            max_iterations: 30,
            max_depth: 1,
            system_prompt_override: None,
        }
    }
}

/// Result from a single RLM iteration
#[derive(Debug, Clone)]
pub struct RlmIteration {
    pub iteration: usize,
    pub llm_response: String,
    pub commands_executed: usize,
    pub outputs: Vec<String>,
    pub llm_queries_made: usize,
}

/// Final result from an RLM completion
#[derive(Debug, Clone)]
pub struct RlmResult {
    pub answer: String,
    pub iterations: Vec<RlmIteration>,
    pub total_llm_calls: usize,
    pub total_tokens: TokenUsage,
}

const RLM_SYSTEM_PROMPT: &str = r#"You are tasked with answering a query with associated context. You can access and transform this context using a mini DSL in ```rlm``` code blocks.

## Available Commands

### Data Operations
- `LET var = CHUNK source BY_LINES n` — Split text into chunks of n lines
- `LET var = CHUNK source BY_CHARS n` — Split text into chunks of ~n characters
- `LET var = CHUNK source BY_REGEX "pattern"` — Split text on regex pattern
- `LET var = SLICE source start end` — Get substring [start..end]
- `LET var = LEN source` — Get length (chars for text, items for list)
- `LET var = JOIN source "separator"` — Join list into text
- `LET var = CONCAT a b` — Concatenate two text variables
- `LET var = INDEX source n` — Get nth item from list
- `LET var = GET source` — Copy variable

### LLM Operations
- `LET var = QUERY "prompt" WITH context_var` — Query sub-LLM with context
- `LET var = QUERY_BATCH "prompt" WITH chunks_var` — Query sub-LLM per chunk
- `LET var = MAP source "prompt template with {item}"` — Map LLM over list
- `LET var = FILTER source "keep if condition"` — Filter list with LLM

### Control
- `PRINT var` — Display variable (truncated)
- `SHOW_VARS` — List all variables
- `FINAL var` — Return variable as final answer
- `FINAL_TEXT "inline answer"` — Return inline text as answer

## Strategy
1. First examine the context metadata to understand size and structure
2. CHUNK the context into manageable pieces
3. QUERY or MAP over chunks to extract/analyze
4. Aggregate results and provide final answer via FINAL

## Example
```rlm
LET size = LEN context
PRINT size
LET chunks = CHUNK context BY_LINES 50
LET summaries = MAP chunks "Summarize this section: {item}"
LET combined = JOIN summaries "\n\n"
LET answer = QUERY "Given these summaries, answer the original question" WITH combined
FINAL answer
```

Think step by step. Execute immediately — don't just plan. Output results to check your work."#;

/// The RLM engine
pub struct RlmEngine {
    provider: Arc<dyn Provider>,
    config: RlmConfig,
    auth: AuthProfile,
}

impl RlmEngine {
    pub fn new(
        provider: Arc<dyn Provider>,
        config: RlmConfig,
        auth: AuthProfile,
    ) -> Self {
        Self { provider, config, auth }
    }

    /// Run an RLM completion over a context
    pub async fn completion(
        &self,
        context: String,
        root_prompt: Option<&str>,
    ) -> SoulResult<RlmResult> {
        let mut env = RlmEnvironment::new();
        env.load_context(context);

        let system = self.config.system_prompt_override.as_deref()
            .unwrap_or(RLM_SYSTEM_PROMPT);

        let metadata = env.context_metadata();

        let mut messages: Vec<Message> = vec![
            Message::assistant(format!("Context loaded. {metadata}")),
        ];

        let user_prompt = if let Some(rp) = root_prompt {
            format!(
                "Answer this question using the RLM environment: {rp}\n\n\
                Use ```rlm``` code blocks to examine and process the context. Your next action:"
            )
        } else {
            "Examine the context and process it using ```rlm``` code blocks. Your next action:".into()
        };
        messages.push(Message::user(user_prompt));

        let mut iterations = Vec::new();
        let mut total_llm_calls = 0;
        let mut total_usage = TokenUsage::new(0, 0);

        for i in 0..self.config.max_iterations {
            // Call LLM
            let (tx, _rx) = mpsc::unbounded_channel();
            let response = self.provider.stream(
                &messages,
                system,
                &[],
                &self.config.model,
                &self.auth,
                tx,
            ).await?;

            if let Some(usage) = &response.usage {
                total_usage.input_tokens += usage.input_tokens;
                total_usage.output_tokens += usage.output_tokens;
            }
            total_llm_calls += 1;

            let response_text = response.text_content();

            // Extract and execute DSL blocks
            let blocks = DslParser::extract_blocks(&response_text);
            let mut iter_outputs = Vec::new();
            let mut commands_executed = 0;
            let mut sub_queries = 0;
            let mut final_answer: Option<String> = None;

            for block in &blocks {
                match DslParser::parse(block) {
                    Ok(commands) => {
                        for cmd in &commands {
                            commands_executed += 1;
                            match env.execute(cmd) {
                                Ok(ExecResult::Output(s)) => {
                                    iter_outputs.push(s);
                                }
                                Ok(ExecResult::Silent) => {}
                                Ok(ExecResult::VarList(s)) => {
                                    iter_outputs.push(s);
                                }
                                Ok(ExecResult::FinalAnswer(answer)) => {
                                    final_answer = Some(answer);
                                }
                                Ok(ExecResult::QueryRequest { target, prompt, context }) => {
                                    // Execute sub-LLM query
                                    let sub_prompt = format!("{prompt}\n\nContext:\n{context}");
                                    let sub_response = self.sub_query(&sub_prompt).await?;
                                    total_llm_calls += 1;
                                    sub_queries += 1;
                                    env.set_var(&target, Variable::Text(sub_response.clone()));
                                    iter_outputs.push(format!("[QUERY → {target}]: {} chars", sub_response.len()));
                                }
                                Ok(ExecResult::QueryBatchRequest { target, prompts, contexts }) => {
                                    let mut results = Vec::new();
                                    for (prompt, ctx) in prompts.iter().zip(contexts.iter()) {
                                        let sub_prompt = format!("{prompt}\n\nContext:\n{ctx}");
                                        let sub_response = self.sub_query(&sub_prompt).await?;
                                        total_llm_calls += 1;
                                        sub_queries += 1;
                                        results.push(sub_response);
                                    }
                                    let count = results.len();
                                    env.set_var(&target, Variable::List(results));
                                    iter_outputs.push(format!("[QUERY_BATCH → {target}]: {count} results"));
                                }
                                Ok(ExecResult::MapRequest { target, items, prompt_template }) => {
                                    let mut results = Vec::new();
                                    for item in &items {
                                        let prompt = prompt_template.replace("{item}", item);
                                        let sub_response = self.sub_query(&prompt).await?;
                                        total_llm_calls += 1;
                                        sub_queries += 1;
                                        results.push(sub_response);
                                    }
                                    let count = results.len();
                                    env.set_var(&target, Variable::List(results));
                                    iter_outputs.push(format!("[MAP → {target}]: {count} results"));
                                }
                                Ok(ExecResult::FilterRequest { target, items, condition }) => {
                                    let mut kept = Vec::new();
                                    for item in &items {
                                        let prompt = format!(
                                            "Does this item satisfy the condition \"{condition}\"? Answer YES or NO only.\n\nItem: {item}"
                                        );
                                        let sub_response = self.sub_query(&prompt).await?;
                                        total_llm_calls += 1;
                                        sub_queries += 1;
                                        if sub_response.trim().to_uppercase().starts_with("YES") {
                                            kept.push(item.clone());
                                        }
                                    }
                                    let count = kept.len();
                                    env.set_var(&target, Variable::List(kept));
                                    iter_outputs.push(format!("[FILTER → {target}]: {count} items kept"));
                                }
                                Err(e) => {
                                    iter_outputs.push(format!("Error: {e}"));
                                }
                            }
                        }
                    }
                    Err(e) => {
                        iter_outputs.push(format!("Parse error: {e}"));
                    }
                }
            }

            let iteration = RlmIteration {
                iteration: i,
                llm_response: response_text.clone(),
                commands_executed,
                outputs: iter_outputs.clone(),
                llm_queries_made: sub_queries,
            };
            iterations.push(iteration);

            // Check for final answer
            if let Some(answer) = final_answer {
                return Ok(RlmResult {
                    answer,
                    iterations,
                    total_llm_calls,
                    total_tokens: total_usage,
                });
            }

            // Build feedback for next iteration
            let feedback = if iter_outputs.is_empty() {
                if blocks.is_empty() {
                    "No ```rlm``` blocks found in your response. Use ```rlm``` blocks to interact with the environment.".to_string()
                } else {
                    "Commands executed with no output.".to_string()
                }
            } else {
                format!("REPL output:\n{}", iter_outputs.join("\n"))
            };

            messages.push(Message::assistant(response_text));
            messages.push(Message::user(format!(
                "{feedback}\n\nContinue using ```rlm``` blocks. When done, use FINAL <var> or FINAL_TEXT \"answer\"."
            )));
        }

        // Ran out of iterations — ask for final answer
        messages.push(Message::user(
            "Maximum iterations reached. Please provide your best answer using FINAL or FINAL_TEXT now.",
        ));

        let (tx, _) = mpsc::unbounded_channel();
        let response = self.provider.stream(
            &messages,
            system,
            &[],
            &self.config.model,
            &self.auth,
            tx,
        ).await?;
        total_llm_calls += 1;

        let answer = response.text_content();

        Ok(RlmResult {
            answer,
            iterations,
            total_llm_calls,
            total_tokens: total_usage,
        })
    }

    /// Execute a sub-LLM query (for QUERY, MAP, FILTER commands)
    async fn sub_query(&self, prompt: &str) -> SoulResult<String> {
        let messages = vec![Message::user(prompt.to_string())];
        let (tx, _) = mpsc::unbounded_channel();

        let response = self.provider.stream(
            &messages,
            "You are a helpful assistant. Answer concisely based on the provided context.",
            &[],
            &self.config.model,
            &self.auth,
            tx,
        ).await?;

        Ok(response.text_content())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rlm_config_defaults() {
        let model = ModelInfo {
            id: "test".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: false,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let config = RlmConfig::new(model);
        assert_eq!(config.max_iterations, 30);
        assert_eq!(config.max_depth, 1);
        assert!(config.system_prompt_override.is_none());
    }

    #[test]
    fn rlm_result_fields() {
        let result = RlmResult {
            answer: "42".into(),
            iterations: vec![],
            total_llm_calls: 5,
            total_tokens: TokenUsage::new(1000, 500),
        };
        assert_eq!(result.answer, "42");
        assert_eq!(result.total_llm_calls, 5);
        assert_eq!(result.total_tokens.total(), 1500);
    }

    #[test]
    fn rlm_iteration_fields() {
        let iter = RlmIteration {
            iteration: 0,
            llm_response: "Let me analyze this".into(),
            commands_executed: 3,
            outputs: vec!["chunked into 5 parts".into()],
            llm_queries_made: 1,
        };
        assert_eq!(iter.commands_executed, 3);
        assert_eq!(iter.llm_queries_made, 1);
    }

    #[test]
    fn system_prompt_contains_dsl_docs() {
        assert!(RLM_SYSTEM_PROMPT.contains("CHUNK"));
        assert!(RLM_SYSTEM_PROMPT.contains("QUERY"));
        assert!(RLM_SYSTEM_PROMPT.contains("FINAL"));
        assert!(RLM_SYSTEM_PROMPT.contains("MAP"));
        assert!(RLM_SYSTEM_PROMPT.contains("BY_LINES"));
    }
}
