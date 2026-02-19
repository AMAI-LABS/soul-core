//! RLM Engine — orchestrates the context-as-document infinite memory loop
//!
//! The engine treats the agent's conversation history as an external document.
//! Instead of truncating context when it grows too large, the full history is
//! serialized and made available through a DSL REPL that the LLM navigates.
//!
//! Supports two code block formats:
//! - ` ```lua ``` ` — Lua 5.4 scripts executed via [`LuaSandbox`](crate::lua::LuaSandbox)
//! - ` ```rlm ``` ` — Legacy DSL commands (backward compatibility)

use std::sync::Arc;
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::lua::{LlmQueryChannel, LlmSubRequest, LuaSandbox};
use crate::lua::functions::register_rlm_functions;
use crate::provider::Provider;
use crate::types::*;

use super::dsl::DslParser;
use super::environment::{ExecResult, RlmEnvironment, Variable};

/// Configuration for an RLM session
#[derive(Debug, Clone)]
pub struct RlmConfig {
    pub model: ModelInfo,
    /// Optional cheaper model for sub-queries (QUERY, MAP, FILTER, llm_query).
    /// Falls back to `model` when None.
    pub sub_model: Option<ModelInfo>,
    pub max_iterations: usize,
    pub max_depth: usize,
    /// Maximum total sub-LLM calls across all iterations. 0 = unlimited.
    pub max_sub_calls: usize,
    /// Maximum characters in REPL output feedback per iteration.
    /// Prevents context overflow from verbose print() output.
    /// 0 = unlimited. Default: 8192 (matches RLM paper).
    pub max_output_chars: usize,
    pub system_prompt_override: Option<String>,
}

impl RlmConfig {
    pub fn new(model: ModelInfo) -> Self {
        Self {
            model,
            sub_model: None,
            max_iterations: 30,
            max_depth: 1,
            max_sub_calls: 0,
            max_output_chars: 8192,
            system_prompt_override: None,
        }
    }

    /// Set a cheaper model for sub-queries (llm_query, QUERY, MAP, FILTER).
    pub fn with_sub_model(mut self, sub_model: ModelInfo) -> Self {
        self.sub_model = Some(sub_model);
        self
    }

    /// Set maximum sub-LLM calls budget. 0 = unlimited.
    pub fn with_max_sub_calls(mut self, max: usize) -> Self {
        self.max_sub_calls = max;
        self
    }

    /// Set maximum output characters per iteration. 0 = unlimited.
    pub fn with_max_output_chars(mut self, max: usize) -> Self {
        self.max_output_chars = max;
        self
    }

    /// Get the model to use for sub-queries (sub_model if set, otherwise main model).
    pub fn sub_query_model(&self) -> &ModelInfo {
        self.sub_model.as_ref().unwrap_or(&self.model)
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

const RLM_SYSTEM_PROMPT: &str = r####"You have access to your complete conversation history as an external document stored in the `context` variable. This document contains every message, tool call, tool result, and decision you have ever made in this session. It grows with every turn — nothing is ever deleted.

Your current context window only shows recent messages. To access older history, write Lua code in ```lua``` blocks to explore and process the context.

**IMPORTANT: On your FIRST interaction, you MUST explore the context before answering. Do NOT call final_answer() until you have actually examined the data. Write code to understand the context structure first.**

## Available Variables

- `context` — full conversation history as text

## Available Functions

### Text Processing
- `chunk_by_lines(text, n)` — split text into n-line chunks, returns table
- `chunk_by_chars(text, n)` — split into ~n-char chunks (good default: 3000-5000)
- `chunk_by_regex(text, pattern)` — split on regex boundary
- `slice(text, start, len)` — substring (0-indexed start)

### Search (BM25 keyword retrieval — zero LLM cost)
- `search(text, query, top_k?)` — find most relevant sections by keyword, returns table of `{text, score, line}`
  - Splits text at blank lines and markdown headers into sections
  - Scores each section by TF-IDF (term frequency × inverse document frequency)
  - Returns top_k results (default 5) sorted by relevance score
  - **Use this FIRST** to find relevant sections before chunking or using llm_query

### Sub-LLM Calls (for semantic analysis of chunks)
- `llm_query(prompt)` — ask a sub-LLM a question, returns string response
- `llm_query_batched(prompts_table)` — ask multiple questions concurrently, returns table of responses

### Output & Termination
- `print_var(x)` / `print(...)` — display values (visible as REPL output)
- `final_answer(x)` — return x as the answer (ends REPL)
- `final_var("varname")` — return a REPL variable as the answer (use for long outputs that would exceed output token limits)

### Standard Lua
- Full `string.*` library (find, sub, gsub, match, format, upper, lower, etc.)
- Full `table.*` library (insert, remove, sort, concat, etc.)
- Full `math.*` library
- `tostring()`, `tonumber()`, `type()`, `pairs()`, `ipairs()`, `select()`, `pcall()`, `xpcall()`

## Context Document Structure

The `context` variable contains your full history in this format:

```
## TURN 1
### USER
<user message>
### ASSISTANT
<your response>
### TOOL_CALL: tool_name (call_id)
Arguments: {...}
### TOOL_RESULT (call_id) [OK]
<tool output>

## TURN 2
...
```

## Strategies

### 1. Peek First — understand structure before processing
```lua
print("Length: " .. #context .. " chars")
print("First 500 chars:")
print(slice(context, 0, 500))
```

### 2. Search — find relevant sections by keyword (fast, no LLM cost)
```lua
local results = search(context, "error test failure", 3)
for i, r in ipairs(results) do
  print("Match " .. i .. " (score=" .. r.score .. ", line=" .. r.line .. "):")
  print(r.text)
end
```

### 3. Grep — filter with string patterns (fast, no LLM cost)
```lua
for line in context:gmatch("[^\n]+") do
  if line:find("TOOL_CALL: read") then
    print(line)
  end
end
```

### 4. Partition + Map + Aggregate — chunk, analyze each, combine
```lua
local turns = chunk_by_regex(context, "## TURN %d+")
local prompts = {}
for i, turn in ipairs(turns) do
  prompts[i] = "Summarize what was accomplished in this turn:\n\n" .. turn
end
local summaries = llm_query_batched(prompts)
local report = ""
for i, s in ipairs(summaries) do
  report = report .. "Turn " .. i .. ": " .. s .. "\n"
end
final_var("report")
```

### 5. Targeted Search + Sub-Query
```lua
-- Find a specific tool result and analyze it
local chunks = chunk_by_regex(context, "### TOOL_RESULT")
for i, chunk in ipairs(chunks) do
  if chunk:find("error") or chunk:find("ERROR") then
    local analysis = llm_query("What went wrong here?\n\n" .. chunk)
    print("Error in chunk " .. i .. ": " .. analysis)
  end
end
```

### 6. Building Long Outputs via Variables
```lua
-- Build a large result across iterations without output token limits
result = ""
local turns = chunk_by_regex(context, "## TURN %d+")
for i, turn in ipairs(turns) do
  if turn:find("TOOL_CALL") then
    result = result .. "Turn " .. i .. " used tools\n"
  end
end
final_var("result")  -- returns the variable, not a string literal
```

Think step by step. Execute immediately — don't just plan."####;

/// The RLM engine
pub struct RlmEngine {
    provider: Arc<dyn Provider>,
    config: RlmConfig,
    auth: AuthProfile,
}

impl RlmEngine {
    pub fn new(provider: Arc<dyn Provider>, config: RlmConfig, auth: AuthProfile) -> Self {
        Self {
            provider,
            config,
            auth,
        }
    }

    /// Run an RLM completion over a context.
    ///
    /// Supports two code block formats:
    /// - ` ```lua ``` ` — Lua 5.4 scripts (preferred, every LLM knows Lua)
    /// - ` ```rlm ``` ` — Legacy DSL commands (backward compatibility)
    pub async fn completion(
        &self,
        context: String,
        root_prompt: Option<&str>,
    ) -> SoulResult<RlmResult> {
        let mut env = RlmEnvironment::new();
        env.load_context(context.clone());

        let system = self
            .config
            .system_prompt_override
            .as_deref()
            .unwrap_or(RLM_SYSTEM_PROMPT);

        let metadata = env.context_metadata();

        let mut messages: Vec<Message> =
            vec![Message::assistant(format!("Context loaded. {metadata}"))];

        let user_prompt = if let Some(rp) = root_prompt {
            format!(
                "Answer this question using the RLM environment: {rp}\n\n\
                Use ```lua``` code blocks to examine and process the context. Your next action:"
            )
        } else {
            "Examine the context and process it using ```lua``` code blocks. Your next action:"
                .into()
        };
        messages.push(Message::user(user_prompt));

        // Create Lua sandbox for the session — persists across iterations
        let lua_sandbox = LuaSandbox::new()?;
        lua_sandbox.set_string("context", &context)?;
        register_rlm_functions(lua_sandbox.lua())?;

        // Set up LLM sub-query channel: Lua blocks → requests → async fulfillment → responses
        let (request_tx, request_rx) = std::sync::mpsc::channel::<LlmSubRequest>();
        let (response_tx, response_rx) = std::sync::mpsc::channel::<String>();
        let request_rx = Arc::new(std::sync::Mutex::new(request_rx));
        let llm_channel = LlmQueryChannel {
            request_tx,
            response_rx: Arc::new(std::sync::Mutex::new(response_rx)),
        };
        lua_sandbox.register_llm_query(llm_channel)?;

        let mut iterations = Vec::new();
        let mut total_llm_calls = 0;
        // Shared counter for sub-query budget enforcement across iterations
        let total_sub_queries = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut total_usage = TokenUsage::new(0, 0);

        for i in 0..self.config.max_iterations {
            // Call LLM
            let (tx, _rx) = mpsc::unbounded_channel();
            let response = self
                .provider
                .stream(&messages, system, &[], &self.config.model, &self.auth, tx)
                .await?;

            if let Some(usage) = &response.usage {
                total_usage.input_tokens += usage.input_tokens;
                total_usage.output_tokens += usage.output_tokens;
            }
            total_llm_calls += 1;

            let response_text = response.text_content();

            // Extract Lua blocks (preferred)
            let lua_blocks = crate::lua::extract_lua_blocks(&response_text);
            // Extract legacy DSL blocks (backward compat)
            let dsl_blocks = DslParser::extract_blocks(&response_text);

            let mut iter_outputs = Vec::new();
            let mut commands_executed = 0;
            let mut sub_queries = 0;
            let mut final_answer: Option<String> = None;

            // Execute Lua blocks — run in spawn_blocking so llm_query() can block
            // while we fulfill sub-query requests on the async side.
            if !lua_blocks.is_empty() {
                for block in &lua_blocks {
                    commands_executed += 1;
                    let block_code = block.clone();

                    // Spawn an async fulfiller task that processes llm_query() requests
                    // from Lua while the Lua block executes on the current thread.
                    // LuaSandbox is !Send so we can't use spawn_blocking — instead we
                    // run Lua synchronously and the fulfiller runs as a tokio task.
                    let provider_c = self.provider.clone();
                    let model_c = self.config.sub_query_model().clone();
                    let auth_c = self.auth.clone();
                    let resp_tx_c = response_tx.clone();
                    let req_rx_c = request_rx.clone();
                    let max_sub = self.config.max_sub_calls;
                    let sub_counter = total_sub_queries.clone();

                    // Create a oneshot to know when Lua execution is done
                    let (lua_done_tx, lua_done_rx) = tokio::sync::oneshot::channel::<()>();

                    // Spawn async fulfiller task that processes sub-query requests
                    let fulfiller_handle = tokio::spawn(async move {
                        let mut sub_count = 0usize;
                        loop {
                            // Check if Lua is done
                            if lua_done_rx.is_terminated() {
                                break;
                            }

                            // Budget check: if max_sub_calls is set, enforce it
                            if max_sub > 0 {
                                let total = sub_counter.load(std::sync::atomic::Ordering::Relaxed);
                                if total >= max_sub {
                                    // Drain remaining requests with budget error
                                    if let Ok(_) = req_rx_c.lock().unwrap().try_recv() {
                                        let _ = resp_tx_c.send(
                                            format!("ERROR: Sub-query budget exhausted ({max_sub} calls max)")
                                        );
                                    }
                                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                                    continue;
                                }
                            }

                            // Try to receive a request (non-blocking with small sleep)
                            let request = req_rx_c.lock().unwrap().try_recv();
                            match request {
                                Ok(LlmSubRequest::Single(prompt)) => {
                                    let result = Self::static_sub_query(
                                        &provider_c, &model_c, &auth_c, &prompt,
                                    ).await;
                                    sub_count += 1;
                                    sub_counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    let response_text = result.unwrap_or_else(|e| format!("ERROR: {e}"));
                                    let _ = resp_tx_c.send(response_text);
                                }
                                Ok(LlmSubRequest::Batched(prompts)) => {
                                    // Execute all prompts concurrently
                                    let futs: Vec<_> = prompts.iter().map(|p| {
                                        Self::static_sub_query(&provider_c, &model_c, &auth_c, p)
                                    }).collect();
                                    let results = futures::future::join_all(futs).await;
                                    sub_count += results.len();
                                    sub_counter.fetch_add(results.len(), std::sync::atomic::Ordering::Relaxed);
                                    for result in results {
                                        let text = result.unwrap_or_else(|e| format!("ERROR: {e}"));
                                        let _ = resp_tx_c.send(text);
                                    }
                                }
                                Err(std::sync::mpsc::TryRecvError::Empty) => {
                                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                                }
                                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                                    break;
                                }
                            }
                        }
                        sub_count
                    });

                    // Execute Lua on the current thread (LuaSandbox is !Send)
                    let exec_result = lua_sandbox.exec(&block_code);
                    let _ = lua_done_tx.send(());

                    // Wait for fulfiller to finish processing any pending requests
                    let lua_sub_queries = fulfiller_handle.await.unwrap_or(0);
                    sub_queries += lua_sub_queries;

                    match exec_result {
                        Ok(_) => {
                            let prints = lua_sandbox.take_output();
                            iter_outputs.extend(prints);
                            if let Some(answer) = lua_sandbox.take_final_answer() {
                                final_answer = Some(answer);
                            }
                        }
                        Err(e) => {
                            iter_outputs
                                .push(format!("-- LUA ERROR: {e}\n-- Fix the code and retry."));
                        }
                    }
                }
            }

            // Execute legacy DSL blocks (backward compat — only if no Lua blocks)
            if lua_blocks.is_empty() {
                for block in &dsl_blocks {
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
                                    Ok(ExecResult::QueryRequest {
                                        target,
                                        prompt,
                                        context,
                                    }) => {
                                        let sub_prompt = format!("{prompt}\n\nContext:\n{context}");
                                        let sub_response = self.sub_query(&sub_prompt).await?;
                                        total_llm_calls += 1;
                                        sub_queries += 1;
                                        env.set_var(&target, Variable::Text(sub_response.clone()));
                                        iter_outputs.push(format!(
                                            "[QUERY → {target}]: {} chars",
                                            sub_response.len()
                                        ));
                                    }
                                    Ok(ExecResult::QueryBatchRequest {
                                        target,
                                        prompts,
                                        contexts,
                                    }) => {
                                        // Concurrent sub-queries
                                        let futs: Vec<_> = prompts
                                            .iter()
                                            .zip(contexts.iter())
                                            .map(|(prompt, ctx)| {
                                                let sub_prompt = format!("{prompt}\n\nContext:\n{ctx}");
                                                self.sub_query_owned(sub_prompt)
                                            })
                                            .collect();
                                        let results = futures::future::join_all(futs).await;
                                        let mut texts = Vec::new();
                                        for result in results {
                                            let text = result?;
                                            total_llm_calls += 1;
                                            sub_queries += 1;
                                            texts.push(text);
                                        }
                                        let count = texts.len();
                                        env.set_var(&target, Variable::List(texts));
                                        iter_outputs.push(format!(
                                            "[QUERY_BATCH → {target}]: {count} results"
                                        ));
                                    }
                                    Ok(ExecResult::MapRequest {
                                        target,
                                        items,
                                        prompt_template,
                                    }) => {
                                        // Concurrent MAP
                                        let futs: Vec<_> = items
                                            .iter()
                                            .map(|item| {
                                                let prompt = prompt_template.replace("{item}", item);
                                                self.sub_query_owned(prompt)
                                            })
                                            .collect();
                                        let results = futures::future::join_all(futs).await;
                                        let mut texts = Vec::new();
                                        for result in results {
                                            let text = result?;
                                            total_llm_calls += 1;
                                            sub_queries += 1;
                                            texts.push(text);
                                        }
                                        let count = texts.len();
                                        env.set_var(&target, Variable::List(texts));
                                        iter_outputs
                                            .push(format!("[MAP → {target}]: {count} results"));
                                    }
                                    Ok(ExecResult::FilterRequest {
                                        target,
                                        items,
                                        condition,
                                    }) => {
                                        // Concurrent FILTER
                                        let futs: Vec<_> = items
                                            .iter()
                                            .map(|item| {
                                                let prompt = format!(
                                                    "Does this item satisfy the condition \"{condition}\"? Answer YES or NO only.\n\nItem: {item}"
                                                );
                                                self.sub_query_owned(prompt)
                                            })
                                            .collect();
                                        let results = futures::future::join_all(futs).await;
                                        let mut kept = Vec::new();
                                        for (result, item) in results.into_iter().zip(items.iter()) {
                                            let text = result?;
                                            total_llm_calls += 1;
                                            sub_queries += 1;
                                            if text.trim().to_uppercase().starts_with("YES") {
                                                kept.push(item.clone());
                                            }
                                        }
                                        let count = kept.len();
                                        env.set_var(&target, Variable::List(kept));
                                        iter_outputs.push(format!(
                                            "[FILTER → {target}]: {count} items kept"
                                        ));
                                    }
                                    Err(e) => {
                                        iter_outputs.push(format!(
                                            "// ERROR: {e}\n// Fix the command and retry."
                                        ));
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            iter_outputs.push(format!(
                                "// PARSE ERROR: {e}\n// Fix the ```rlm``` block syntax and retry."
                            ));
                        }
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

            // Build feedback for next iteration (with output cap)
            let feedback = if iter_outputs.is_empty() {
                if lua_blocks.is_empty() && dsl_blocks.is_empty() {
                    "No ```lua``` blocks found in your response. Use ```lua``` blocks to interact with the environment.".to_string()
                } else {
                    "Commands executed with no output.".to_string()
                }
            } else {
                let raw = iter_outputs.join("\n");
                let max = self.config.max_output_chars;
                if max > 0 && raw.len() > max {
                    format!(
                        "REPL output (TRUNCATED — {}/{max} chars, use `search()` or `slice()` for targeted access):\n{}",
                        raw.len(),
                        &raw[..max]
                    )
                } else {
                    format!("REPL output:\n{raw}")
                }
            };

            messages.push(Message::assistant(response_text));
            messages.push(Message::user(format!(
                "{feedback}\n\nContinue using ```lua``` blocks. When done, call `final_answer(\"your answer\")` or `final_var(\"varname\")`."
            )));
        }

        // Ran out of iterations — ask for final answer
        messages.push(Message::user(
            "Maximum iterations reached. Please provide your best answer using `final_answer(\"...\")` or `final_var(\"varname\")` in a ```lua``` block now.",
        ));

        let (tx, _) = mpsc::unbounded_channel();
        let response = self
            .provider
            .stream(&messages, system, &[], &self.config.model, &self.auth, tx)
            .await?;
        total_llm_calls += 1;

        let answer = response.text_content();

        Ok(RlmResult {
            answer,
            iterations,
            total_llm_calls,
            total_tokens: total_usage,
        })
    }

    /// Execute a sub-LLM query (for QUERY, MAP, FILTER commands).
    /// Uses sub_model if configured, otherwise falls back to main model.
    async fn sub_query(&self, prompt: &str) -> SoulResult<String> {
        Self::static_sub_query(&self.provider, self.config.sub_query_model(), &self.auth, prompt).await
    }

    /// Owned-prompt version of sub_query (needed for futures::future::join_all)
    async fn sub_query_owned(&self, prompt: String) -> SoulResult<String> {
        self.sub_query(&prompt).await
    }

    /// Static sub-query that doesn't borrow self (for use in spawned tasks)
    async fn static_sub_query(
        provider: &Arc<dyn Provider>,
        model: &ModelInfo,
        auth: &AuthProfile,
        prompt: &str,
    ) -> SoulResult<String> {
        let messages = vec![Message::user(prompt.to_string())];
        let (tx, _) = mpsc::unbounded_channel();

        let response = provider
            .stream(
                &messages,
                "You are a helpful assistant. Answer concisely based on the provided context.",
                &[],
                model,
                auth,
                tx,
            )
            .await?;

        Ok(response.text_content())
    }

    // ─── Context-as-Document Methods ─────────────────────────────────────

    /// Serialize a conversation history into a structured text document.
    ///
    /// This is the core of the context-as-document paradigm: the agent's full
    /// conversation history becomes an external document that grows forever.
    /// The LLM navigates it via RLM DSL commands instead of receiving it directly.
    pub fn serialize_conversation(messages: &[Message]) -> String {
        let mut doc = String::new();
        let mut turn = 0;
        let mut in_turn = false;

        for msg in messages {
            match msg.role {
                Role::User => {
                    turn += 1;
                    in_turn = true;
                    doc.push_str(&format!("\n## TURN {turn}\n### USER\n"));
                    doc.push_str(&msg.text_content());
                    doc.push('\n');
                }
                Role::Assistant => {
                    if !in_turn {
                        turn += 1;
                        in_turn = true;
                        doc.push_str(&format!("\n## TURN {turn}\n"));
                    }
                    doc.push_str("### ASSISTANT\n");

                    // Serialize all content blocks
                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } => {
                                doc.push_str(text);
                                doc.push('\n');
                            }
                            ContentBlock::ToolCall {
                                id,
                                name,
                                arguments,
                            } => {
                                doc.push_str(&format!(
                                    "### TOOL_CALL: {name} ({id})\nArguments: {arguments}\n"
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                Role::Tool => {
                    for block in &msg.content {
                        if let ContentBlock::ToolResult {
                            tool_call_id,
                            content,
                            is_error,
                        } = block
                        {
                            let status = if *is_error { "ERROR" } else { "OK" };
                            doc.push_str(&format!(
                                "### TOOL_RESULT ({tool_call_id}) [{status}]\n{content}\n"
                            ));
                        }
                    }
                }
                Role::System => {
                    // System messages are metadata, include for completeness
                    let text = msg.text_content();
                    if !text.is_empty() {
                        doc.push_str(&format!("### SYSTEM\n{text}\n"));
                    }
                }
            }
        }

        doc
    }

    /// Create a compact metadata summary of the context document.
    ///
    /// Returns a string suitable for injecting into the system prompt so the LLM
    /// knows the size and structure of its history without receiving the full content.
    pub fn context_metadata(context_doc: &str) -> String {
        let chars = context_doc.len();
        let lines = context_doc.lines().count();
        let turns = context_doc.matches("## TURN").count();
        let tool_calls = context_doc.matches("### TOOL_CALL:").count();
        let tool_results = context_doc.matches("### TOOL_RESULT").count();
        let est_tokens = crate::types::estimate_tokens_heuristic_pub(context_doc);

        // Compute recommended chunk sizes for the LLM
        let avg_chars_per_turn = if turns > 0 { chars / turns } else { chars };
        let recommended_chunk = (avg_chars_per_turn * 3).min(10_000).max(1_000);

        format!(
            "[Context: {turns} turns, {tool_calls} tool calls, {tool_results} tool results, \
             {lines} lines, {chars} chars, ~{est_tokens} tokens | \
             Avg {avg_chars_per_turn} chars/turn, recommended chunk_by_chars size: {recommended_chunk}]"
        )
    }

    /// Run an RLM context query — lets the LLM explore its own conversation history.
    ///
    /// This is the primary integration point for AgentLoop: when the agent needs to
    /// recall past work, it serializes the full conversation into a document and
    /// runs an RLM REPL loop to extract the needed information.
    pub async fn query_context(
        &self,
        messages: &[Message],
        query: &str,
    ) -> SoulResult<RlmResult> {
        let context_doc = Self::serialize_conversation(messages);
        self.completion(context_doc, Some(query)).await
    }

    /// Build the recent window + metadata for the LLM.
    ///
    /// Returns (recent_messages, metadata_line) where:
    /// - recent_messages: the last N messages that fit in the token budget
    /// - metadata_line: context metadata string to append to system prompt
    ///
    /// The LLM receives recent_messages directly and can use RLM DSL to access
    /// older history stored in the external document.
    pub fn build_context_window(
        messages: &[Message],
        max_recent_tokens: usize,
    ) -> (Vec<Message>, String) {
        let full_doc = Self::serialize_conversation(messages);
        Self::build_context_window_with_doc(messages, max_recent_tokens, &full_doc)
    }

    /// Build the recent window using a pre-computed serialized document.
    ///
    /// Same as `build_context_window` but avoids re-serializing the conversation
    /// when the caller already has a cached document string.
    pub fn build_context_window_with_doc(
        messages: &[Message],
        max_recent_tokens: usize,
        cached_doc: &str,
    ) -> (Vec<Message>, String) {
        let metadata = Self::context_metadata(cached_doc);

        // Walk backward from end, collecting messages until budget
        let mut recent = Vec::new();
        let mut tokens = 0;

        for msg in messages.iter().rev() {
            let msg_tokens = msg.estimate_tokens();
            if tokens + msg_tokens > max_recent_tokens && !recent.is_empty() {
                break;
            }
            recent.push(msg.clone());
            tokens += msg_tokens;
        }

        recent.reverse();
        (recent, metadata)
    }

    /// Serialize a single message into the context document format.
    ///
    /// Used for incremental document building: instead of re-serializing the
    /// entire conversation each turn, append only new messages.
    pub fn serialize_message(msg: &Message, turn_number: usize) -> String {
        let mut doc = String::new();
        match msg.role {
            Role::User => {
                doc.push_str(&format!("\n## TURN {turn_number}\n### USER\n"));
                doc.push_str(&msg.text_content());
                doc.push('\n');
            }
            Role::Assistant => {
                doc.push_str("### ASSISTANT\n");
                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => {
                            doc.push_str(text);
                            doc.push('\n');
                        }
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            doc.push_str(&format!(
                                "### TOOL_CALL: {name} ({id})\nArguments: {arguments}\n"
                            ));
                        }
                        _ => {}
                    }
                }
            }
            Role::Tool => {
                for block in &msg.content {
                    if let ContentBlock::ToolResult {
                        tool_call_id,
                        content,
                        is_error,
                    } = block
                    {
                        let status = if *is_error { "ERROR" } else { "OK" };
                        doc.push_str(&format!(
                            "### TOOL_RESULT ({tool_call_id}) [{status}]\n{content}\n"
                        ));
                    }
                }
            }
            Role::System => {
                let text = msg.text_content();
                if !text.is_empty() {
                    doc.push_str(&format!("### SYSTEM\n{text}\n"));
                }
            }
        }
        doc
    }

    /// Count the number of user turns in a message slice.
    pub fn count_turns(messages: &[Message]) -> usize {
        messages.iter().filter(|m| m.role == Role::User).count()
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
        assert_eq!(config.max_sub_calls, 0);
        assert_eq!(config.max_output_chars, 8192);
        assert!(config.sub_model.is_none());
        assert!(config.system_prompt_override.is_none());
    }

    #[test]
    fn rlm_config_output_cap() {
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
        // Default is 8192
        let config = RlmConfig::new(model.clone());
        assert_eq!(config.max_output_chars, 8192);

        // Can override
        let config = RlmConfig::new(model.clone()).with_max_output_chars(4096);
        assert_eq!(config.max_output_chars, 4096);

        // 0 = unlimited
        let config = RlmConfig::new(model).with_max_output_chars(0);
        assert_eq!(config.max_output_chars, 0);
    }

    #[test]
    fn rlm_config_sub_model_tiering() {
        let main_model = ModelInfo {
            id: "claude-opus".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 15.0,
            cost_per_output_token: 75.0,
        };
        let cheap_model = ModelInfo {
            id: "claude-haiku".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 4096,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.25,
            cost_per_output_token: 1.25,
        };

        let config = RlmConfig::new(main_model.clone())
            .with_sub_model(cheap_model.clone())
            .with_max_sub_calls(20);

        // Main model used for REPL iterations
        assert_eq!(config.model.id, "claude-opus");
        // Sub-queries use cheaper model
        assert_eq!(config.sub_query_model().id, "claude-haiku");
        assert_eq!(config.max_sub_calls, 20);

        // Without sub_model, falls back to main
        let config_no_sub = RlmConfig::new(main_model.clone());
        assert_eq!(config_no_sub.sub_query_model().id, "claude-opus");
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
    fn system_prompt_contains_lua_docs() {
        assert!(RLM_SYSTEM_PROMPT.contains("chunk_by_lines"));
        assert!(RLM_SYSTEM_PROMPT.contains("chunk_by_regex"));
        assert!(RLM_SYSTEM_PROMPT.contains("final_answer"));
        assert!(RLM_SYSTEM_PROMPT.contains("print_var"));
        assert!(RLM_SYSTEM_PROMPT.contains("search("));
        assert!(RLM_SYSTEM_PROMPT.contains("BM25"));
        assert!(RLM_SYSTEM_PROMPT.contains("```lua"));
    }

    #[test]
    fn system_prompt_describes_context_as_document() {
        assert!(RLM_SYSTEM_PROMPT.contains("conversation history"));
        assert!(RLM_SYSTEM_PROMPT.contains("## TURN"));
        assert!(RLM_SYSTEM_PROMPT.contains("nothing is ever deleted"));
    }

    // ─── Context Serialization Tests ─────────────────────────────────

    #[test]
    fn serialize_simple_conversation() {
        let messages = vec![
            Message::user("Hello, help me with Rust"),
            Message::assistant("Sure! What do you need?"),
            Message::user("How do I use async?"),
            Message::assistant("Use tokio and async/await"),
        ];

        let doc = RlmEngine::serialize_conversation(&messages);

        assert!(doc.contains("## TURN 1"));
        assert!(doc.contains("## TURN 2"));
        assert!(doc.contains("### USER"));
        assert!(doc.contains("### ASSISTANT"));
        assert!(doc.contains("Hello, help me with Rust"));
        assert!(doc.contains("Use tokio and async/await"));
    }

    #[test]
    fn serialize_with_tool_calls() {
        let messages = vec![
            Message::user("Read the file"),
            Message::new(
                Role::Assistant,
                vec![ContentBlock::tool_call(
                    "tc1",
                    "read",
                    serde_json::json!({"path": "/src/main.rs"}),
                )],
            ),
            Message::tool_result("tc1", "fn main() { println!(\"hello\"); }", false),
            Message::assistant("The file contains a hello world program"),
        ];

        let doc = RlmEngine::serialize_conversation(&messages);

        assert!(doc.contains("### TOOL_CALL: read (tc1)"));
        assert!(doc.contains("### TOOL_RESULT (tc1) [OK]"));
        assert!(doc.contains("fn main()"));
        assert!(doc.contains("hello world program"));
    }

    #[test]
    fn serialize_with_tool_error() {
        let messages = vec![
            Message::user("Read missing file"),
            Message::new(
                Role::Assistant,
                vec![ContentBlock::tool_call(
                    "tc1",
                    "read",
                    serde_json::json!({"path": "/nope"}),
                )],
            ),
            Message::tool_result("tc1", "File not found", true),
        ];

        let doc = RlmEngine::serialize_conversation(&messages);
        assert!(doc.contains("[ERROR]"));
        assert!(doc.contains("File not found"));
    }

    #[test]
    fn serialize_empty_conversation() {
        let doc = RlmEngine::serialize_conversation(&[]);
        assert!(doc.is_empty() || doc.trim().is_empty());
    }

    #[test]
    fn context_metadata_counts() {
        let messages = vec![
            Message::user("Q1"),
            Message::assistant("A1"),
            Message::user("Q2"),
            Message::new(
                Role::Assistant,
                vec![ContentBlock::tool_call(
                    "tc1",
                    "read",
                    serde_json::json!({}),
                )],
            ),
            Message::tool_result("tc1", "data", false),
            Message::assistant("A2 with data"),
        ];

        let doc = RlmEngine::serialize_conversation(&messages);
        let meta = RlmEngine::context_metadata(&doc);

        assert!(meta.contains("2 turns"));
        assert!(meta.contains("1 tool calls"));
        assert!(meta.contains("1 tool results"));
    }

    #[test]
    fn build_context_window_respects_budget() {
        // Create a long conversation
        let mut messages = Vec::new();
        for i in 0..50 {
            messages.push(Message::user(format!("Question {i}: {}", "x".repeat(100))));
            messages.push(Message::assistant(format!("Answer {i}: {}", "y".repeat(100))));
        }

        // Small budget — should only get recent messages
        let (recent, metadata) = RlmEngine::build_context_window(&messages, 500);

        assert!(recent.len() < messages.len());
        assert!(recent.len() > 0);
        assert!(metadata.contains("50 turns"));

        // The most recent message should always be included
        let last = recent.last().unwrap();
        assert!(last.text_content().contains("Answer 49"));
    }

    #[test]
    fn build_context_window_full_budget() {
        let messages = vec![
            Message::user("short"),
            Message::assistant("also short"),
        ];

        let (recent, metadata) = RlmEngine::build_context_window(&messages, 100_000);

        // With huge budget, should get all messages
        assert_eq!(recent.len(), 2);
        assert!(metadata.contains("1 turns"));
    }

    #[test]
    fn serialization_preserves_turn_structure() {
        // Verify we can chunk by turn boundary
        let messages = vec![
            Message::user("Turn 1 question"),
            Message::assistant("Turn 1 answer"),
            Message::user("Turn 2 question"),
            Message::assistant("Turn 2 answer"),
            Message::user("Turn 3 question"),
            Message::assistant("Turn 3 answer"),
        ];

        let doc = RlmEngine::serialize_conversation(&messages);

        // Should be splittable by "## TURN" regex
        let turns: Vec<&str> = doc.split("## TURN").filter(|s| !s.trim().is_empty()).collect();
        assert_eq!(turns.len(), 3);
    }

    // ─── Incremental Serialization Tests ──────────────────────────────

    #[test]
    fn incremental_matches_full_serialization() {
        let messages = vec![
            Message::user("Hello, help me with Rust"),
            Message::assistant("Sure! What do you need?"),
            Message::user("How do I use async?"),
            Message::new(
                Role::Assistant,
                vec![ContentBlock::tool_call(
                    "tc1",
                    "read",
                    serde_json::json!({"path": "/src/main.rs"}),
                )],
            ),
            Message::tool_result("tc1", "fn main() {}", false),
            Message::assistant("Use tokio and async/await"),
        ];

        // Full serialization
        let full_doc = RlmEngine::serialize_conversation(&messages);

        // Incremental serialization
        let mut incremental_doc = String::new();
        let mut turn_count = 0;
        for msg in &messages {
            if msg.role == Role::User {
                turn_count += 1;
            }
            incremental_doc.push_str(&RlmEngine::serialize_message(msg, turn_count));
        }

        assert_eq!(full_doc, incremental_doc);
    }

    #[test]
    fn incremental_with_tool_errors() {
        let messages = vec![
            Message::user("Read missing file"),
            Message::new(
                Role::Assistant,
                vec![ContentBlock::tool_call(
                    "tc1",
                    "read",
                    serde_json::json!({"path": "/nope"}),
                )],
            ),
            Message::tool_result("tc1", "File not found", true),
        ];

        let full_doc = RlmEngine::serialize_conversation(&messages);

        let mut incremental_doc = String::new();
        let mut turn_count = 0;
        for msg in &messages {
            if msg.role == Role::User {
                turn_count += 1;
            }
            incremental_doc.push_str(&RlmEngine::serialize_message(msg, turn_count));
        }

        assert_eq!(full_doc, incremental_doc);
    }

    #[test]
    fn count_turns_works() {
        let messages = vec![
            Message::user("Q1"),
            Message::assistant("A1"),
            Message::user("Q2"),
            Message::assistant("A2"),
        ];
        assert_eq!(RlmEngine::count_turns(&messages), 2);
        assert_eq!(RlmEngine::count_turns(&[]), 0);
    }

    #[test]
    fn build_context_window_with_doc_matches_original() {
        let messages = vec![
            Message::user("short"),
            Message::assistant("also short"),
        ];
        let doc = RlmEngine::serialize_conversation(&messages);

        let (recent1, meta1) = RlmEngine::build_context_window(&messages, 100_000);
        let (recent2, meta2) =
            RlmEngine::build_context_window_with_doc(&messages, 100_000, &doc);

        assert_eq!(recent1.len(), recent2.len());
        assert_eq!(meta1, meta2);
    }
}
