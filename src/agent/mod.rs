use std::sync::Arc;
use tokio::sync::mpsc;

use crate::context::{CompactionResult, CompactionStrategy, ContextConfig, ContextManager};
use crate::cost::budget::{BudgetEnforcer, BudgetStatus};
use crate::cost::CostTracker;
use crate::error::{SoulError, SoulResult};
use crate::executor::ExecutorRegistry;
use crate::hook::{BeforeAgentStartContext, BeforeToolCallContext, HookAction, HookPipeline};
use crate::provider::Provider;
use crate::rlm::{RecallTool, RlmEngine, SharedContextDoc};
use crate::semantic_recursion::{RetrievalQuery, SemanticContextEngine};
use crate::subagent::{SubagentConfig, SubagentSpawner};
use crate::tool::{ToolOutput, ToolRegistry};
use crate::types::*;

/// Normalize tool names from LLM outputs that use non-canonical names.
///
/// Weaker models (especially qwen/local models) hallucinate tool names from their
/// training data (OpenAI conventions) rather than the actual registered names.
/// Map known aliases to their canonical amai tool names.
fn normalize_tool_name(name: &str) -> &str {
    match name {
        "read_file" | "read_text_file" | "cat" | "view_file" => "read",
        "write_file" | "create_file" | "save_file" | "write_text_file" => "write",
        "list_files" | "list_directory" | "list_dir" => "ls",
        "search_files" | "search_code" | "search_text" => "grep",
        "run_command" | "execute_command" | "run_shell" | "shell" | "terminal" | "exec" | "execute" | "run" => "bash",
        "edit_file" | "update_file" | "modify_file" | "replace_in_file" => "edit",
        "append_to_file" | "append_text" => "append",
        "find_files" | "search_directory" | "search_filesystem" => "find",
        "glob_files" | "glob_pattern" => "glob",
        "grep_search" | "grep_files" | "search_in_files" => "grep",
        _ => name,
    }
}

/// Maximum number of retries when a structural failure is detected in the LLM response.
///
/// Structural failures are protocol-level breakages from weaker/free-tier models:
/// - Empty/whitespace-only responses with no tool calls
/// - JSON tool invocations written as text instead of using the tool calling mechanism
/// - No tool calls on the first turn when tools are available (model failed to engage)
/// - Degenerate short responses (<50 chars) with no tool calls
///
/// These failures are intermittent — the same model usually succeeds on retry.
/// Failed responses are DISCARDED (not appended to conversation history) to prevent
/// conversation poisoning where a broken response causes all subsequent turns to fail.
const MAX_STRUCTURAL_RETRIES: usize = 2;

/// Detects structural failures in LLM responses — protocol-level breakages that waste turns.
///
/// This is NOT quality scoring. These heuristics catch responses that are fundamentally broken:
/// the model failed to use the tool calling mechanism, returned nothing, or produced degenerate
/// output. Each heuristic is documented with the production failure it catches and its false
/// positive risk.
///
/// Evidence: MiniMax went from 0% success (Run 1: 1 turn, 0 tool calls) to 100% (Run 3)
/// through prompt engineering. The failures are intermittent and recoverable — retry with
/// the same model usually succeeds.
pub struct StructuralFailureDetector;

impl StructuralFailureDetector {
    /// Analyze an LLM response for structural failures.
    ///
    /// Returns `Some(failure_kind)` if the response is structurally broken,
    /// `None` if the response appears valid (even if low quality).
    ///
    /// # Arguments
    /// * `response` - The assistant message from the LLM
    /// * `turn` - Current turn number (0-indexed). Turn 0 has special handling.
    /// * `tools_available` - Whether tools were offered to the model
    pub fn detect(response: &Message, turn: usize, tools_available: bool) -> Option<String> {
        let text = response.text_content();
        let has_tool_calls = response.has_tool_calls();
        let text_trimmed = text.trim();

        // Heuristic 1: Empty response — text is empty/whitespace AND no tool calls.
        //
        // Catches: Model returns empty string or whitespace-only content. This happens when
        // the model's generation is cut short or it produces a degenerate empty completion.
        // False positive risk: Near zero. A valid response always has either text or tool calls.
        if text_trimmed.is_empty() && !has_tool_calls {
            return Some("empty_response".into());
        }

        // Heuristic 2: JSON-as-text — model wrote tool invocations as text instead of using
        // the structured tool calling mechanism.
        //
        // Catches: MiniMax Run 1 failure pattern where the model outputs
        // `{"name": "read", "arguments": {"path": "/foo"}}` as a text block instead of
        // producing a ToolCall content block. This is the #1 failure mode on free-tier models
        // when prompt pressure overwhelms the tool calling mechanism.
        // False positive risk: Low. Legitimate text rarely contains both "name" and "arguments"
        // in a JSON-like structure without also having real tool calls.
        if !has_tool_calls
            && text_trimmed.contains("\"name\"")
            && text_trimmed.contains("\"arguments\"")
            && (text_trimmed.starts_with('{') || text_trimmed.starts_with('['))
        {
            return Some("json_as_text".into());
        }

        // Heuristic 3: No tools on first turn — turn 0, tools available, model produced text
        // but no tool calls. On the first turn an agent should ALWAYS call at least one tool
        // (typically `ls` or `read`) to orient itself.
        //
        // Catches: Model ignoring tool definitions and producing a text-only "plan" or
        // conversational response instead of acting. Common when prompt is too large and
        // the model can't parse tool schemas.
        // False positive risk: Moderate for non-agent use cases, but this detector is only
        // used in the AgentLoop which is always tool-augmented. A >20 char threshold
        // avoids triggering on legitimate short completions.
        if turn == 0 && tools_available && !has_tool_calls && text_trimmed.len() > 20 {
            return Some("no_tools_turn_0".into());
        }

        // Heuristic 4: Degenerate short on turn 0 — response is very short (<50 chars),
        // no tool calls, on the first turn.
        //
        // Catches: Model outputs "OK", "Sure, I'll help", "Let me think..." without actually
        // doing anything on the very first turn. These parrot responses waste the critical
        // orientation turn.
        // False positive risk: Low — restricted to turn 0 only. On later turns, short text
        // responses are legitimate completions ("Done.", "Tests pass.").
        if turn == 0
            && tools_available
            && !has_tool_calls
            && text_trimmed.len() < 50
            && !text_trimmed.is_empty()
        {
            return Some("degenerate_short".into());
        }

        None
    }
}

/// The steerable agent loop — core execution engine
pub struct AgentLoop {
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    config: AgentConfig,
    hooks: HookPipeline,
    context_manager: ContextManager,
    semantic_engine: Option<SemanticContextEngine>,
    cost_tracker: Option<CostTracker>,
    budget: Option<BudgetEnforcer>,
    executor_registry: Option<ExecutorRegistry>,
    summarizer: SubagentSpawner,
}

/// Options for running the agent loop
pub struct RunOptions {
    pub session_id: String,
    pub initial_messages: Vec<Message>,
}

impl AgentLoop {
    pub fn new(provider: Arc<dyn Provider>, tools: ToolRegistry, config: AgentConfig) -> Self {
        let context_config = ContextConfig {
            max_tokens: config.model.context_window,
            compaction_threshold: config.compaction_threshold,
            safety_margin: config.token_safety_margin,
            ..Default::default()
        };

        let semantic_engine = match config.context_strategy {
            ContextStrategy::SemanticGraph => Some(SemanticContextEngine::new()),
            _ => None,
        };

        let mut summarizer = SubagentSpawner::new();
        summarizer.add_provider(provider.kind(), provider.clone());

        Self {
            provider,
            tools,
            config,
            hooks: HookPipeline::new(),
            context_manager: ContextManager::new(context_config),
            semantic_engine,
            cost_tracker: None,
            budget: None,
            executor_registry: None,
            summarizer,
        }
    }

    pub fn with_hooks(mut self, hooks: HookPipeline) -> Self {
        self.hooks = hooks;
        self
    }

    pub fn with_cost_tracker(mut self, tracker: CostTracker) -> Self {
        self.cost_tracker = Some(tracker);
        self
    }

    pub fn with_budget(mut self, policy: crate::cost::BudgetPolicy) -> Self {
        self.budget = Some(BudgetEnforcer::new(policy));
        self
    }

    pub fn with_executor_registry(mut self, registry: ExecutorRegistry) -> Self {
        self.executor_registry = Some(registry);
        self
    }

    /// Run the steerable agent loop
    ///
    /// Returns the new messages produced during this run.
    /// The `steering_rx` channel allows injecting messages mid-execution (interruption).
    pub async fn run(
        &mut self,
        options: RunOptions,
        event_tx: mpsc::UnboundedSender<AgentEvent>,
        mut steering_rx: mpsc::UnboundedReceiver<Message>,
    ) -> SoulResult<Vec<Message>> {
        let _ = event_tx.send(AgentEvent::AgentStart {
            session_id: options.session_id.clone(),
        });

        // Run before_agent_start hooks
        let hook_ctx = BeforeAgentStartContext {
            system_prompt: self.config.system_prompt.clone(),
            tools: self.tools.definitions(),
            messages: options.initial_messages.clone(),
            session_id: options.session_id.clone(),
        };

        let hook_result = self.hooks.run_before_agent_start(hook_ctx).await?;
        let (system_prompt, mut messages) = match hook_result {
            HookAction::Continue(ctx) => (ctx.system_prompt, ctx.messages),
            HookAction::Cancel(reason) => {
                let _ = event_tx.send(AgentEvent::Error {
                    message: format!("Agent start cancelled: {reason}"),
                });
                return Ok(Vec::new());
            }
        };

        let tool_defs = self.tools.definitions();
        let mut new_messages: Vec<Message> = Vec::new();
        let mut turn = 0;
        let max_turns = self.config.max_turns.unwrap_or(100);
        let mut has_more_tool_calls = true;

        // For RLM/SemanticGraph: keep the full history separately (never truncated)
        // `messages` may be windowed for the LLM, but `full_history` grows forever
        let mut full_history: Vec<Message> = messages.clone();

        // For RLM: maintain incrementally-updated cached serialized document.
        // Instead of re-serializing the entire conversation each turn (O(n²) over
        // the session), we serialize once and append new messages incrementally.
        // Shared via Arc<RwLock> so the RecallTool can read it concurrently.
        let cached_doc: SharedContextDoc = Arc::new(std::sync::RwLock::new(
            if self.config.context_strategy == ContextStrategy::Rlm {
                RlmEngine::serialize_conversation(&full_history)
            } else {
                String::new()
            },
        ));
        let mut cached_turn_count = if self.config.context_strategy == ContextStrategy::Rlm {
            RlmEngine::count_turns(&full_history)
        } else {
            0
        };

        // Register the RecallTool so the agent can search its own history
        if self.config.context_strategy == ContextStrategy::Rlm {
            let recall_tool = RecallTool::new(
                self.provider.clone(),
                self.config.model.clone(),
                AuthProfile::new(self.config.model.provider.clone(), ""),
                cached_doc.clone(),
            );
            self.tools.dynamic_handle().register(Arc::new(recall_tool));
        }

        // Ingest initial messages into semantic engine
        if let Some(ref mut engine) = self.semantic_engine {
            for msg in &messages {
                if msg.role == Role::User {
                    engine.ingest_user_request(&msg.text_content());
                }
            }
        }

        while has_more_tool_calls && turn < max_turns {
            // Check for steering messages
            while let Ok(steering_msg) = steering_rx.try_recv() {
                messages.push(steering_msg.clone());
                new_messages.push(steering_msg.clone());
                if self.config.context_strategy == ContextStrategy::Rlm {
                    if steering_msg.role == Role::User {
                        cached_turn_count += 1;
                    }
                    cached_doc.write().unwrap().push_str(&RlmEngine::serialize_message(
                        &steering_msg, cached_turn_count,
                    ));
                }
                full_history.push(steering_msg);
            }

            // ─── Context Strategy ────────────────────────────────────────
            let system_tokens = system_prompt.len() / 4;

            match self.config.context_strategy {
                ContextStrategy::Classic => {
                    // LLM-first compaction — summarize old messages, fall back to prune
                    if self
                        .context_manager
                        .needs_compaction(&messages, system_tokens)
                    {
                        let messages_before = messages.len();
                        let tokens_before = ContextManager::estimate_tokens(&messages);

                        let _ = event_tx.send(AgentEvent::CompactionStart {
                            messages_before,
                            tokens_before,
                        });

                        // Attempt LLM summarization of old messages.
                        // Keep the most recent min_preserved_messages verbatim; summarize the rest.
                        let keep_recent = self.context_manager.config().min_preserved_messages.max(4);
                        let summarize_result = if messages.len() > keep_recent {
                            let old_msgs = &messages[..messages.len() - keep_recent];
                            let auth = AuthProfile::new(self.config.model.provider.clone(), "");
                            let sub_config = SubagentConfig {
                                name: "summarizer".into(),
                                model: self.config.model.clone(),
                                system_prompt: "You are a concise conversation summarizer. Preserve all key decisions, file paths modified, tool outputs, and task state. Be brief but complete.".into(),
                                max_turns: 1,
                                tools: vec![],
                            };
                            self.summarizer.summarize(&sub_config, old_msgs, &auth).await
                        } else {
                            Err(SoulError::CompactionFailed("not enough messages to summarize".into()))
                        };

                        let compaction_result = match summarize_result {
                            Ok(summary) => {
                                // Replace old messages with summary + keep recent verbatim
                                let keep_start = messages.len() - keep_recent;
                                let recent: Vec<Message> = messages[keep_start..].to_vec();
                                let summary_msg = Message::user(format!(
                                    "[Conversation summary — prior context compressed]\n\n{}",
                                    summary.content
                                ));
                                messages = std::iter::once(summary_msg).chain(recent).collect();

                                let tokens_after = ContextManager::estimate_tokens(&messages) + system_tokens;
                                Ok(CompactionResult {
                                    messages_before,
                                    messages_after: messages.len(),
                                    tokens_before,
                                    tokens_after,
                                    strategy: CompactionStrategy::Summarize,
                                })
                            }
                            Err(_) => {
                                // Fall back to structural compaction (offload + prune)
                                self.context_manager.compact(&mut messages, None, system_tokens)
                            }
                        };

                        match compaction_result {
                            Ok(result) => {
                                let _ = event_tx.send(AgentEvent::CompactionEnd {
                                    messages_after: result.messages_after,
                                    tokens_after: result.tokens_after,
                                });
                            }
                            Err(e) => {
                                let _ = event_tx.send(AgentEvent::Error {
                                    message: format!("Compaction failed: {e}"),
                                });
                            }
                        }
                    }
                }
                ContextStrategy::Rlm => {
                    // RLM: no compaction. Window the full history for the LLM.
                    // Reserve ~20% of context for system prompt + tools + output
                    let recent_budget = (self.config.model.context_window as f64 * 0.6) as usize;
                    let doc_snapshot = cached_doc.read().unwrap().clone();
                    let (recent, _metadata) =
                        RlmEngine::build_context_window_with_doc(
                            &full_history, recent_budget, &doc_snapshot,
                        );
                    messages = recent;
                }
                ContextStrategy::SemanticGraph => {
                    // SemanticGraph: no compaction. Use retrieval to build optimal window.
                    if let Some(ref mut engine) = self.semantic_engine {
                        // Get the current task/query from the most recent user message
                        let query_text = full_history
                            .iter()
                            .rev()
                            .find(|m| m.role == Role::User)
                            .map(|m| m.text_content())
                            .unwrap_or_default();

                        let budget = (self.config.model.context_window as f64 * 0.6) as usize;
                        let window = engine.retrieve(&RetrievalQuery {
                            text: query_text,
                            max_tokens: budget,
                            max_results: 50,
                            include_graph_neighbors: true,
                        });

                        // Replace messages with retrieved context + last few messages
                        // Always include the most recent messages for continuity
                        let recent_count = 6.min(full_history.len());
                        let recent_start = full_history.len().saturating_sub(recent_count);

                        let mut retrieved = window.messages;
                        // Append recent messages that aren't already in retrieved
                        for msg in &full_history[recent_start..] {
                            retrieved.push(msg.clone());
                        }
                        messages = retrieved;
                    }
                }
            }

            let _ = event_tx.send(AgentEvent::TurnStart { turn });

            // Inject turn budget awareness + context metadata into system prompt
            let remaining = max_turns.saturating_sub(turn + 1);
            let mut effective_prompt = if remaining <= 5 {
                format!(
                    "{system_prompt}\n\n[TURN {}/{max_turns} — {remaining} remaining. \
                    Stop exploring and produce your final output NOW.]",
                    turn + 1
                )
            } else {
                format!(
                    "{system_prompt}\n\n[Turn {}/{max_turns} — {remaining} remaining]",
                    turn + 1
                )
            };

            // For RLM: inject context metadata so LLM knows its full history size
            if self.config.context_strategy == ContextStrategy::Rlm {
                let metadata = RlmEngine::context_metadata(&cached_doc.read().unwrap());
                effective_prompt.push_str(&format!("\n\n{metadata}"));
            }

            // Stream LLM response with structural failure retry loop.
            // Failed responses are DISCARDED — never appended to conversation history.
            // This prevents conversation poisoning where a broken response causes all
            // subsequent turns to degrade.
            let tools_available = !tool_defs.is_empty();
            let mut structural_attempt = 0;

            let assistant_msg = loop {
                let (delta_tx, mut delta_rx) = mpsc::unbounded_channel();

                let msg_id = uuid::Uuid::new_v4().to_string();
                let _ = event_tx.send(AgentEvent::MessageStart {
                    message_id: msg_id.clone(),
                });

                // Forward deltas as events
                let event_tx_clone = event_tx.clone();
                let msg_id_clone = msg_id.clone();
                let delta_forwarder = tokio::spawn(async move {
                    while let Some(delta) = delta_rx.recv().await {
                        let _ = event_tx_clone.send(AgentEvent::MessageDelta {
                            message_id: msg_id_clone.clone(),
                            delta,
                        });
                    }
                });

                let candidate = self
                    .provider
                    .stream(
                        &messages,
                        &effective_prompt,
                        &tool_defs,
                        &self.config.model,
                        &AuthProfile::new(self.config.model.provider.clone(), ""),
                        delta_tx,
                    )
                    .await?;

                delta_forwarder.await.ok();

                // Check for structural failure
                if let Some(failure_kind) =
                    StructuralFailureDetector::detect(&candidate, turn, tools_available)
                {
                    if structural_attempt < MAX_STRUCTURAL_RETRIES {
                        let preview = candidate.text_content();
                        let preview_truncated = if preview.len() > 100 {
                            format!("{}...", &preview[..100])
                        } else {
                            preview
                        };

                        let _ = event_tx.send(AgentEvent::StructuralRetry {
                            turn,
                            attempt: structural_attempt + 1,
                            failure_kind: failure_kind.clone(),
                            model_response_preview: preview_truncated,
                        });

                        tracing::warn!(
                            turn,
                            attempt = structural_attempt + 1,
                            failure_kind,
                            "Structural failure detected — retrying (discarding broken response)"
                        );

                        structural_attempt += 1;
                        continue; // Discard candidate, retry
                    } else {
                        // Exhausted retries — proceed with broken response
                        tracing::error!(
                            turn,
                            attempts = structural_attempt + 1,
                            failure_kind,
                            "Structural failure persists after {} retries — proceeding with broken response",
                            MAX_STRUCTURAL_RETRIES
                        );
                    }
                }

                break candidate;
            };

            self.context_manager.reset_circuit_breaker();

            // Record cost after LLM response
            if let Some(ref cost_tracker) = self.cost_tracker {
                let tool_names: Vec<String> = assistant_msg
                    .tool_calls()
                    .iter()
                    .filter_map(|tc| {
                        if let ContentBlock::ToolCall { name, .. } = tc {
                            Some(name.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                let usage = assistant_msg
                    .usage
                    .clone()
                    .unwrap_or_else(|| TokenUsage::new(0, 0));
                let cost_event = cost_tracker
                    .record_turn(
                        &self.config.model.id,
                        &usage,
                        &self.config.model,
                        &tool_names,
                    )
                    .await;
                let _ = event_tx.send(AgentEvent::Cost(cost_event));

                // Check budget
                if let Some(ref budget) = self.budget {
                    let total_cost = cost_tracker.total_cost_usd().await;
                    let total_tokens = cost_tracker.total_tokens().await;
                    let total_turns = cost_tracker.total_turns().await;

                    match budget.check(total_cost, total_tokens, total_turns) {
                        BudgetStatus::Exceeded => {
                            let _ = event_tx.send(AgentEvent::Error {
                                message: format!("Budget exceeded: ${:.4} spent", total_cost),
                            });
                            return Err(SoulError::BudgetExceeded {
                                message: format!(
                                    "Budget exceeded: ${:.4} spent, {} tokens, {} turns",
                                    total_cost, total_tokens, total_turns
                                ),
                            });
                        }
                        BudgetStatus::Warning => {
                            let _ = event_tx.send(AgentEvent::Error {
                                message: format!("Budget warning: ${:.4} spent", total_cost),
                            });
                        }
                        BudgetStatus::Ok => {}
                    }
                }
            }

            let _ = event_tx.send(AgentEvent::MessageEnd {
                message: assistant_msg.clone(),
            });

            messages.push(assistant_msg.clone());
            new_messages.push(assistant_msg.clone());
            // For infinite context modes, also push to full history
            if !matches!(self.config.context_strategy, ContextStrategy::Classic) {
                if self.config.context_strategy == ContextStrategy::Rlm {
                    cached_doc.write().unwrap().push_str(&RlmEngine::serialize_message(
                        &assistant_msg, cached_turn_count,
                    ));
                }
                full_history.push(assistant_msg.clone());
            }

            // Ingest into semantic engine if active
            let response_node_id = if let Some(ref mut engine) = self.semantic_engine {
                // Find the last user request node
                let last_user_node = full_history
                    .iter()
                    .rev()
                    .position(|m| m.role == Role::User)
                    .map(|pos| full_history.len() - 1 - pos)
                    .unwrap_or(0);

                Some(engine.ingest_llm_response(
                    &assistant_msg.text_content(),
                    last_user_node as u64,
                    Some(&self.config.model.id),
                ))
            } else {
                None
            };

            // Extract and execute tool calls — parallel when multiple calls present
            let tool_calls: Vec<&ContentBlock> = assistant_msg.tool_calls();
            has_more_tool_calls = !tool_calls.is_empty();

            if has_more_tool_calls {
                // Phase 1: Run before_tool_call hooks sequentially (hooks may cancel/modify)
                struct PreparedCall {
                    call_id: String,
                    tool_name: String,
                    tool_args: serde_json::Value,
                    tool_args_str: String,
                }
                let mut prepared: Vec<PreparedCall> = Vec::new();

                for tc in &tool_calls {
                    if let ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } = tc
                    {
                        let hook_ctx = BeforeToolCallContext {
                            tool_name: name.clone(),
                            tool_call_id: id.clone(),
                            arguments: arguments.clone(),
                            session_id: options.session_id.clone(),
                        };

                        let hook_result = self.hooks.run_before_tool_call(hook_ctx).await?;
                        match hook_result {
                            HookAction::Continue(ctx) => {
                                let args_str = ctx.arguments.to_string();
                                prepared.push(PreparedCall {
                                    call_id: id.clone(),
                                    tool_name: ctx.tool_name,
                                    tool_args: ctx.arguments,
                                    tool_args_str: args_str,
                                });
                            }
                            HookAction::Cancel(reason) => {
                                let err_msg = Message::tool_result(
                                    id.clone(),
                                    format!("Blocked: {reason}"),
                                    true,
                                );
                                messages.push(err_msg.clone());
                                new_messages.push(err_msg.clone());
                                if !matches!(self.config.context_strategy, ContextStrategy::Classic)
                                {
                                    if self.config.context_strategy == ContextStrategy::Rlm {
                                        cached_doc.write().unwrap().push_str(&RlmEngine::serialize_message(
                                            &err_msg, cached_turn_count,
                                        ));
                                    }
                                    full_history.push(err_msg);
                                }
                            }
                        };
                    }
                }

                // Phase 2: Execute all prepared tool calls concurrently
                // Emit start events before launching
                for p in &prepared {
                    let _ = event_tx.send(AgentEvent::ToolExecutionStart {
                        tool_call_id: p.call_id.clone(),
                        tool_name: p.tool_name.clone(),
                        arguments: p.tool_args.clone(),
                    });
                }

                // Build futures for concurrent execution (no spawn needed —
                // futures::future::join_all runs on the current task, so shared borrows are fine)
                // Take shared references to avoid moving &mut self into closures.
                let exec_registry_ref = self.executor_registry.as_ref();
                let tools_ref = &self.tools;

                let exec_futures: Vec<_> = prepared.iter().map(|p| {
                    let event_tx = &event_tx;
                    async move {
                        let (partial_tx, mut partial_rx) = mpsc::unbounded_channel::<String>();
                        let event_tx_c = event_tx.clone();
                        let tc_id = p.call_id.clone();
                        let partial_forwarder = tokio::spawn(async move {
                            while let Some(partial) = partial_rx.recv().await {
                                let _ = event_tx_c.send(AgentEvent::ToolExecutionUpdate {
                                    tool_call_id: tc_id.clone(),
                                    partial_result: partial,
                                });
                            }
                        });

                        // Try the original name first; fall back to normalized alias if not found.
                        // This allows test tools registered as "read_file" to work normally,
                        // while also handling weak models that hallucinate OpenAI-convention names.
                        let alias = normalize_tool_name(&p.tool_name);
                        let effective_name = if alias != p.tool_name.as_str()
                            && exec_registry_ref.is_none()
                            && tools_ref.get(&p.tool_name).is_none()
                            && tools_ref.get_dynamic(&p.tool_name).is_none()
                        {
                            alias
                        } else {
                            p.tool_name.as_str()
                        };
                        let output = if let Some(exec_registry) = exec_registry_ref {
                            exec_registry
                                .execute(effective_name, &p.call_id, p.tool_args.clone(), Some(partial_tx))
                                .await
                        } else if let Some(tool) = tools_ref.get(effective_name) {
                            tool.execute(&p.call_id, p.tool_args.clone(), Some(partial_tx)).await
                        } else if let Some(tool) = tools_ref.get_dynamic(effective_name) {
                            tool.execute(&p.call_id, p.tool_args.clone(), Some(partial_tx)).await
                        } else {
                            drop(partial_tx);
                            partial_forwarder.await.ok();
                            return ToolOutput::error(format!("Unknown tool: {}", p.tool_name));
                        };

                        partial_forwarder.await.ok();

                        match output {
                            Ok(out) => out,
                            Err(e) => ToolOutput::error(format!("Tool error: {e}")),
                        }
                    }
                }).collect();

                let results = futures::future::join_all(exec_futures).await;

                // Phase 3: Process results in order (hooks, events, messages, semantic engine)
                for (p, result) in prepared.iter().zip(results.into_iter()) {
                    let (content, is_error) = self.hooks.transform_tool_result(
                        &p.tool_name,
                        result.content.clone(),
                        result.is_error,
                    );

                    let result_block =
                        ContentBlock::tool_result(p.call_id.clone(), &content, is_error);
                    let _ = event_tx.send(AgentEvent::ToolExecutionEnd {
                        tool_call_id: p.call_id.clone(),
                        result: result_block.clone(),
                    });

                    let result_msg =
                        Message::tool_result(p.call_id.clone(), content.clone(), is_error);
                    messages.push(result_msg.clone());
                    new_messages.push(result_msg.clone());

                    if !matches!(self.config.context_strategy, ContextStrategy::Classic) {
                        if self.config.context_strategy == ContextStrategy::Rlm {
                            cached_doc.write().unwrap().push_str(&RlmEngine::serialize_message(
                                &result_msg, cached_turn_count,
                            ));
                        }
                        full_history.push(result_msg);
                    }

                    if let (Some(ref mut engine), Some(resp_node)) =
                        (&mut self.semantic_engine, response_node_id)
                    {
                        engine.ingest_tool_interaction(
                            &p.tool_name,
                            &p.tool_args_str,
                            &content,
                            resp_node,
                        );
                    }
                }

                // Phase 4: Check steering after all parallel tools complete
                if let Ok(steering_msg) = steering_rx.try_recv() {
                    messages.push(steering_msg.clone());
                    new_messages.push(steering_msg.clone());
                    if !matches!(self.config.context_strategy, ContextStrategy::Classic) {
                        if self.config.context_strategy == ContextStrategy::Rlm {
                            if steering_msg.role == Role::User {
                                cached_turn_count += 1;
                            }
                            cached_doc.write().unwrap().push_str(&RlmEngine::serialize_message(
                                &steering_msg, cached_turn_count,
                            ));
                        }
                        full_history.push(steering_msg);
                    }
                }
            }

            let _ = event_tx.send(AgentEvent::TurnEnd {
                turn,
                message: assistant_msg,
            });

            turn += 1;
        }

        // Fire void hooks
        self.hooks.fire_agent_end(&new_messages).await;

        let _ = event_tx.send(AgentEvent::AgentEnd {
            session_id: options.session_id,
            messages: new_messages.clone(),
        });

        Ok(new_messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use async_trait::async_trait;
    use serde_json::json;

    // Mock provider that returns canned responses
    struct MockProvider {
        responses: std::sync::Mutex<Vec<Message>>,
    }

    impl MockProvider {
        fn new(responses: Vec<Message>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
            }
        }
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Custom("mock".into())
        }

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _model: &ModelInfo,
            _auth: &AuthProfile,
            event_tx: mpsc::UnboundedSender<StreamDelta>,
        ) -> SoulResult<Message> {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                return Err(SoulError::Provider("No more mock responses".into()));
            }
            let msg = responses.remove(0);

            // Simulate streaming
            for block in &msg.content {
                if let ContentBlock::Text { text } = block {
                    let _ = event_tx.send(StreamDelta::TextDelta { text: text.clone() });
                }
            }
            Ok(msg)
        }

        async fn count_tokens(
            &self,
            messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _model: &ModelInfo,
            _auth: &AuthProfile,
        ) -> SoulResult<usize> {
            Ok(ContextManager::estimate_tokens(messages))
        }

        async fn probe(
            &self,
            _model: &ModelInfo,
            _auth: &AuthProfile,
        ) -> SoulResult<crate::provider::ProbeResult> {
            Ok(crate::provider::ProbeResult {
                healthy: true,
                rate_limit_remaining: Some(1.0),
                rate_limit_utilization: Some(0.0),
            })
        }
    }

    struct UppercaseTool;

    #[async_trait]
    impl Tool for UppercaseTool {
        fn name(&self) -> &str {
            "uppercase"
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "uppercase".into(),
                description: "Convert text to uppercase".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"]
                }),
            }
        }

        async fn execute(
            &self,
            _call_id: &str,
            arguments: serde_json::Value,
            _partial_tx: Option<mpsc::UnboundedSender<String>>,
        ) -> SoulResult<ToolOutput> {
            let text = arguments.get("text").and_then(|v| v.as_str()).unwrap_or("");
            Ok(ToolOutput::success(text.to_uppercase()))
        }
    }

    fn test_model() -> ModelInfo {
        ModelInfo {
            id: "test-model".into(),
            provider: ProviderKind::Custom("mock".into()),
            context_window: 100_000,
            max_output_tokens: 4096,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        }
    }

    #[tokio::test]
    async fn agent_loop_simple_response() {
        let response = Message::assistant("Hello! How can I help?");
        let provider = Arc::new(MockProvider::new(vec![response]));

        let tools = ToolRegistry::new();
        let config = AgentConfig::new(test_model(), "You are helpful");

        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let (_steering_tx, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "test-session".into(),
            initial_messages: vec![Message::user("hi")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text_content(), "Hello! How can I help?");

        // Verify events
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentStart { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::AgentEnd { .. })));
    }

    #[tokio::test]
    async fn agent_loop_with_tool_call() {
        // First response: tool call
        let tool_call_response = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "uppercase",
                json!({"text": "hello world"}),
            )],
        );
        // Second response: final text
        let final_response = Message::assistant("The uppercase is: HELLO WORLD");

        let provider = Arc::new(MockProvider::new(vec![tool_call_response, final_response]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let config = AgentConfig::new(test_model(), "You are helpful");
        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let (_steering_tx, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "test-session".into(),
            initial_messages: vec![Message::user("uppercase hello world")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();

        // Should have: tool_call_msg, tool_result_msg, final_msg
        assert_eq!(result.len(), 3);

        // Verify tool execution events
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionStart { tool_name, .. } if tool_name == "uppercase")));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. })));
    }

    #[tokio::test]
    async fn agent_loop_unknown_tool() {
        let tool_call_response = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "nonexistent_tool",
                json!({}),
            )],
        );
        let final_response = Message::assistant("I see, that tool doesn't exist");

        let provider = Arc::new(MockProvider::new(vec![tool_call_response, final_response]));
        let tools = ToolRegistry::new(); // empty registry

        let config = AgentConfig::new(test_model(), "You are helpful");
        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, _event_rx) = mpsc::unbounded_channel();
        let (_steering_tx, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "test-session".into(),
            initial_messages: vec![Message::user("call nonexistent")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();

        // Should have tool_call, error result, and final response
        assert_eq!(result.len(), 3);

        // The error result should mention unknown tool
        let error_msg = &result[1];
        assert_eq!(error_msg.role, Role::Tool);
    }

    #[tokio::test]
    async fn agent_loop_max_turns() {
        // Provider always returns tool calls (infinite loop scenario)
        let responses: Vec<Message> = (0..5)
            .map(|i| {
                Message::new(
                    Role::Assistant,
                    vec![ContentBlock::tool_call(
                        format!("tc{i}"),
                        "uppercase",
                        json!({"text": "test"}),
                    )],
                )
            })
            .collect();

        let provider = Arc::new(MockProvider::new(responses));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let mut config = AgentConfig::new(test_model(), "You are helpful");
        config.max_turns = Some(3);

        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, _) = mpsc::unbounded_channel();
        let (_, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "test-session".into(),
            initial_messages: vec![Message::user("loop forever")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();
        // Should have stopped at max_turns (3 turns × 2 messages each = 6)
        assert!(result.len() <= 6);
    }

    #[tokio::test]
    async fn agent_loop_steering_interrupts() {
        // Response with two tool calls
        let multi_tool_response = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::tool_call("tc1", "uppercase", json!({"text": "first"})),
                ContentBlock::tool_call("tc2", "uppercase", json!({"text": "second"})),
            ],
        );
        let final_response = Message::assistant("Interrupted and redirected");

        let provider = Arc::new(MockProvider::new(vec![multi_tool_response, final_response]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let config = AgentConfig::new(test_model(), "You are helpful");
        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, _) = mpsc::unbounded_channel();
        let (steering_tx, steering_rx) = mpsc::unbounded_channel();

        // Send steering message immediately (will be picked up between tool calls)
        let _ = steering_tx.send(Message::user("Stop! Do something else."));

        let options = RunOptions {
            session_id: "test-session".into(),
            initial_messages: vec![Message::user("do two things")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();
        // Should have been interrupted
        assert!(!result.is_empty());
    }

    // ─── Context Strategy Tests ─────────────────────────────────────────

    #[tokio::test]
    async fn agent_loop_rlm_strategy_simple() {
        let response = Message::assistant("Hello from RLM mode!");
        let provider = Arc::new(MockProvider::new(vec![response]));

        let tools = ToolRegistry::new();
        let mut config = AgentConfig::new(test_model(), "You are helpful");
        config.context_strategy = ContextStrategy::Rlm;

        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, _) = mpsc::unbounded_channel();
        let (_, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "rlm-test".into(),
            initial_messages: vec![Message::user("hi")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text_content(), "Hello from RLM mode!");
    }

    #[tokio::test]
    async fn agent_loop_rlm_with_tools() {
        let tool_call = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "uppercase",
                json!({"text": "hello"}),
            )],
        );
        let final_response = Message::assistant("Done: HELLO");

        let provider = Arc::new(MockProvider::new(vec![tool_call, final_response]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let mut config = AgentConfig::new(test_model(), "You are helpful");
        config.context_strategy = ContextStrategy::Rlm;

        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, _) = mpsc::unbounded_channel();
        let (_, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "rlm-tools-test".into(),
            initial_messages: vec![Message::user("uppercase hello")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();
        assert_eq!(result.len(), 3); // tool_call + tool_result + final
    }

    #[tokio::test]
    async fn agent_loop_semantic_graph_strategy() {
        let response = Message::assistant("Hello from graph mode!");
        let provider = Arc::new(MockProvider::new(vec![response]));

        let tools = ToolRegistry::new();
        let mut config = AgentConfig::new(test_model(), "You are helpful");
        config.context_strategy = ContextStrategy::SemanticGraph;

        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, _) = mpsc::unbounded_channel();
        let (_, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "graph-test".into(),
            initial_messages: vec![Message::user("hi")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].text_content(), "Hello from graph mode!");
    }

    // ─── Structural Failure Detector Tests ──────────────────────────────

    #[test]
    fn structural_detector_catches_empty() {
        // Empty response with no tool calls = structural failure
        let msg = Message::new(Role::Assistant, vec![ContentBlock::text("")]);
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert_eq!(result, Some("empty_response".into()));

        // Whitespace-only also caught
        let msg = Message::new(Role::Assistant, vec![ContentBlock::text("   \n  ")]);
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert_eq!(result, Some("empty_response".into()));

        // Completely empty content vec
        let msg = Message::new(Role::Assistant, vec![]);
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert_eq!(result, Some("empty_response".into()));
    }

    #[test]
    fn structural_detector_catches_json_as_text() {
        // Model writes JSON tool invocation as text instead of structured tool call
        let json_text = r#"{"name": "read", "arguments": {"path": "/src/main.rs"}}"#;
        let msg = Message::new(Role::Assistant, vec![ContentBlock::text(json_text)]);
        let result = StructuralFailureDetector::detect(&msg, 1, true);
        assert_eq!(result, Some("json_as_text".into()));

        // Array of tool calls as text
        let json_array = r#"[{"name": "ls", "arguments": {"path": "."}}]"#;
        let msg = Message::new(Role::Assistant, vec![ContentBlock::text(json_array)]);
        let result = StructuralFailureDetector::detect(&msg, 1, true);
        assert_eq!(result, Some("json_as_text".into()));
    }

    #[test]
    fn structural_detector_catches_no_tools_turn_0() {
        // Turn 0, tools available, but model just talks without acting
        let msg = Message::assistant(
            "I'll analyze the codebase and help you with that. Let me start by understanding the project structure.",
        );
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert_eq!(result, Some("no_tools_turn_0".into()));

        // Turn 1 should NOT trigger this heuristic (only turn 0)
        let result = StructuralFailureDetector::detect(&msg, 1, true);
        assert_ne!(result, Some("no_tools_turn_0".into()));
    }

    #[test]
    fn structural_detector_catches_degenerate_short() {
        // Very short non-actionable response on turn 0 (<= 20 chars, below no_tools_turn_0 threshold)
        let msg = Message::assistant("OK");
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert_eq!(result, Some("degenerate_short".into()));

        let msg = Message::assistant("Sure thing.");
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert_eq!(result, Some("degenerate_short".into()));

        // Longer short text (>20 chars) gets caught by no_tools_turn_0 first — still detected
        let msg = Message::assistant("Sure, I'll help with that.");
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert!(result.is_some()); // caught by no_tools_turn_0

        // On later turns, short text is a valid completion — NOT a failure
        let msg = Message::assistant("Done. Tests pass.");
        let result = StructuralFailureDetector::detect(&msg, 1, true);
        assert!(result.is_none());
    }

    #[test]
    fn structural_detector_passes_valid_tool_call() {
        // Valid tool call — no structural failure
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("Let me read the file"),
                ContentBlock::tool_call("tc1", "read", json!({"path": "/src/main.rs"})),
            ],
        );
        let result = StructuralFailureDetector::detect(&msg, 0, true);
        assert!(result.is_none());
    }

    #[test]
    fn structural_detector_passes_text_only_no_tools_available() {
        // Text-only response when no tools available = perfectly valid
        let msg = Message::assistant("The answer is 42. The codebase uses Axum 0.7.");
        let result = StructuralFailureDetector::detect(&msg, 0, false);
        assert!(result.is_none());

        // Short text with no tools available is also fine
        let msg = Message::assistant("Done.");
        let result = StructuralFailureDetector::detect(&msg, 1, false);
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn agent_loop_retries_structural_failure() {
        // MockProvider returns empty first, then valid tool call
        let empty_response = Message::new(Role::Assistant, vec![ContentBlock::text("")]);
        let valid_response = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "uppercase",
                json!({"text": "hello"}),
            )],
        );
        let final_response = Message::assistant("Done: HELLO");

        let provider = Arc::new(MockProvider::new(vec![
            empty_response,
            valid_response,
            final_response,
        ]));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let config = AgentConfig::new(test_model(), "You are helpful");
        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let (_, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "retry-test".into(),
            initial_messages: vec![Message::user("uppercase hello")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();

        // Should succeed: tool_call + tool_result + final = 3
        assert_eq!(result.len(), 3);

        // Should have emitted a StructuralRetry event
        let mut events = Vec::new();
        while let Ok(event) = event_rx.try_recv() {
            events.push(event);
        }
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEvent::StructuralRetry { .. })));
    }

    #[tokio::test]
    async fn agent_loop_exhausts_retries() {
        // MockProvider returns 3 empty responses (exceeds MAX_STRUCTURAL_RETRIES=2)
        // After exhausting retries, proceeds with the broken response
        let responses: Vec<Message> = (0..3)
            .map(|_| Message::new(Role::Assistant, vec![ContentBlock::text("")]))
            .collect();

        let provider = Arc::new(MockProvider::new(responses));

        let mut tools = ToolRegistry::new();
        tools.register(Box::new(UppercaseTool));

        let config = AgentConfig::new(test_model(), "You are helpful");
        let mut agent = AgentLoop::new(provider, tools, config);

        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let (_, steering_rx) = mpsc::unbounded_channel();

        let options = RunOptions {
            session_id: "exhaust-test".into(),
            initial_messages: vec![Message::user("do something")],
        };

        let result = agent.run(options, event_tx, steering_rx).await.unwrap();

        // Should have 1 message (the broken one after retries exhausted)
        // The empty message has no tool calls so loop exits
        assert_eq!(result.len(), 1);

        // Should have emitted exactly MAX_STRUCTURAL_RETRIES retry events
        let mut retry_count = 0;
        while let Ok(event) = event_rx.try_recv() {
            if matches!(event, AgentEvent::StructuralRetry { .. }) {
                retry_count += 1;
            }
        }
        assert_eq!(retry_count, MAX_STRUCTURAL_RETRIES);
    }
}
