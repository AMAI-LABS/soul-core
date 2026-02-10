use std::sync::Arc;
use tokio::sync::mpsc;

use crate::context::{ContextConfig, ContextManager};
use crate::error::{SoulError, SoulResult};
use crate::hook::{BeforeAgentStartContext, BeforeToolCallContext, HookAction, HookPipeline};
use crate::provider::Provider;
use crate::tool::{ToolOutput, ToolRegistry};
use crate::types::*;

/// The steerable agent loop — core execution engine
pub struct AgentLoop {
    provider: Arc<dyn Provider>,
    tools: ToolRegistry,
    config: AgentConfig,
    hooks: HookPipeline,
    context_manager: ContextManager,
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

        Self {
            provider,
            tools,
            config,
            hooks: HookPipeline::new(),
            context_manager: ContextManager::new(context_config),
        }
    }

    pub fn with_hooks(mut self, hooks: HookPipeline) -> Self {
        self.hooks = hooks;
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

        while has_more_tool_calls && turn < max_turns {
            // Check for steering messages
            while let Ok(steering_msg) = steering_rx.try_recv() {
                messages.push(steering_msg.clone());
                new_messages.push(steering_msg);
            }

            // Check if compaction needed
            let system_tokens = system_prompt.len() / 4;
            if self
                .context_manager
                .needs_compaction(&messages, system_tokens)
            {
                let _ = event_tx.send(AgentEvent::CompactionStart {
                    messages_before: messages.len(),
                    tokens_before: ContextManager::estimate_tokens(&messages),
                });

                match self
                    .context_manager
                    .compact(&mut messages, None, system_tokens)
                {
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

            let _ = event_tx.send(AgentEvent::TurnStart { turn });

            // Stream LLM response
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

            let assistant_msg = self
                .provider
                .stream(
                    &messages,
                    &system_prompt,
                    &tool_defs,
                    &self.config.model,
                    &AuthProfile::new(self.config.model.provider.clone(), ""),
                    delta_tx,
                )
                .await?;

            delta_forwarder.await.ok();
            self.context_manager.reset_circuit_breaker();

            let _ = event_tx.send(AgentEvent::MessageEnd {
                message: assistant_msg.clone(),
            });

            messages.push(assistant_msg.clone());
            new_messages.push(assistant_msg.clone());

            // Extract and execute tool calls
            let tool_calls: Vec<&ContentBlock> = assistant_msg.tool_calls();
            has_more_tool_calls = !tool_calls.is_empty();

            if has_more_tool_calls {
                for tc in &tool_calls {
                    if let ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } = tc
                    {
                        // Run before_tool_call hook
                        let hook_ctx = BeforeToolCallContext {
                            tool_name: name.clone(),
                            tool_call_id: id.clone(),
                            arguments: arguments.clone(),
                            session_id: options.session_id.clone(),
                        };

                        let hook_result = self.hooks.run_before_tool_call(hook_ctx).await?;
                        let (tool_name, tool_args) = match hook_result {
                            HookAction::Continue(ctx) => (ctx.tool_name, ctx.arguments),
                            HookAction::Cancel(reason) => {
                                let err_msg = Message::tool_result(
                                    id.clone(),
                                    format!("Blocked: {reason}"),
                                    true,
                                );
                                messages.push(err_msg.clone());
                                new_messages.push(err_msg);
                                continue;
                            }
                        };

                        let _ = event_tx.send(AgentEvent::ToolExecutionStart {
                            tool_call_id: id.clone(),
                            tool_name: tool_name.clone(),
                        });

                        let result = if let Some(tool) = self.tools.get(&tool_name) {
                            let (partial_tx, mut partial_rx) = mpsc::unbounded_channel();
                            let event_tx_c = event_tx.clone();
                            let tc_id = id.clone();
                            let partial_forwarder = tokio::spawn(async move {
                                while let Some(partial) = partial_rx.recv().await {
                                    let _ = event_tx_c.send(AgentEvent::ToolExecutionUpdate {
                                        tool_call_id: tc_id.clone(),
                                        partial_result: partial,
                                    });
                                }
                            });

                            let output = tool.execute(id, tool_args, Some(partial_tx)).await;
                            partial_forwarder.await.ok();

                            match output {
                                Ok(output) => output,
                                Err(e) => ToolOutput::error(format!("Tool error: {e}")),
                            }
                        } else {
                            ToolOutput::error(format!("Unknown tool: {tool_name}"))
                        };

                        // Apply persist hooks
                        let (content, is_error) = self.hooks.transform_tool_result(
                            &tool_name,
                            result.content.clone(),
                            result.is_error,
                        );

                        let result_block =
                            ContentBlock::tool_result(id.clone(), &content, is_error);
                        let _ = event_tx.send(AgentEvent::ToolExecutionEnd {
                            tool_call_id: id.clone(),
                            result: result_block.clone(),
                        });

                        let result_msg = Message::tool_result(id.clone(), content, is_error);
                        messages.push(result_msg.clone());
                        new_messages.push(result_msg);

                        // Check for steering between tool calls
                        if let Ok(steering_msg) = steering_rx.try_recv() {
                            messages.push(steering_msg.clone());
                            new_messages.push(steering_msg);
                            // Skip remaining tool calls
                            break;
                        }
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
}
