use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::mpsc;

use soul_core::agent::{AgentLoop, RunOptions};
use soul_core::context::{ContextConfig, ContextManager};
use soul_core::error::SoulResult;
use soul_core::hook::*;
use soul_core::memory::MemoryStore;
use soul_core::provider::{Provider, ProbeResult};
use soul_core::session::{Session, SessionStore};
use soul_core::subagent::{SubagentRole, SubagentSpawner};
use soul_core::tool::{Tool, ToolOutput, ToolRegistry};
use soul_core::types::*;

// ─── Mock Provider ──────────────────────────────────────────────────────────

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
            return Err(soul_core::error::SoulError::Provider(
                "No more responses".into(),
            ));
        }
        let msg = responses.remove(0);
        for block in &msg.content {
            if let ContentBlock::Text { text } = block {
                let _ = event_tx.send(StreamDelta::TextDelta {
                    text: text.clone(),
                });
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
    ) -> SoulResult<ProbeResult> {
        Ok(ProbeResult {
            healthy: true,
            rate_limit_remaining: Some(1.0),
            rate_limit_utilization: Some(0.0),
        })
    }
}

// ─── Test Tools ─────────────────────────────────────────────────────────────

struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".into(),
            description: "Read a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: serde_json::Value,
        partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .unwrap_or("/unknown");

        // Simulate partial streaming
        if let Some(tx) = partial_tx {
            let _ = tx.send("Reading...".into());
            let _ = tx.send("Done.".into());
        }

        Ok(ToolOutput::success(format!("Contents of {path}: hello world")))
    }
}

struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".into(),
            description: "Write a file".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or("?");
        let content = arguments.get("content").and_then(|v| v.as_str()).unwrap_or("");
        Ok(ToolOutput::success(format!(
            "Wrote {} bytes to {path}",
            content.len()
        )))
    }
}

fn test_model() -> ModelInfo {
    ModelInfo {
        id: "mock-model".into(),
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

// ─── Integration Tests ──────────────────────────────────────────────────────

#[tokio::test]
async fn full_agent_loop_with_tools() {
    // Simulate: user asks to read a file, agent reads it, then responds
    let responses = vec![
        // Turn 1: tool call
        Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "read_file",
                json!({"path": "/tmp/test.txt"}),
            )],
        ),
        // Turn 2: final response
        Message::assistant("The file contains: hello world"),
    ];

    let provider = Arc::new(MockProvider::new(responses));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ReadFileTool));

    let config = AgentConfig::new(test_model(), "You are a helpful assistant");
    let mut agent = AgentLoop::new(provider, tools, config);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (_, steering_rx) = mpsc::unbounded_channel();

    let options = RunOptions {
        session_id: "integration-test-1".into(),
        initial_messages: vec![Message::user("Read /tmp/test.txt")],
    };

    let result = agent.run(options, event_tx, steering_rx).await.unwrap();

    // Should have: tool_call_msg, tool_result, final_response
    assert_eq!(result.len(), 3);
    assert!(result[2].text_content().contains("hello world"));

    // Collect and verify events
    let mut events = Vec::new();
    while let Ok(event) = event_rx.try_recv() {
        events.push(event);
    }

    // Must have agent start/end, turn start/end, tool execution
    assert!(events.iter().any(|e| matches!(e, AgentEvent::AgentStart { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::AgentEnd { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::TurnStart { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolExecutionStart { tool_name, .. } if tool_name == "read_file")));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. })));
    assert!(events.iter().any(|e| matches!(e, AgentEvent::ToolExecutionUpdate { .. })));
}

#[tokio::test]
async fn agent_loop_multi_tool_calls() {
    let responses = vec![
        // Turn 1: multiple tool calls
        Message::new(
            Role::Assistant,
            vec![
                ContentBlock::tool_call("tc1", "read_file", json!({"path": "/a.txt"})),
                ContentBlock::tool_call("tc2", "write_file", json!({"path": "/b.txt", "content": "new"})),
            ],
        ),
        // Turn 2: final
        Message::assistant("Done! Read a.txt and wrote b.txt"),
    ];

    let provider = Arc::new(MockProvider::new(responses));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ReadFileTool));
    tools.register(Box::new(WriteTool));

    let config = AgentConfig::new(test_model(), "You are helpful");
    let mut agent = AgentLoop::new(provider, tools, config);

    let (event_tx, _) = mpsc::unbounded_channel();
    let (_, steering_rx) = mpsc::unbounded_channel();

    let options = RunOptions {
        session_id: "multi-tool-test".into(),
        initial_messages: vec![Message::user("read a.txt and write b.txt")],
    };

    let result = agent.run(options, event_tx, steering_rx).await.unwrap();

    // tool_call_msg + tool_result_1 + tool_result_2 + final = 4
    assert_eq!(result.len(), 4);
}

#[tokio::test]
async fn agent_with_hooks() {
    let responses = vec![
        Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call("tc1", "read_file", json!({"path": "/test"}))],
        ),
        Message::assistant("File read successfully"),
    ];

    let provider = Arc::new(MockProvider::new(responses));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ReadFileTool));

    let config = AgentConfig::new(test_model(), "Base prompt");

    // Add hooks
    struct SystemInjector;
    #[async_trait]
    impl ModifyingHook for SystemInjector {
        fn name(&self) -> &str { "injector" }
        async fn before_agent_start(
            &self,
            mut ctx: BeforeAgentStartContext,
        ) -> SoulResult<HookAction<BeforeAgentStartContext>> {
            ctx.system_prompt.push_str("\n\nInjected by hook.");
            Ok(HookAction::Continue(ctx))
        }
    }

    struct LoggingVoidHook {
        called: Arc<std::sync::atomic::AtomicBool>,
    }
    #[async_trait]
    impl VoidHook for LoggingVoidHook {
        fn name(&self) -> &str { "logger" }
        async fn on_agent_end(&self, _messages: &[Message]) {
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    let hook_called = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mut hooks = HookPipeline::new();
    hooks.add_modifying(Arc::new(SystemInjector));
    hooks.add_void(Arc::new(LoggingVoidHook { called: hook_called.clone() }));

    let mut agent = AgentLoop::new(provider, tools, config).with_hooks(hooks);

    let (event_tx, _) = mpsc::unbounded_channel();
    let (_, steering_rx) = mpsc::unbounded_channel();

    let options = RunOptions {
        session_id: "hook-test".into(),
        initial_messages: vec![Message::user("test hooks")],
    };

    let result = agent.run(options, event_tx, steering_rx).await.unwrap();
    assert!(!result.is_empty());
    assert!(hook_called.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn session_persistence_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(dir.path());

    let mut session = Session::new().with_lane("test");
    let msg1 = Message::user("hello");
    let msg2 = Message::assistant("hi there");
    let msg3 = Message::new(
        Role::Assistant,
        vec![ContentBlock::tool_call("tc1", "read", json!({"path": "/x"}))],
    );
    let msg4 = Message::tool_result("tc1", "file contents", false);

    session.append(msg1.clone());
    session.append(msg2.clone());
    session.append(msg3.clone());
    session.append(msg4.clone());

    // Persist
    for msg in &session.messages {
        store.append_message(&session.id, msg).await.unwrap();
    }
    store.save_session(&session).await.unwrap();

    // Load back
    let loaded_messages = store.load_messages(&session.id).await.unwrap();
    assert_eq!(loaded_messages.len(), 4);
    assert_eq!(loaded_messages[0].role, Role::User);
    assert_eq!(loaded_messages[1].role, Role::Assistant);
    assert!(loaded_messages[2].has_tool_calls());
    assert_eq!(loaded_messages[3].role, Role::Tool);

    // Verify index
    let sessions = store.list_sessions().await.unwrap();
    assert_eq!(sessions.len(), 1);
    assert_eq!(sessions[0].message_count, 4);
    assert_eq!(sessions[0].lane, Some("test".to_string()));
}

#[tokio::test]
async fn memory_hierarchy_integration() {
    let dir = tempfile::tempdir().unwrap();
    let store = MemoryStore::new(dir.path());

    // Write memory
    store.write_main("# Key Patterns\n\n- Use TDD\n- Axum 0.7 patterns\n").await.unwrap();
    store.write_topic("debugging", "# Debug Notes\n\n- Check logs first\n").await.unwrap();
    store.write_topic("architecture", "# Architecture\n\n- Microservices\n").await.unwrap();

    // Build prompt section
    let section = store.build_prompt_section().await.unwrap();
    assert!(section.contains("Key Patterns"));
    assert!(section.contains("debugging"));
    assert!(section.contains("architecture"));

    // Topics list
    let topics = store.list_topics().await.unwrap();
    assert_eq!(topics.len(), 2);

    // Read back
    let main = store.read_main().await.unwrap().unwrap();
    assert!(main.contains("TDD"));

    let debug = store.read_topic("debugging").await.unwrap().unwrap();
    assert!(debug.contains("logs first"));

    // Delete topic
    store.delete_topic("debugging").await.unwrap();
    let topics = store.list_topics().await.unwrap();
    assert_eq!(topics.len(), 1);
}

#[tokio::test]
async fn context_compaction_integration() {
    let config = ContextConfig {
        max_tokens: 500,
        compaction_threshold: 0.6,
        safety_margin: 1.2,
        min_preserved_messages: 2,
        reserve_tokens_floor: 50,
    };
    let mut mgr = ContextManager::new(config);

    // Build up a large conversation
    let mut messages: Vec<Message> = Vec::new();
    for i in 0..20 {
        messages.push(Message::user(format!("Question {i}: {}", "x".repeat(50))));
        messages.push(Message::assistant(format!("Answer {i}: {}", "y".repeat(50))));
    }
    // Add some tool results
    for i in 0..5 {
        messages.push(Message::tool_result(
            format!("tc_{i}"),
            "z".repeat(300),
            false,
        ));
    }

    assert!(mgr.needs_compaction(&messages, 100));

    // Structured state
    let mut state = StructuredState::new();
    state.add("Build soul-core", Some("Building soul-core".into()));
    state.set_status(0, ItemStatus::InProgress);
    state.add("Write tests", Some("Writing tests".into()));

    let result = mgr
        .compact(&mut messages, Some(&state), 100)
        .unwrap();

    assert!(result.messages_after < result.messages_before);
    assert!(result.tokens_after < result.tokens_before);

    // Structured state should be in the first message
    let first_text = messages[0].text_content();
    assert!(first_text.contains("Build soul-core"));
    assert!(first_text.contains("in_progress"));
}

#[tokio::test]
async fn compaction_circuit_breaker_integration() {
    let config = ContextConfig {
        max_tokens: 100,
        compaction_threshold: 0.5,
        min_preserved_messages: 1,
        ..Default::default()
    };
    let mut mgr = ContextManager::new(config);

    let mut messages = vec![Message::user("x".repeat(200))];

    // First compaction succeeds
    let result = mgr.compact(&mut messages, None, 0);
    assert!(result.is_ok());

    // Second attempt fails (circuit breaker)
    let mut messages2 = vec![Message::user("x".repeat(200))];
    let result = mgr.compact(&mut messages2, None, 0);
    assert!(result.is_err());

    // Reset and try again
    mgr.reset_circuit_breaker();
    let mut messages3 = vec![Message::user("x".repeat(200))];
    let result = mgr.compact(&mut messages3, None, 0);
    assert!(result.is_ok());
}

#[tokio::test]
async fn structured_state_survives_compaction() {
    let config = ContextConfig {
        max_tokens: 200,
        compaction_threshold: 0.5,
        min_preserved_messages: 2,
        reserve_tokens_floor: 20,
        ..Default::default()
    };
    let mut mgr = ContextManager::new(config);

    // Many messages
    let mut messages: Vec<Message> = (0..30)
        .map(|i| Message::user(format!("Message {i}: {}", "content ".repeat(10))))
        .collect();

    let mut state = StructuredState::new();
    state.add("Task A", None);
    state.set_status(0, ItemStatus::Completed);
    state.add("Task B", None);
    state.set_status(1, ItemStatus::InProgress);
    state.add("Task C", None);

    let result = mgr.compact(&mut messages, Some(&state), 0).unwrap();

    // State survived
    let has_state = messages.iter().any(|m| {
        let text = m.text_content();
        text.contains("Task A") && text.contains("Task B") && text.contains("Task C")
    });
    assert!(has_state, "Structured state must survive compaction");
    assert!(result.messages_after < result.messages_before);
}

#[tokio::test]
async fn subagent_spawner_no_provider() {
    let spawner = SubagentSpawner::new();
    let config = soul_core::subagent::SubagentConfig {
        name: "test".into(),
        model: test_model(),
        system_prompt: "test".into(),
        max_turns: 1,
        tools: vec![],
    };
    let auth = AuthProfile::new(ProviderKind::Custom("mock".into()), "key");

    let result = spawner
        .spawn_stateless(&config, vec![Message::user("test")], &auth)
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn event_stream_completeness() {
    let responses = vec![
        Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call("tc1", "read_file", json!({"path": "/x"}))],
        ),
        Message::assistant("Done"),
    ];

    let provider = Arc::new(MockProvider::new(responses));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(ReadFileTool));

    let config = AgentConfig::new(test_model(), "test");
    let mut agent = AgentLoop::new(provider, tools, config);

    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (_, steering_rx) = mpsc::unbounded_channel();

    let options = RunOptions {
        session_id: "event-test".into(),
        initial_messages: vec![Message::user("test")],
    };

    agent.run(options, event_tx, steering_rx).await.unwrap();

    let mut events = Vec::new();
    while let Ok(event) = event_rx.try_recv() {
        events.push(event);
    }

    // Verify event ordering
    let event_types: Vec<String> = events
        .iter()
        .map(|e| match e {
            AgentEvent::AgentStart { .. } => "agent_start",
            AgentEvent::AgentEnd { .. } => "agent_end",
            AgentEvent::TurnStart { .. } => "turn_start",
            AgentEvent::TurnEnd { .. } => "turn_end",
            AgentEvent::MessageStart { .. } => "message_start",
            AgentEvent::MessageDelta { .. } => "message_delta",
            AgentEvent::MessageEnd { .. } => "message_end",
            AgentEvent::ToolExecutionStart { .. } => "tool_exec_start",
            AgentEvent::ToolExecutionUpdate { .. } => "tool_exec_update",
            AgentEvent::ToolExecutionEnd { .. } => "tool_exec_end",
            AgentEvent::CompactionStart { .. } => "compact_start",
            AgentEvent::CompactionEnd { .. } => "compact_end",
            AgentEvent::Error { .. } => "error",
        })
        .map(|s| s.to_string())
        .collect();

    // Agent start must be first
    assert_eq!(event_types[0], "agent_start");
    // Agent end must be last
    assert_eq!(event_types[event_types.len() - 1], "agent_end");

    // Must contain tool execution
    assert!(event_types.contains(&"tool_exec_start".to_string()));
    assert!(event_types.contains(&"tool_exec_end".to_string()));
}

#[tokio::test]
async fn hook_blocks_dangerous_tool() {
    struct DangerBlocker;

    #[async_trait]
    impl ModifyingHook for DangerBlocker {
        fn name(&self) -> &str { "danger_blocker" }
        async fn before_tool_call(
            &self,
            ctx: BeforeToolCallContext,
        ) -> SoulResult<HookAction<BeforeToolCallContext>> {
            // Block write operations
            if ctx.tool_name == "write_file" {
                return Ok(HookAction::Cancel("Write operations blocked by policy".into()));
            }
            Ok(HookAction::Continue(ctx))
        }
    }

    let responses = vec![
        Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "write_file",
                json!({"path": "/etc/passwd", "content": "hacked"}),
            )],
        ),
        Message::assistant("Write was blocked"),
    ];

    let provider = Arc::new(MockProvider::new(responses));
    let mut tools = ToolRegistry::new();
    tools.register(Box::new(WriteTool));

    let config = AgentConfig::new(test_model(), "test");

    let mut hooks = HookPipeline::new();
    hooks.add_modifying(Arc::new(DangerBlocker));

    let mut agent = AgentLoop::new(provider, tools, config).with_hooks(hooks);

    let (event_tx, _) = mpsc::unbounded_channel();
    let (_, steering_rx) = mpsc::unbounded_channel();

    let options = RunOptions {
        session_id: "block-test".into(),
        initial_messages: vec![Message::user("write to /etc/passwd")],
    };

    let result = agent.run(options, event_tx, steering_rx).await.unwrap();

    // Should have tool call + blocked result + final
    assert_eq!(result.len(), 3);

    // The tool result should be an error about blocking
    let blocked_msg = &result[1];
    assert_eq!(blocked_msg.role, Role::Tool);
    if let ContentBlock::ToolResult { content, is_error, .. } = &blocked_msg.content[0] {
        assert!(content.contains("Blocked"));
        assert!(*is_error);
    }
}

#[tokio::test]
async fn persist_hook_transforms_results() {
    struct RedactHook;
    impl PersistHook for RedactHook {
        fn name(&self) -> &str { "redact" }
        fn transform_tool_result(
            &self,
            _tool_name: &str,
            content: String,
            is_error: bool,
        ) -> (String, bool) {
            let redacted = content.replace("password123", "[REDACTED]");
            (redacted, is_error)
        }
    }

    let responses = vec![
        Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call("tc1", "read_file", json!({"path": "/secrets"}))],
        ),
        Message::assistant("Got the secrets"),
    ];

    let provider = Arc::new(MockProvider::new(responses));

    // Custom tool that returns sensitive data
    struct SecretTool;
    #[async_trait]
    impl Tool for SecretTool {
        fn name(&self) -> &str { "read_file" }
        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "read_file".into(),
                description: "Read".into(),
                input_schema: json!({"type": "object"}),
            }
        }
        async fn execute(
            &self,
            _call_id: &str,
            _args: serde_json::Value,
            _tx: Option<mpsc::UnboundedSender<String>>,
        ) -> SoulResult<ToolOutput> {
            Ok(ToolOutput::success("The password is password123"))
        }
    }

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(SecretTool));

    let config = AgentConfig::new(test_model(), "test");

    let mut hooks = HookPipeline::new();
    hooks.add_persist(Arc::new(RedactHook));

    let mut agent = AgentLoop::new(provider, tools, config).with_hooks(hooks);

    let (event_tx, _) = mpsc::unbounded_channel();
    let (_, steering_rx) = mpsc::unbounded_channel();

    let options = RunOptions {
        session_id: "redact-test".into(),
        initial_messages: vec![Message::user("read secrets")],
    };

    let result = agent.run(options, event_tx, steering_rx).await.unwrap();

    // The tool result in the messages should have redacted content
    let tool_result_msg = &result[1];
    if let ContentBlock::ToolResult { content, .. } = &tool_result_msg.content[0] {
        assert!(content.contains("[REDACTED]"));
        assert!(!content.contains("password123"));
    }
}
