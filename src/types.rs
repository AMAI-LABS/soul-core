use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ─── Message Types ──────────────────────────────────────────────────────────

/// Role in a conversation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A content block within a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    ToolResult {
        tool_call_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {
        text: String,
    },
    Image {
        media_type: String,
        data: String,
    },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        ContentBlock::Text { text: s.into() }
    }

    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        args: serde_json::Value,
    ) -> Self {
        ContentBlock::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: args,
        }
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        ContentBlock::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            is_error,
        }
    }

    pub fn thinking(s: impl Into<String>) -> Self {
        ContentBlock::Thinking { text: s.into() }
    }

    /// Estimate token count for this content block.
    ///
    /// Uses a word+punctuation heuristic that's more accurate than chars/4 for code and JSON:
    /// - Words contribute ~1.3 tokens each (subword tokenization)
    /// - Punctuation/symbols are often individual tokens
    /// Falls back to chars/4 minimum to avoid undercount on dense text.
    pub fn estimate_tokens(&self) -> usize {
        let text = match self {
            ContentBlock::Text { text } => text.as_str(),
            ContentBlock::ToolCall {
                name, arguments, ..
            } => {
                // Tool calls have name + JSON args — JSON is token-dense
                let args_str = arguments.to_string();
                let name_tokens = name.len().div_ceil(4) + 2; // name + framing
                let args_tokens = estimate_tokens_heuristic(&args_str);
                return name_tokens + args_tokens;
            }
            ContentBlock::ToolResult { content, .. } => content.as_str(),
            ContentBlock::Thinking { text } => text.as_str(),
            ContentBlock::Image { data, .. } => {
                // Images: base64 encoded, ~4 bytes per 3 raw bytes,
                // but image tokens are model-specific. Use rough estimate.
                return data.len() / 16;
            }
        };
        estimate_tokens_heuristic(text)
    }
}

/// Estimate token count from text using a word+punctuation heuristic.
///
/// LLM tokenizers (BPE) split text into subwords. Rough rules:
/// - Common English words → 1 token
/// - Longer/uncommon words → 1-3 tokens (avg ~1.3)
/// - Punctuation, brackets, operators → 1 token each
/// - Whitespace is usually merged with adjacent tokens
///
/// This heuristic counts words and standalone punctuation, then applies
/// a 1.3x multiplier. Returns max(word_estimate, chars/4) to avoid
/// undercounting dense text like minified JSON or base64.
pub(crate) fn estimate_tokens_heuristic_pub(text: &str) -> usize {
    estimate_tokens_heuristic(text)
}

fn estimate_tokens_heuristic(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }

    let mut words = 0usize;
    let mut punctuation = 0usize;
    let mut in_word = false;

    for ch in text.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            if !in_word {
                words += 1;
                in_word = true;
            }
        } else {
            in_word = false;
            if !ch.is_whitespace() {
                punctuation += 1; // brackets, operators, quotes, etc.
            }
        }
    }

    let word_estimate = (words as f64 * 1.3) as usize + punctuation;
    let char_estimate = text.len().div_ceil(4);

    // Take the higher estimate — word-based is better for prose/code,
    // char-based is better for dense encoded data
    word_estimate.max(char_estimate)
}

/// A message in a conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: Vec<ContentBlock>,
    pub timestamp: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl Message {
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            role,
            content,
            timestamp: Utc::now(),
            model: None,
            usage: None,
        }
    }

    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::User, vec![ContentBlock::text(text)])
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::Assistant, vec![ContentBlock::text(text)])
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::System, vec![ContentBlock::text(text)])
    }

    pub fn tool_result(
        tool_call_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self::new(
            Role::Tool,
            vec![ContentBlock::tool_result(tool_call_id, content, is_error)],
        )
    }

    /// Extract tool calls from this message
    pub fn tool_calls(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|c| matches!(c, ContentBlock::ToolCall { .. }))
            .collect()
    }

    /// Check if this message contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.content
            .iter()
            .any(|c| matches!(c, ContentBlock::ToolCall { .. }))
    }

    /// Get text content concatenated
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Estimate total tokens for this message.
    ///
    /// Uses actual LLM-reported token count when available (assistant messages from providers),
    /// falls back to heuristic estimation for user messages, tool results, and messages
    /// without usage data.
    pub fn estimate_tokens(&self) -> usize {
        // Use actual output_tokens from LLM when available
        if let Some(ref usage) = self.usage {
            if self.role == Role::Assistant && usage.output_tokens > 0 {
                return usage.output_tokens;
            }
        }
        let content_tokens: usize = self.content.iter().map(|c| c.estimate_tokens()).sum();
        content_tokens + 4 // role + framing overhead
    }
}

// ─── Token Usage ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    #[serde(default)]
    pub cache_read_tokens: usize,
    #[serde(default)]
    pub cache_write_tokens: usize,
}

impl TokenUsage {
    pub fn new(input: usize, output: usize) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }
    }

    pub fn total(&self) -> usize {
        self.input_tokens + self.output_tokens
    }
}

// ─── Model Info ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub provider: ProviderKind,
    pub context_window: usize,
    #[serde(default)]
    pub max_output_tokens: usize,
    #[serde(default)]
    pub supports_thinking: bool,
    #[serde(default)]
    pub supports_tools: bool,
    #[serde(default)]
    pub supports_images: bool,
    #[serde(default)]
    pub cost_per_input_token: f64,
    #[serde(default)]
    pub cost_per_output_token: f64,
}

/// Known LLM providers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderKind {
    Anthropic,
    OpenAI,
    Ollama,
    Gemini,
    Bedrock,
    Local,
    Custom(String),
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderKind::Anthropic => write!(f, "anthropic"),
            ProviderKind::OpenAI => write!(f, "openai"),
            ProviderKind::Ollama => write!(f, "ollama"),
            ProviderKind::Gemini => write!(f, "gemini"),
            ProviderKind::Bedrock => write!(f, "bedrock"),
            ProviderKind::Local => write!(f, "local"),
            ProviderKind::Custom(s) => write!(f, "{s}"),
        }
    }
}

// ─── Streaming Events ────────────────────────────────────────────────────────

/// Events emitted during agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    AgentStart {
        session_id: String,
    },
    AgentEnd {
        session_id: String,
        messages: Vec<Message>,
    },
    TurnStart {
        turn: usize,
    },
    TurnEnd {
        turn: usize,
        message: Message,
    },
    MessageStart {
        message_id: String,
    },
    MessageDelta {
        message_id: String,
        delta: StreamDelta,
    },
    MessageEnd {
        message: Message,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        partial_result: String,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        result: ContentBlock,
    },
    CompactionStart {
        messages_before: usize,
        tokens_before: usize,
    },
    CompactionEnd {
        messages_after: usize,
        tokens_after: usize,
    },
    /// A structural failure was detected in the LLM response and will be retried.
    ///
    /// Structural failures are protocol-level breakages — not "low quality" but fundamentally
    /// broken outputs from weaker/free-tier models (empty responses, JSON-as-text instead of
    /// tool calls, degenerate short replies). These are intermittent and recoverable via retry.
    StructuralRetry {
        turn: usize,
        attempt: usize,
        failure_kind: String,
        model_response_preview: String,
    },
    Error {
        message: String,
    },
    Cost(crate::cost::CostEvent),
    PermissionCheck {
        tool_name: String,
        decision: String,
    },
    McpServerConnected {
        server_name: String,
        tool_count: usize,
    },
    SkillLoaded {
        skill_name: String,
    },
    SubagentStart {
        parent_session: String,
        child_session: String,
        purpose: String,
        max_turns: usize,
    },
    SubagentEnd {
        child_session: String,
        purpose: String,
        turns_used: usize,
        result_preview: String,
    },
    PlanExecutionStart {
        parent_session: String,
        task_count: usize,
    },
    PlanTaskStart {
        parent_session: String,
        task_id: String,
        description: String,
    },
    PlanTaskEnd {
        task_id: String,
        status: String,
        result_preview: String,
    },
    PlanExecutionEnd {
        completed: usize,
        failed: usize,
        skipped: usize,
        total: usize,
    },
}

/// Delta updates during streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamDelta {
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        text: String,
    },
    ToolCallDelta {
        id: String,
        name: String,
        arguments_delta: String,
    },
}

// ─── Tool Definition ─────────────────────────────────────────────────────────

/// Schema for a tool's input parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// ─── Context Strategy ────────────────────────────────────────────────────────

/// Strategy for managing conversation context as it grows.
///
/// - `Classic`: Traditional compaction — truncate/offload old messages when context fills.
///   Information is irreversibly lost. Simple, low overhead.
///
/// - `Rlm`: Context-as-document — full conversation history is serialized as an external
///   document. The LLM navigates it via DSL REPL (QUERY, CHUNK, MAP). Nothing is ever lost.
///   Higher token cost per turn (REPL iterations), but infinite memory.
///
/// - `SemanticGraph`: Knowledge graph — every message becomes a graph node with typed edges.
///   Symlink compression replaces old content with 6-char hash references. Intelligent retrieval
///   via keyword + vector + graph neighbor search. Nothing is ever lost.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextStrategy {
    /// Traditional truncation-based compaction (default)
    #[default]
    Classic,
    /// RLM: context-as-document with DSL REPL navigation
    Rlm,
    /// Semantic graph: knowledge graph with symlink compression
    SemanticGraph,
}

// ─── Agent Configuration ─────────────────────────────────────────────────────

/// Configuration for an agent loop run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model: ModelInfo,
    pub system_prompt: String,
    #[serde(default)]
    pub max_turns: Option<usize>,
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: f64,
    #[serde(default = "default_safety_margin")]
    pub token_safety_margin: f64,
    #[serde(default)]
    pub fallback_models: Vec<ModelInfo>,
    /// Context management strategy — how to handle growing conversation history
    #[serde(default)]
    pub context_strategy: ContextStrategy,
}

fn default_compaction_threshold() -> f64 {
    0.75
}

fn default_safety_margin() -> f64 {
    1.2
}

impl AgentConfig {
    pub fn new(model: ModelInfo, system_prompt: impl Into<String>) -> Self {
        Self {
            model,
            system_prompt: system_prompt.into(),
            max_turns: None,
            compaction_threshold: default_compaction_threshold(),
            token_safety_margin: default_safety_margin(),
            fallback_models: Vec::new(),
            context_strategy: ContextStrategy::default(),
        }
    }

    /// Effective max tokens considering safety margin
    pub fn effective_max_tokens(&self) -> usize {
        (self.model.context_window as f64 / self.token_safety_margin) as usize
    }

    /// Token count that triggers compaction
    pub fn compaction_trigger_tokens(&self) -> usize {
        (self.model.context_window as f64 * self.compaction_threshold) as usize
    }
}

// ─── Auth Profile ────────────────────────────────────────────────────────────

/// Authentication profile for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthProfile {
    pub id: String,
    pub provider: ProviderKind,
    pub api_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub org_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cooldown_until: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

impl AuthProfile {
    pub fn new(provider: ProviderKind, api_key: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            provider,
            api_key: api_key.into(),
            base_url: None,
            org_id: None,
            cooldown_until: None,
            failure_reason: None,
        }
    }

    pub fn is_in_cooldown(&self) -> bool {
        self.cooldown_until.map(|t| Utc::now() < t).unwrap_or(false)
    }
}

// ─── Structured State (TodoWrite pattern) ────────────────────────────────────

/// Structured state that survives compaction (inspired by Claude Code's TodoWrite)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredState {
    pub items: Vec<StateItem>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateItem {
    pub content: String,
    pub status: ItemStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_form: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ItemStatus {
    Pending,
    InProgress,
    Completed,
}

impl StructuredState {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn add(&mut self, content: impl Into<String>, active_form: Option<String>) {
        self.items.push(StateItem {
            content: content.into(),
            status: ItemStatus::Pending,
            active_form,
        });
    }

    pub fn set_status(&mut self, index: usize, status: ItemStatus) {
        if let Some(item) = self.items.get_mut(index) {
            item.status = status;
        }
    }

    pub fn pending_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == ItemStatus::Pending)
            .count()
    }

    pub fn completed_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.status == ItemStatus::Completed)
            .count()
    }
}

impl Default for StructuredState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Message Tests ──────────────────────────────────────────────────

    #[test]
    fn message_user_creates_text() {
        let msg = Message::user("hello world");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text_content(), "hello world");
        assert!(!msg.id.is_empty());
    }

    #[test]
    fn message_assistant_creates_text() {
        let msg = Message::assistant("I can help");
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.text_content(), "I can help");
    }

    #[test]
    fn message_system_creates_text() {
        let msg = Message::system("You are helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.text_content(), "You are helpful");
    }

    #[test]
    fn message_tool_result_creates() {
        let msg = Message::tool_result("call_123", "file contents here", false);
        assert_eq!(msg.role, Role::Tool);
        assert!(!msg.has_tool_calls());
    }

    #[test]
    fn message_with_tool_calls() {
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("Let me read that file"),
                ContentBlock::tool_call("tc_1", "read", serde_json::json!({"path": "/foo.txt"})),
                ContentBlock::tool_call("tc_2", "glob", serde_json::json!({"pattern": "*.rs"})),
            ],
        );
        assert!(msg.has_tool_calls());
        assert_eq!(msg.tool_calls().len(), 2);
        assert_eq!(msg.text_content(), "Let me read that file");
    }

    #[test]
    fn message_without_tool_calls() {
        let msg = Message::assistant("just text");
        assert!(!msg.has_tool_calls());
        assert_eq!(msg.tool_calls().len(), 0);
    }

    // ─── Serialization Tests ────────────────────────────────────────────

    #[test]
    fn message_serializes_roundtrip() {
        let msg = Message::user("test message");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role, msg.role);
        assert_eq!(deserialized.text_content(), msg.text_content());
        assert_eq!(deserialized.id, msg.id);
    }

    #[test]
    fn content_block_serializes_tagged() {
        let block = ContentBlock::text("hello");
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"text""#));

        let block = ContentBlock::tool_call("id1", "bash", serde_json::json!({"cmd": "ls"}));
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains(r#""type":"tool_call""#));
        assert!(json.contains(r#""name":"bash""#));
    }

    #[test]
    fn role_serializes_lowercase() {
        let json = serde_json::to_string(&Role::Assistant).unwrap();
        assert_eq!(json, r#""assistant""#);

        let json = serde_json::to_string(&Role::User).unwrap();
        assert_eq!(json, r#""user""#);
    }

    #[test]
    fn agent_event_serializes_tagged() {
        let event = AgentEvent::AgentStart {
            session_id: "s1".into(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"agent_start""#));
        assert!(json.contains(r#""session_id":"s1""#));
    }

    #[test]
    fn stream_delta_serializes_tagged() {
        let delta = StreamDelta::TextDelta {
            text: "hello".into(),
        };
        let json = serde_json::to_string(&delta).unwrap();
        assert!(json.contains(r#""type":"text_delta""#));
    }

    // ─── Token Estimation Tests ─────────────────────────────────────────

    #[test]
    fn token_estimation_text() {
        let block = ContentBlock::text("hello world!"); // 12 chars → ceil(12/4) = 3
        let tokens = block.estimate_tokens();
        assert_eq!(tokens, 3);
    }

    #[test]
    fn token_estimation_empty() {
        let block = ContentBlock::text(""); // 0 chars → ceil(0/4) = 0
        let tokens = block.estimate_tokens();
        assert_eq!(tokens, 0);
    }

    #[test]
    fn token_estimation_code() {
        // Code has many short tokens: fn, let, =, (), {}, ;
        let code = r#"fn main() { let x = 42; println!("{}", x); }"#;
        let block = ContentBlock::text(code);
        let tokens = block.estimate_tokens();
        // Should be higher than chars/4 due to punctuation-heavy code
        assert!(tokens >= code.len().div_ceil(4));
    }

    #[test]
    fn token_estimation_json() {
        let json = r#"{"name":"test","values":[1,2,3],"nested":{"key":"val"}}"#;
        let block = ContentBlock::text(json);
        let tokens = block.estimate_tokens();
        // JSON is punctuation-dense, should estimate more tokens than chars/4
        assert!(tokens >= json.len().div_ceil(4));
    }

    #[test]
    fn message_token_estimation() {
        let msg = Message::user("hello world!");
        let tokens = msg.estimate_tokens();
        assert!(tokens > 0);
        assert!(tokens >= 4); // at least the text content
    }

    #[test]
    fn message_token_estimation_uses_actual_usage() {
        let mut msg = Message::assistant("This is a test response from the LLM.");
        msg.usage = Some(TokenUsage::new(50, 25));
        // Should use actual output_tokens (25), not heuristic
        assert_eq!(msg.estimate_tokens(), 25);
    }

    #[test]
    fn message_token_estimation_fallback_without_usage() {
        let msg = Message::user("This is a user message without usage data.");
        // No usage data → heuristic
        let tokens = msg.estimate_tokens();
        assert!(tokens > 4); // more than just overhead
    }

    // ─── TokenUsage Tests ───────────────────────────────────────────────

    #[test]
    fn token_usage_total() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn token_usage_serialization() {
        let usage = TokenUsage::new(100, 50);
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: TokenUsage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, usage);
    }

    // ─── ModelInfo Tests ────────────────────────────────────────────────

    #[test]
    fn provider_kind_display() {
        assert_eq!(ProviderKind::Anthropic.to_string(), "anthropic");
        assert_eq!(ProviderKind::OpenAI.to_string(), "openai");
        assert_eq!(ProviderKind::Custom("ollama".into()).to_string(), "ollama");
    }

    #[test]
    fn provider_kind_serializes() {
        let json = serde_json::to_string(&ProviderKind::Anthropic).unwrap();
        assert_eq!(json, r#""anthropic""#);

        let deserialized: ProviderKind = serde_json::from_str(r#""openai""#).unwrap();
        assert_eq!(deserialized, ProviderKind::OpenAI);
    }

    // ─── AgentConfig Tests ──────────────────────────────────────────────

    #[test]
    fn agent_config_defaults() {
        let model = ModelInfo {
            id: "claude-sonnet-4-5-20250929".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: true,
            supports_tools: true,
            supports_images: true,
            cost_per_input_token: 0.000003,
            cost_per_output_token: 0.000015,
        };
        let config = AgentConfig::new(model, "You are helpful");

        assert_eq!(config.compaction_threshold, 0.75);
        assert_eq!(config.token_safety_margin, 1.2);
        assert!(config.max_turns.is_none());
        assert_eq!(config.context_strategy, ContextStrategy::Classic);
    }

    #[test]
    fn context_strategy_default_is_classic() {
        assert_eq!(ContextStrategy::default(), ContextStrategy::Classic);
    }

    #[test]
    fn context_strategy_serde_roundtrip() {
        let strategies = vec![
            ContextStrategy::Classic,
            ContextStrategy::Rlm,
            ContextStrategy::SemanticGraph,
        ];
        for s in strategies {
            let json = serde_json::to_string(&s).unwrap();
            let parsed: ContextStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(s, parsed);
        }
    }

    #[test]
    fn context_strategy_serde_names() {
        assert_eq!(
            serde_json::to_string(&ContextStrategy::Rlm).unwrap(),
            "\"rlm\""
        );
        assert_eq!(
            serde_json::to_string(&ContextStrategy::SemanticGraph).unwrap(),
            "\"semantic_graph\""
        );
        assert_eq!(
            serde_json::to_string(&ContextStrategy::Classic).unwrap(),
            "\"classic\""
        );
    }

    #[test]
    fn agent_config_effective_tokens() {
        let model = ModelInfo {
            id: "test".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let config = AgentConfig::new(model, "test");

        // 200_000 / 1.2 = 166_666
        assert_eq!(config.effective_max_tokens(), 166_666);

        // 200_000 * 0.75 = 150_000
        assert_eq!(config.compaction_trigger_tokens(), 150_000);
    }

    // ─── AuthProfile Tests ──────────────────────────────────────────────

    #[test]
    fn auth_profile_not_in_cooldown_by_default() {
        let profile = AuthProfile::new(ProviderKind::Anthropic, "sk-test-123");
        assert!(!profile.is_in_cooldown());
    }

    #[test]
    fn auth_profile_in_cooldown() {
        let mut profile = AuthProfile::new(ProviderKind::Anthropic, "sk-test-123");
        profile.cooldown_until = Some(Utc::now() + chrono::Duration::hours(1));
        assert!(profile.is_in_cooldown());
    }

    #[test]
    fn auth_profile_cooldown_expired() {
        let mut profile = AuthProfile::new(ProviderKind::Anthropic, "sk-test-123");
        profile.cooldown_until = Some(Utc::now() - chrono::Duration::hours(1));
        assert!(!profile.is_in_cooldown());
    }

    // ─── StructuredState Tests ──────────────────────────────────────────

    #[test]
    fn structured_state_lifecycle() {
        let mut state = StructuredState::new();
        assert_eq!(state.items.len(), 0);
        assert_eq!(state.pending_count(), 0);
        assert_eq!(state.completed_count(), 0);

        state.add("Initialize project", Some("Initializing project".into()));
        state.add("Write tests", Some("Writing tests".into()));
        state.add("Implement code", None);

        assert_eq!(state.items.len(), 3);
        assert_eq!(state.pending_count(), 3);
        assert_eq!(state.completed_count(), 0);

        state.set_status(0, ItemStatus::InProgress);
        assert_eq!(state.pending_count(), 2);

        state.set_status(0, ItemStatus::Completed);
        assert_eq!(state.pending_count(), 2);
        assert_eq!(state.completed_count(), 1);
    }

    #[test]
    fn structured_state_serializes() {
        let mut state = StructuredState::new();
        state.add("Task 1", None);
        state.set_status(0, ItemStatus::Completed);

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: StructuredState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, state);
    }

    // ─── ToolDefinition Tests ───────────────────────────────────────────

    #[test]
    fn tool_definition_creates() {
        let tool = ToolDefinition {
            name: "read".into(),
            description: "Read a file".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        };
        assert_eq!(tool.name, "read");
    }

    #[test]
    fn tool_definition_serializes() {
        let tool = ToolDefinition {
            name: "bash".into(),
            description: "Run a command".into(),
            input_schema: serde_json::json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: ToolDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, tool);
    }
}
