use std::sync::Arc;
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::provider::Provider;
use crate::types::*;

/// Subagent configuration
#[derive(Debug, Clone)]
pub struct SubagentConfig {
    pub name: String,
    pub model: ModelInfo,
    pub system_prompt: String,
    pub max_turns: usize,
    pub tools: Vec<ToolDefinition>,
}

/// Subagent role — specialization for cheap model tiers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubagentRole {
    /// Extract metadata from tool outputs (file paths, status)
    MetadataExtractor,
    /// Validate command safety before execution
    SafetyValidator,
    /// Summarize content for compaction
    Summarizer,
    /// General purpose subagent
    General,
}

/// Result from a subagent execution
#[derive(Debug, Clone)]
pub struct SubagentResult {
    pub role: SubagentRole,
    pub content: String,
    pub model_used: String,
    pub usage: Option<TokenUsage>,
}

/// Subagent spawner — manages dual-model architecture
pub struct SubagentSpawner {
    providers: Vec<(ProviderKind, Arc<dyn Provider>)>,
}

impl SubagentSpawner {
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
        }
    }

    pub fn add_provider(&mut self, kind: ProviderKind, provider: Arc<dyn Provider>) {
        self.providers.push((kind, provider));
    }

    fn get_provider(&self, kind: &ProviderKind) -> Option<Arc<dyn Provider>> {
        self.providers
            .iter()
            .find(|(k, _)| k == kind)
            .map(|(_, p)| p.clone())
    }

    /// Spawn a stateless subagent — single turn, no tools, cheap model
    pub async fn spawn_stateless(
        &self,
        config: &SubagentConfig,
        messages: Vec<Message>,
        auth: &AuthProfile,
    ) -> SoulResult<SubagentResult> {
        let provider = self
            .get_provider(&config.model.provider)
            .ok_or_else(|| {
                crate::error::SoulError::Provider(format!(
                    "No provider for {}",
                    config.model.provider
                ))
            })?;

        let (tx, _rx) = mpsc::unbounded_channel();

        let response = provider
            .stream(
                &messages,
                &config.system_prompt,
                &config.tools,
                &config.model,
                auth,
                tx,
            )
            .await?;

        Ok(SubagentResult {
            role: SubagentRole::General,
            content: response.text_content(),
            model_used: config.model.id.clone(),
            usage: response.usage,
        })
    }

    /// Extract metadata from tool output (dual-model pattern)
    pub async fn extract_metadata(
        &self,
        config: &SubagentConfig,
        tool_name: &str,
        tool_output: &str,
        auth: &AuthProfile,
    ) -> SoulResult<SubagentResult> {
        let prompt = format!(
            "Analyze this {} tool output and extract key metadata (file paths, status changes, important values). Be concise.\n\nOutput:\n{}",
            tool_name, tool_output
        );

        let messages = vec![Message::user(prompt)];
        let mut result = self.spawn_stateless(config, messages, auth).await?;
        result.role = SubagentRole::MetadataExtractor;
        Ok(result)
    }

    /// Validate command safety (injection detection pattern)
    pub async fn validate_command(
        &self,
        config: &SubagentConfig,
        command: &str,
        auth: &AuthProfile,
    ) -> SoulResult<SubagentResult> {
        let prompt = format!(
            "Analyze this shell command and output ONLY the command prefix (e.g., 'git commit', 'npm install') or 'command_injection_detected' if it contains injection patterns.\n\nCommand: {}",
            command
        );

        let messages = vec![Message::user(prompt)];
        let mut result = self.spawn_stateless(config, messages, auth).await?;
        result.role = SubagentRole::SafetyValidator;
        Ok(result)
    }

    /// Summarize messages for compaction
    pub async fn summarize(
        &self,
        config: &SubagentConfig,
        messages: &[Message],
        auth: &AuthProfile,
    ) -> SoulResult<SubagentResult> {
        let messages_text: Vec<String> = messages
            .iter()
            .map(|m| format!("[{}] {}", m.role.to_string(), m.text_content()))
            .collect();

        let prompt = format!(
            "Summarize this conversation, preserving: 1) Key decisions made, 2) Files modified, 3) Current task state, 4) Important context.\n\n{}",
            messages_text.join("\n\n")
        );

        let msgs = vec![Message::user(prompt)];
        let mut result = self.spawn_stateless(config, msgs, auth).await?;
        result.role = SubagentRole::Summarizer;
        Ok(result)
    }
}

impl Default for SubagentSpawner {
    fn default() -> Self {
        Self::new()
    }
}

// Display for Role (needed by summarize)
impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_config_creates() {
        let config = SubagentConfig {
            name: "metadata_extractor".into(),
            model: ModelInfo {
                id: "claude-haiku-4-5-20251001".into(),
                provider: ProviderKind::Anthropic,
                context_window: 200_000,
                max_output_tokens: 4096,
                supports_thinking: false,
                supports_tools: false,
                supports_images: false,
                cost_per_input_token: 0.0000008,
                cost_per_output_token: 0.000004,
            },
            system_prompt: "Extract metadata from tool outputs".into(),
            max_turns: 1,
            tools: vec![],
        };
        assert_eq!(config.name, "metadata_extractor");
        assert!(config.tools.is_empty());
    }

    #[test]
    fn subagent_result_creates() {
        let result = SubagentResult {
            role: SubagentRole::MetadataExtractor,
            content: "Files: /tmp/test.txt, /tmp/other.txt".into(),
            model_used: "claude-haiku-4-5-20251001".into(),
            usage: Some(TokenUsage::new(100, 20)),
        };
        assert_eq!(result.role, SubagentRole::MetadataExtractor);
        assert!(result.content.contains("test.txt"));
    }

    #[test]
    fn subagent_spawner_empty() {
        let spawner = SubagentSpawner::new();
        assert!(spawner.get_provider(&ProviderKind::Anthropic).is_none());
    }

    #[test]
    fn subagent_spawner_add_provider() {
        use crate::provider::AnthropicProvider;

        let mut spawner = SubagentSpawner::new();
        spawner.add_provider(ProviderKind::Anthropic, Arc::new(AnthropicProvider::new()));
        assert!(spawner.get_provider(&ProviderKind::Anthropic).is_some());
        assert!(spawner.get_provider(&ProviderKind::OpenAI).is_none());
    }

    #[test]
    fn role_display() {
        assert_eq!(Role::User.to_string(), "user");
        assert_eq!(Role::Assistant.to_string(), "assistant");
        assert_eq!(Role::System.to_string(), "system");
        assert_eq!(Role::Tool.to_string(), "tool");
    }

    #[test]
    fn subagent_role_eq() {
        assert_eq!(SubagentRole::MetadataExtractor, SubagentRole::MetadataExtractor);
        assert_ne!(SubagentRole::MetadataExtractor, SubagentRole::SafetyValidator);
    }
}
