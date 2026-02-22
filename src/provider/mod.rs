mod anthropic;
pub mod balanced;
mod ollama;
mod openai;
mod openai_codex;
mod registry;
pub mod remap;
mod traits;

pub use anthropic::AnthropicProvider;
pub use balanced::BalancedProvider;
pub use ollama::OllamaProvider;
pub use openai::OpenAIProvider;
pub use openai_codex::{gpt_5_3_codex, OpenAICodexProvider};
pub use registry::ProviderRegistry;
pub use remap::ToolRemap;
pub use traits::*;

use std::time::Duration;

/// Default timeout for LLM API requests (connection + full response).
///
/// LLM streaming responses can take 60-120s for long generations.
/// 5 minutes covers worst-case while preventing infinite hangs
/// from stalled connections.
const LLM_REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Default connection timeout — fail fast if the server is unreachable.
const LLM_CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

/// Build a reqwest client with appropriate timeouts for LLM API calls.
fn llm_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(LLM_CONNECT_TIMEOUT)
        .timeout(LLM_REQUEST_TIMEOUT)
        .build()
        .expect("Failed to build HTTP client")
}
