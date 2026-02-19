use std::collections::HashMap;
use std::sync::Arc;

use crate::types::ProviderKind;

use super::traits::Provider;

/// Registry of available providers
pub struct ProviderRegistry {
    providers: HashMap<ProviderKind, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn register(&mut self, provider: Arc<dyn Provider>) {
        self.providers.insert(provider.kind(), provider);
    }

    pub fn get(&self, kind: &ProviderKind) -> Option<Arc<dyn Provider>> {
        self.providers.get(kind).cloned()
    }

    pub fn has(&self, kind: &ProviderKind) -> bool {
        self.providers.contains_key(kind)
    }

    pub fn providers(&self) -> Vec<ProviderKind> {
        self.providers.keys().cloned().collect()
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::AnthropicProvider;
    use crate::provider::OpenAIProvider;

    #[test]
    fn registry_register_and_get() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(AnthropicProvider::new()));
        registry.register(Arc::new(OpenAIProvider::new()));

        assert!(registry.has(&ProviderKind::Anthropic));
        assert!(registry.has(&ProviderKind::OpenAI));
        assert!(!registry.has(&ProviderKind::Gemini));

        let provider = registry.get(&ProviderKind::Anthropic).unwrap();
        assert_eq!(provider.kind(), ProviderKind::Anthropic);
    }

    #[test]
    fn registry_lists_providers() {
        let mut registry = ProviderRegistry::new();
        registry.register(Arc::new(AnthropicProvider::new()));

        let providers = registry.providers();
        assert_eq!(providers.len(), 1);
        assert!(providers.contains(&ProviderKind::Anthropic));
    }

    #[test]
    fn registry_empty() {
        let registry = ProviderRegistry::new();
        assert!(registry.get(&ProviderKind::Anthropic).is_none());
        assert!(registry.providers().is_empty());
    }
}
