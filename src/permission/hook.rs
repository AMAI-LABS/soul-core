use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::SoulResult;
use crate::hook::{BeforeToolCallContext, HookAction, ModifyingHook};

use super::{PermissionDecision, PermissionGate};

/// A modifying hook that enforces permission checks before tool calls.
///
/// Plugs into the existing `HookPipeline` via `pipeline.add_modifying()`.
///
/// - `Allow` → `HookAction::Continue`
/// - `Deny` → `HookAction::Cancel`
/// - `Ask` → calls the ask handler callback; if it returns `true`, continue; otherwise cancel
pub struct PermissionHook {
    gate: Arc<dyn PermissionGate>,
    ask_handler: Option<Arc<dyn AskHandler>>,
}

/// Handler for Ask decisions — returns true to allow, false to deny.
pub trait AskHandler: Send + Sync {
    fn handle<'a>(
        &'a self,
        prompt: &'a str,
        tool_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>>;
}

/// Simple auto-deny handler for when no interactive handler is provided.
struct AutoDenyHandler;

impl AskHandler for AutoDenyHandler {
    fn handle<'a>(
        &'a self,
        _prompt: &'a str,
        _tool_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
        Box::pin(async { false })
    }
}

/// Auto-allow handler for permissive configurations.
pub struct AutoAllowHandler;

impl AskHandler for AutoAllowHandler {
    fn handle<'a>(
        &'a self,
        _prompt: &'a str,
        _tool_name: &'a str,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
        Box::pin(async { true })
    }
}

impl PermissionHook {
    pub fn new(gate: Arc<dyn PermissionGate>) -> Self {
        Self {
            gate,
            ask_handler: None,
        }
    }

    pub fn with_ask_handler(mut self, handler: Arc<dyn AskHandler>) -> Self {
        self.ask_handler = Some(handler);
        self
    }
}

#[async_trait]
impl ModifyingHook for PermissionHook {
    fn name(&self) -> &str {
        "permission"
    }

    async fn before_tool_call(
        &self,
        ctx: BeforeToolCallContext,
    ) -> SoulResult<HookAction<BeforeToolCallContext>> {
        let decision = self.gate.check(&ctx.tool_name, &ctx.arguments).await;

        match decision {
            PermissionDecision::Allow => Ok(HookAction::Continue(ctx)),
            PermissionDecision::Deny { reason } => Ok(HookAction::Cancel(reason)),
            PermissionDecision::Ask {
                prompt, tool_name, ..
            } => {
                let handler = self
                    .ask_handler
                    .as_ref()
                    .map(|h| h.clone())
                    .unwrap_or_else(|| Arc::new(AutoDenyHandler));

                let allowed = handler.handle(&prompt, &tool_name).await;
                if allowed {
                    Ok(HookAction::Continue(ctx))
                } else {
                    Ok(HookAction::Cancel(format!(
                        "Permission denied by user for '{tool_name}'"
                    )))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hook::BeforeToolCallContext;
    use crate::permission::{PermissionDecision, PermissionGate};
    use serde_json::json;

    struct AlwaysAllowGate;
    impl PermissionGate for AlwaysAllowGate {
        fn check<'a>(
            &'a self,
            _tool_name: &'a str,
            _arguments: &'a serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = PermissionDecision> + Send + 'a>> {
            Box::pin(async { PermissionDecision::Allow })
        }
    }

    struct AlwaysDenyGate;
    impl PermissionGate for AlwaysDenyGate {
        fn check<'a>(
            &'a self,
            _tool_name: &'a str,
            _arguments: &'a serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = PermissionDecision> + Send + 'a>> {
            Box::pin(async {
                PermissionDecision::Deny {
                    reason: "always denied".into(),
                }
            })
        }
    }

    struct AlwaysAskGate;
    impl PermissionGate for AlwaysAskGate {
        fn check<'a>(
            &'a self,
            tool_name: &'a str,
            _arguments: &'a serde_json::Value,
        ) -> Pin<Box<dyn Future<Output = PermissionDecision> + Send + 'a>> {
            let name = tool_name.to_string();
            Box::pin(async move {
                PermissionDecision::Ask {
                    prompt: "Allow this?".into(),
                    tool_name: name,
                    risk_level: crate::permission::RiskLevel::Execution,
                }
            })
        }
    }

    fn test_ctx(tool_name: &str) -> BeforeToolCallContext {
        BeforeToolCallContext {
            tool_name: tool_name.into(),
            tool_call_id: "tc1".into(),
            arguments: json!({}),
            session_id: "s1".into(),
        }
    }

    #[tokio::test]
    async fn allow_continues() {
        let hook = PermissionHook::new(Arc::new(AlwaysAllowGate));
        let result = hook.before_tool_call(test_ctx("read")).await.unwrap();
        assert!(matches!(result, HookAction::Continue(_)));
    }

    #[tokio::test]
    async fn deny_cancels() {
        let hook = PermissionHook::new(Arc::new(AlwaysDenyGate));
        let result = hook.before_tool_call(test_ctx("bash")).await.unwrap();
        assert!(matches!(result, HookAction::Cancel(_)));
    }

    #[tokio::test]
    async fn ask_without_handler_denies() {
        let hook = PermissionHook::new(Arc::new(AlwaysAskGate));
        let result = hook.before_tool_call(test_ctx("write")).await.unwrap();
        assert!(matches!(result, HookAction::Cancel(_)));
    }

    #[tokio::test]
    async fn ask_with_allow_handler_continues() {
        let hook = PermissionHook::new(Arc::new(AlwaysAskGate))
            .with_ask_handler(Arc::new(AutoAllowHandler));
        let result = hook.before_tool_call(test_ctx("write")).await.unwrap();
        assert!(matches!(result, HookAction::Continue(_)));
    }

    #[tokio::test]
    async fn ask_with_deny_handler_cancels() {
        let hook = PermissionHook::new(Arc::new(AlwaysAskGate));
        // No handler = auto deny
        let result = hook.before_tool_call(test_ctx("write")).await.unwrap();
        match result {
            HookAction::Cancel(reason) => {
                assert!(reason.contains("Permission denied"));
            }
            _ => panic!("Expected Cancel"),
        }
    }

    #[tokio::test]
    async fn hook_name() {
        let hook = PermissionHook::new(Arc::new(AlwaysAllowGate));
        assert_eq!(hook.name(), "permission");
    }

    #[tokio::test]
    async fn custom_ask_handler() {
        struct ConditionalHandler;
        impl AskHandler for ConditionalHandler {
            fn handle<'a>(
                &'a self,
                _prompt: &'a str,
                tool_name: &'a str,
            ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
                let allow = tool_name == "safe_tool";
                Box::pin(async move { allow })
            }
        }

        let hook = PermissionHook::new(Arc::new(AlwaysAskGate))
            .with_ask_handler(Arc::new(ConditionalHandler));

        let result = hook.before_tool_call(test_ctx("safe_tool")).await.unwrap();
        assert!(matches!(result, HookAction::Continue(_)));

        let result = hook
            .before_tool_call(test_ctx("unsafe_tool"))
            .await
            .unwrap();
        assert!(matches!(result, HookAction::Cancel(_)));
    }

    #[test]
    fn permission_hook_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PermissionHook>();
    }
}
