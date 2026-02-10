use async_trait::async_trait;
use std::sync::Arc;

use crate::error::SoulResult;
use crate::types::*;

/// Context passed to hooks before agent starts
#[derive(Debug, Clone)]
pub struct BeforeAgentStartContext {
    pub system_prompt: String,
    pub tools: Vec<ToolDefinition>,
    pub messages: Vec<Message>,
    pub session_id: String,
}

/// Context passed to hooks before tool execution
#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    pub tool_name: String,
    pub tool_call_id: String,
    pub arguments: serde_json::Value,
    pub session_id: String,
}

/// Result of a modifying hook — can alter or cancel
#[derive(Debug, Clone)]
pub enum HookAction<T> {
    /// Continue with modified value
    Continue(T),
    /// Cancel the operation
    Cancel(String),
}

/// Hook trait for modifying hooks (sequential, can alter state)
#[async_trait]
pub trait ModifyingHook: Send + Sync {
    fn name(&self) -> &str;

    /// Called before agent starts — can modify system prompt, tools, messages
    async fn before_agent_start(
        &self,
        ctx: BeforeAgentStartContext,
    ) -> SoulResult<HookAction<BeforeAgentStartContext>> {
        Ok(HookAction::Continue(ctx))
    }

    /// Called before tool execution — can modify or block
    async fn before_tool_call(
        &self,
        ctx: BeforeToolCallContext,
    ) -> SoulResult<HookAction<BeforeToolCallContext>> {
        Ok(HookAction::Continue(ctx))
    }
}

/// Hook trait for void hooks (parallel, fire-and-forget)
#[async_trait]
pub trait VoidHook: Send + Sync {
    fn name(&self) -> &str;

    async fn on_agent_end(&self, _messages: &[Message]) {}
    async fn on_tool_result(&self, _tool_name: &str, _result: &str, _is_error: bool) {}
    async fn on_compaction_start(&self, _messages_count: usize, _tokens: usize) {}
    async fn on_compaction_end(&self, _messages_count: usize, _tokens: usize) {}
    async fn on_error(&self, _error: &str) {}
}

/// Synchronous hook for tool result persistence (hot path)
pub trait PersistHook: Send + Sync {
    fn name(&self) -> &str;

    /// Transform tool result before persistence — MUST be synchronous (hot path)
    fn transform_tool_result(
        &self,
        tool_name: &str,
        content: String,
        is_error: bool,
    ) -> (String, bool) {
        (content, is_error)
    }
}

/// Pipeline that orchestrates hook execution
pub struct HookPipeline {
    modifying: Vec<Arc<dyn ModifyingHook>>,
    void_hooks: Vec<Arc<dyn VoidHook>>,
    persist: Vec<Arc<dyn PersistHook>>,
}

impl HookPipeline {
    pub fn new() -> Self {
        Self {
            modifying: Vec::new(),
            void_hooks: Vec::new(),
            persist: Vec::new(),
        }
    }

    pub fn add_modifying(&mut self, hook: Arc<dyn ModifyingHook>) {
        self.modifying.push(hook);
    }

    pub fn add_void(&mut self, hook: Arc<dyn VoidHook>) {
        self.void_hooks.push(hook);
    }

    pub fn add_persist(&mut self, hook: Arc<dyn PersistHook>) {
        self.persist.push(hook);
    }

    /// Run modifying hooks sequentially — each sees the output of the previous
    pub async fn run_before_agent_start(
        &self,
        mut ctx: BeforeAgentStartContext,
    ) -> SoulResult<HookAction<BeforeAgentStartContext>> {
        for hook in &self.modifying {
            match hook.before_agent_start(ctx.clone()).await? {
                HookAction::Continue(new_ctx) => ctx = new_ctx,
                HookAction::Cancel(reason) => return Ok(HookAction::Cancel(reason)),
            }
        }
        Ok(HookAction::Continue(ctx))
    }

    /// Run modifying hooks for tool calls
    pub async fn run_before_tool_call(
        &self,
        mut ctx: BeforeToolCallContext,
    ) -> SoulResult<HookAction<BeforeToolCallContext>> {
        for hook in &self.modifying {
            match hook.before_tool_call(ctx.clone()).await? {
                HookAction::Continue(new_ctx) => ctx = new_ctx,
                HookAction::Cancel(reason) => return Ok(HookAction::Cancel(reason)),
            }
        }
        Ok(HookAction::Continue(ctx))
    }

    /// Run void hooks in parallel (fire-and-forget)
    pub async fn fire_agent_end(&self, messages: &[Message]) {
        let futures: Vec<_> = self
            .void_hooks
            .iter()
            .map(|h| {
                let hook = h.clone();
                let msgs = messages.to_vec();
                tokio::spawn(async move {
                    hook.on_agent_end(&msgs).await;
                })
            })
            .collect();
        for f in futures {
            let _ = f.await;
        }
    }

    /// Run void hooks for errors
    pub async fn fire_error(&self, error: &str) {
        let futures: Vec<_> = self
            .void_hooks
            .iter()
            .map(|h| {
                let hook = h.clone();
                let err = error.to_string();
                tokio::spawn(async move {
                    hook.on_error(&err).await;
                })
            })
            .collect();
        for f in futures {
            let _ = f.await;
        }
    }

    /// Run persist hooks synchronously (hot path)
    pub fn transform_tool_result(
        &self,
        tool_name: &str,
        mut content: String,
        mut is_error: bool,
    ) -> (String, bool) {
        for hook in &self.persist {
            let (new_content, new_error) =
                hook.transform_tool_result(tool_name, content, is_error);
            content = new_content;
            is_error = new_error;
        }
        (content, is_error)
    }
}

impl Default for HookPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct InjectSystemHook {
        suffix: String,
    }

    #[async_trait]
    impl ModifyingHook for InjectSystemHook {
        fn name(&self) -> &str {
            "inject_system"
        }

        async fn before_agent_start(
            &self,
            mut ctx: BeforeAgentStartContext,
        ) -> SoulResult<HookAction<BeforeAgentStartContext>> {
            ctx.system_prompt.push_str(&self.suffix);
            Ok(HookAction::Continue(ctx))
        }
    }

    struct BlockBashHook;

    #[async_trait]
    impl ModifyingHook for BlockBashHook {
        fn name(&self) -> &str {
            "block_bash"
        }

        async fn before_tool_call(
            &self,
            ctx: BeforeToolCallContext,
        ) -> SoulResult<HookAction<BeforeToolCallContext>> {
            if ctx.tool_name == "bash" {
                Ok(HookAction::Cancel("bash blocked by policy".into()))
            } else {
                Ok(HookAction::Continue(ctx))
            }
        }
    }

    struct CounterHook {
        counter: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl VoidHook for CounterHook {
        fn name(&self) -> &str {
            "counter"
        }

        async fn on_agent_end(&self, _messages: &[Message]) {
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    struct TruncatePersistHook {
        max_len: usize,
    }

    impl PersistHook for TruncatePersistHook {
        fn name(&self) -> &str {
            "truncate"
        }

        fn transform_tool_result(
            &self,
            _tool_name: &str,
            content: String,
            is_error: bool,
        ) -> (String, bool) {
            if content.len() > self.max_len {
                let truncated = format!("{}... (truncated)", &content[..self.max_len]);
                (truncated, is_error)
            } else {
                (content, is_error)
            }
        }
    }

    #[tokio::test]
    async fn modifying_hooks_chain() {
        let mut pipeline = HookPipeline::new();
        pipeline.add_modifying(Arc::new(InjectSystemHook {
            suffix: "\nHook 1 was here".into(),
        }));
        pipeline.add_modifying(Arc::new(InjectSystemHook {
            suffix: "\nHook 2 was here".into(),
        }));

        let ctx = BeforeAgentStartContext {
            system_prompt: "Base prompt".into(),
            tools: vec![],
            messages: vec![],
            session_id: "s1".into(),
        };

        let result = pipeline.run_before_agent_start(ctx).await.unwrap();
        if let HookAction::Continue(ctx) = result {
            assert!(ctx.system_prompt.contains("Hook 1 was here"));
            assert!(ctx.system_prompt.contains("Hook 2 was here"));
        } else {
            panic!("Expected Continue");
        }
    }

    #[tokio::test]
    async fn modifying_hook_cancels() {
        let mut pipeline = HookPipeline::new();
        pipeline.add_modifying(Arc::new(BlockBashHook));

        let ctx = BeforeToolCallContext {
            tool_name: "bash".into(),
            tool_call_id: "tc1".into(),
            arguments: serde_json::json!({"command": "rm -rf /"}),
            session_id: "s1".into(),
        };

        let result = pipeline.run_before_tool_call(ctx).await.unwrap();
        assert!(matches!(result, HookAction::Cancel(_)));
    }

    #[tokio::test]
    async fn modifying_hook_allows() {
        let mut pipeline = HookPipeline::new();
        pipeline.add_modifying(Arc::new(BlockBashHook));

        let ctx = BeforeToolCallContext {
            tool_name: "read".into(),
            tool_call_id: "tc1".into(),
            arguments: serde_json::json!({"path": "/tmp/test.txt"}),
            session_id: "s1".into(),
        };

        let result = pipeline.run_before_tool_call(ctx).await.unwrap();
        assert!(matches!(result, HookAction::Continue(_)));
    }

    #[tokio::test]
    async fn void_hooks_fire_parallel() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let mut pipeline = HookPipeline::new();
        pipeline.add_void(Arc::new(CounterHook {
            counter: counter.clone(),
        }));
        pipeline.add_void(Arc::new(CounterHook {
            counter: counter.clone(),
        }));

        pipeline.fire_agent_end(&[]).await;
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn persist_hook_transforms() {
        let mut pipeline = HookPipeline::new();
        pipeline.add_persist(Arc::new(TruncatePersistHook { max_len: 10 }));

        let (content, is_error) =
            pipeline.transform_tool_result("read", "short".into(), false);
        assert_eq!(content, "short");
        assert!(!is_error);

        let (content, _) = pipeline.transform_tool_result(
            "read",
            "this is a very long tool result that should be truncated".into(),
            false,
        );
        assert!(content.contains("... (truncated)"));
        assert!(content.len() < 60);
    }

    #[test]
    fn empty_pipeline_passthrough() {
        let pipeline = HookPipeline::new();
        let (content, is_error) =
            pipeline.transform_tool_result("test", "hello".into(), false);
        assert_eq!(content, "hello");
        assert!(!is_error);
    }
}
