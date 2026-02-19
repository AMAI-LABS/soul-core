//! Virtual Executor — platform-agnostic command execution.
//!
//! Provides a [`VirtualExecutor`] trait that decouples process spawning from the
//! underlying OS. Ship with [`NativeExecutor`] (behind `native` feature) for real
//! subprocess execution and [`NoopExecutor`] for WASM / testing.

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::SoulResult;

/// Output from a virtual command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl ExecOutput {
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// Platform-agnostic command execution trait.
///
/// Implementations run commands in whatever environment they target:
/// real OS subprocesses, sandboxed WASM, or test stubs.
pub trait VirtualExecutor: Send + Sync {
    /// Execute a shell command string (e.g. `sh -c "echo hello"`).
    fn exec_shell<'a>(
        &'a self,
        command: &'a str,
        timeout_secs: u64,
        cwd: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = SoulResult<ExecOutput>> + Send + 'a>>;
}

// ─── NoopExecutor ──────────────────────────────────────────────────────────

/// Executor that always returns an error — for WASM and test environments.
pub struct NoopExecutor;

impl VirtualExecutor for NoopExecutor {
    fn exec_shell<'a>(
        &'a self,
        _command: &'a str,
        _timeout_secs: u64,
        _cwd: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = SoulResult<ExecOutput>> + Send + 'a>> {
        Box::pin(async {
            Ok(ExecOutput {
                stdout: String::new(),
                stderr: "Shell execution not available in this environment".to_string(),
                exit_code: 1,
            })
        })
    }
}

// ─── MockExecutor for testing ──────────────────────────────────────────────

/// Test executor with canned responses.
pub struct MockExecutor {
    responses: std::sync::Mutex<Vec<ExecOutput>>,
}

impl MockExecutor {
    pub fn new(responses: Vec<ExecOutput>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }

    /// Create a mock that always succeeds with the given stdout.
    pub fn always_ok(stdout: impl Into<String>) -> Self {
        let out = stdout.into();
        // Return a large vec so it doesn't run out
        Self {
            responses: std::sync::Mutex::new(vec![
                ExecOutput {
                    stdout: out.clone(),
                    stderr: String::new(),
                    exit_code: 0,
                };
                100
            ]),
        }
    }
}

impl VirtualExecutor for MockExecutor {
    fn exec_shell<'a>(
        &'a self,
        _command: &'a str,
        _timeout_secs: u64,
        _cwd: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = SoulResult<ExecOutput>> + Send + 'a>> {
        Box::pin(async {
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Ok(ExecOutput {
                    stdout: String::new(),
                    stderr: "No more mock responses".to_string(),
                    exit_code: 1,
                })
            } else {
                Ok(responses.remove(0))
            }
        })
    }
}

// ─── NativeExecutor (behind `native` feature) ─────────────────────────────

#[cfg(feature = "native")]
mod native;
#[cfg(feature = "native")]
pub use native::NativeExecutor;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_executor_returns_error() {
        let exec = NoopExecutor;
        let result = exec.exec_shell("echo hello", 30, None).await.unwrap();
        assert_eq!(result.exit_code, 1);
        assert!(result.stderr.contains("not available"));
    }

    #[tokio::test]
    async fn mock_executor_returns_canned() {
        let exec = MockExecutor::new(vec![ExecOutput {
            stdout: "hello\n".into(),
            stderr: String::new(),
            exit_code: 0,
        }]);
        let result = exec.exec_shell("echo hello", 30, None).await.unwrap();
        assert_eq!(result.stdout, "hello\n");
        assert!(result.success());
    }

    #[tokio::test]
    async fn mock_executor_drains() {
        let exec = MockExecutor::new(vec![
            ExecOutput {
                stdout: "first".into(),
                stderr: String::new(),
                exit_code: 0,
            },
            ExecOutput {
                stdout: "second".into(),
                stderr: String::new(),
                exit_code: 0,
            },
        ]);
        let r1 = exec.exec_shell("cmd1", 30, None).await.unwrap();
        assert_eq!(r1.stdout, "first");
        let r2 = exec.exec_shell("cmd2", 30, None).await.unwrap();
        assert_eq!(r2.stdout, "second");
        let r3 = exec.exec_shell("cmd3", 30, None).await.unwrap();
        assert_eq!(r3.exit_code, 1); // exhausted
    }

    #[tokio::test]
    async fn mock_always_ok() {
        let exec = MockExecutor::always_ok("output");
        let r1 = exec.exec_shell("a", 30, None).await.unwrap();
        let r2 = exec.exec_shell("b", 30, None).await.unwrap();
        assert_eq!(r1.stdout, "output");
        assert_eq!(r2.stdout, "output");
        assert!(r1.success());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_executor_echo() {
        let exec = NativeExecutor::new();
        let result = exec.exec_shell("echo hello", 30, None).await.unwrap();
        assert_eq!(result.stdout.trim(), "hello");
        assert!(result.success());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_executor_exit_code() {
        let exec = NativeExecutor::new();
        let result = exec.exec_shell("exit 42", 30, None).await.unwrap();
        assert_eq!(result.exit_code, 42);
        assert!(!result.success());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_executor_stderr() {
        let exec = NativeExecutor::new();
        let result = exec.exec_shell("echo err >&2", 30, None).await.unwrap();
        assert_eq!(result.stderr.trim(), "err");
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_executor_timeout() {
        let exec = NativeExecutor::new();
        let result = exec.exec_shell("sleep 10", 1, None).await;
        assert!(result.is_err());
    }

    #[cfg(all(feature = "native", not(target_os = "windows")))]
    #[tokio::test]
    async fn native_executor_cwd() {
        let dir = tempfile::tempdir().unwrap();
        let exec = NativeExecutor::new();
        let result = exec
            .exec_shell("pwd", 30, Some(dir.path().to_str().unwrap()))
            .await
            .unwrap();
        assert!(result.stdout.trim().contains(dir.path().to_str().unwrap()));
    }

    #[test]
    fn exec_output_success_check() {
        let ok = ExecOutput {
            stdout: "ok".into(),
            stderr: String::new(),
            exit_code: 0,
        };
        assert!(ok.success());

        let fail = ExecOutput {
            stdout: String::new(),
            stderr: "fail".into(),
            exit_code: 1,
        };
        assert!(!fail.success());
    }
}
