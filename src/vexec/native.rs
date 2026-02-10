//! Native OS command executor using `tokio::process`.

use std::future::Future;
use std::pin::Pin;

use crate::error::{SoulError, SoulResult};

use super::{ExecOutput, VirtualExecutor};

/// Executes commands as real OS subprocesses via `sh -c`.
pub struct NativeExecutor;

impl NativeExecutor {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NativeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualExecutor for NativeExecutor {
    fn exec_shell<'a>(
        &'a self,
        command: &'a str,
        timeout_secs: u64,
        cwd: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = SoulResult<ExecOutput>> + Send + 'a>> {
        Box::pin(async move {
            let mut cmd = tokio::process::Command::new("sh");
            cmd.arg("-c").arg(command);

            if let Some(dir) = cwd {
                cmd.current_dir(dir);
            }

            let output =
                tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output())
                    .await
                    .map_err(|_| SoulError::ToolExecution {
                        tool_name: "virtual_executor".into(),
                        message: format!("Command timed out after {timeout_secs}s"),
                    })?
                    .map_err(|e| SoulError::ToolExecution {
                        tool_name: "virtual_executor".into(),
                        message: format!("Failed to execute: {e}"),
                    })?;

            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();

            #[cfg(unix)]
            let exit_code = {
                use std::os::unix::process::ExitStatusExt;
                output
                    .status
                    .code()
                    .unwrap_or_else(|| output.status.signal().map(|s| 128 + s).unwrap_or(1))
            };
            #[cfg(not(unix))]
            let exit_code = output.status.code().unwrap_or(1);

            Ok(ExecOutput {
                stdout,
                stderr,
                exit_code,
            })
        })
    }
}
