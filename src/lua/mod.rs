//! Lua sandbox — embedded Lua 5.4 runtime for skill execution and RLM REPL.
//!
//! Provides a sandboxed Lua VM that removes dangerous modules (io, os, debug,
//! require, loadfile, dofile, load) while exposing controlled functions for
//! HTTP, JSON, and context manipulation.

pub mod crypto;
pub mod functions;

use std::sync::{Arc, Mutex};

use mlua::{Lua, MultiValue, Result as LuaResult, Value};

use crate::error::{SoulError, SoulResult};

/// Channel pair for synchronous LLM sub-queries from Lua code.
///
/// When `llm_query(prompt)` is called in Lua, it sends the prompt through `request_tx`
/// and blocks on `response_rx`. The RLM engine fulfills requests asynchronously.
pub struct LlmQueryChannel {
    pub request_tx: std::sync::mpsc::Sender<LlmSubRequest>,
    pub response_rx: Arc<Mutex<std::sync::mpsc::Receiver<String>>>,
}

/// A sub-LLM query request from Lua code.
pub enum LlmSubRequest {
    /// Single query: prompt → response
    Single(String),
    /// Batched queries: vec of prompts → vec of responses (concurrent)
    Batched(Vec<String>),
}

/// A sandboxed Lua 5.4 VM.
pub struct LuaSandbox {
    lua: Lua,
    output_buffer: Arc<Mutex<Vec<String>>>,
    final_answer: Arc<Mutex<Option<String>>>,
}

impl LuaSandbox {
    /// Create a new sandboxed Lua VM.
    ///
    /// Removes dangerous globals: `io`, `os`, `debug`, `require`, `loadfile`,
    /// `dofile`, `load`.
    pub fn new() -> SoulResult<Self> {
        let lua = Lua::new();

        // Remove dangerous modules
        {
            let globals = lua.globals();
            for name in &[
                "io",
                "os",
                "debug",
                "require",
                "loadfile",
                "dofile",
                "load",
            ] {
                globals
                    .set(*name, Value::Nil)
                    .map_err(|e| SoulError::ToolExecution {
                        tool_name: "lua_sandbox".into(),
                        message: format!("Failed to remove {name}: {e}"),
                    })?;
            }
        }

        let output_buffer = Arc::new(Mutex::new(Vec::new()));
        let final_answer = Arc::new(Mutex::new(None));

        // Register print_var
        let buf = output_buffer.clone();
        lua.globals()
            .set(
                "print_var",
                lua.create_function(move |_, val: Value| {
                    let s = lua_value_to_string(&val);
                    buf.lock().unwrap().push(s);
                    Ok(())
                })
                .map_err(|e| SoulError::ToolExecution {
                    tool_name: "lua_sandbox".into(),
                    message: format!("Failed to register print_var: {e}"),
                })?,
            )
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set print_var: {e}"),
            })?;

        // Override `print` to also go to output buffer
        let buf2 = output_buffer.clone();
        lua.globals()
            .set(
                "print",
                lua.create_function(move |_, vals: MultiValue| {
                    let parts: Vec<String> = vals.iter().map(lua_value_to_string).collect();
                    buf2.lock().unwrap().push(parts.join("\t"));
                    Ok(())
                })
                .map_err(|e| SoulError::ToolExecution {
                    tool_name: "lua_sandbox".into(),
                    message: format!("Failed to register print: {e}"),
                })?,
            )
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set print: {e}"),
            })?;

        // Register final_answer
        let fa = final_answer.clone();
        lua.globals()
            .set(
                "final_answer",
                lua.create_function(move |_, val: Value| {
                    let s = lua_value_to_string(&val);
                    *fa.lock().unwrap() = Some(s);
                    Ok(())
                })
                .map_err(|e| SoulError::ToolExecution {
                    tool_name: "lua_sandbox".into(),
                    message: format!("Failed to register final_answer: {e}"),
                })?,
            )
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set final_answer: {e}"),
            })?;

        // Register final_var — return a REPL variable as the answer (for long outputs)
        let fa2 = final_answer.clone();
        lua.globals()
            .set(
                "final_var",
                lua.create_function(move |lua, varname: String| {
                    let val: Value = lua.globals().get(varname.clone()).map_err(|e| {
                        mlua::Error::external(format!("final_var: variable '{varname}' not found: {e}"))
                    })?;
                    let s = lua_value_to_string(&val);
                    *fa2.lock().unwrap() = Some(s);
                    Ok(())
                })
                .map_err(|e| SoulError::ToolExecution {
                    tool_name: "lua_sandbox".into(),
                    message: format!("Failed to register final_var: {e}"),
                })?,
            )
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set final_var: {e}"),
            })?;

        Ok(Self {
            lua,
            output_buffer,
            final_answer,
        })
    }

    /// Get a reference to the inner Lua VM (for registering custom functions).
    pub fn lua(&self) -> &Lua {
        &self.lua
    }

    /// Set a string global variable.
    pub fn set_string(&self, name: &str, value: &str) -> SoulResult<()> {
        self.lua
            .globals()
            .set(name.to_string(), value.to_string())
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set string '{name}': {e}"),
            })
    }

    /// Set a JSON value as a Lua global (recursively converts objects/arrays to tables).
    pub fn set_json(&self, name: &str, value: &serde_json::Value) -> SoulResult<()> {
        let lua_val =
            json_to_lua(&self.lua, value).map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to convert JSON to Lua for '{name}': {e}"),
            })?;
        self.lua
            .globals()
            .set(name.to_string(), lua_val)
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set JSON '{name}': {e}"),
            })
    }

    /// Execute Lua code and return the result as a string.
    pub fn exec(&self, code: &str) -> SoulResult<String> {
        let result: Value = self
            .lua
            .load(code)
            .eval()
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Lua execution error: {e}"),
            })?;

        Ok(lua_value_to_string(&result))
    }

    /// Execute Lua code with a timeout (wall-clock seconds).
    pub async fn exec_with_timeout(&self, code: &str, timeout_secs: u64) -> SoulResult<String> {
        let timeout = std::time::Duration::from_secs(timeout_secs);

        // Lua execution is synchronous, wrap in spawn_blocking to not block the runtime
        let code = code.to_string();
        let lua_code = code.clone();

        // We can't move self into spawn_blocking, so execute directly with tokio timeout
        match tokio::time::timeout(timeout, async { self.exec(&lua_code) }).await {
            Ok(result) => result,
            Err(_) => Err(SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Lua execution timed out after {timeout_secs}s"),
            }),
        }
    }

    /// Take accumulated output buffer contents.
    pub fn take_output(&self) -> Vec<String> {
        std::mem::take(&mut self.output_buffer.lock().unwrap())
    }

    /// Check if final_answer has been called, and take the value.
    pub fn take_final_answer(&self) -> Option<String> {
        self.final_answer.lock().unwrap().take()
    }

    /// Check if final_answer has been set (without consuming it).
    pub fn has_final_answer(&self) -> bool {
        self.final_answer.lock().unwrap().is_some()
    }

    /// Read a Lua global variable and return it as a string.
    ///
    /// Used by `final_var(varname)` to return a REPL variable as the answer
    /// without requiring the LLM to serialize it into `final_answer()`.
    pub fn get_global_string(&self, name: &str) -> SoulResult<Option<String>> {
        let val: Value = self.lua.globals().get(name.to_string()).map_err(|e| {
            SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to read global '{name}': {e}"),
            }
        })?;
        match val {
            Value::Nil => Ok(None),
            _ => Ok(Some(lua_value_to_string(&val))),
        }
    }

    /// Register `llm_query(prompt)` and `llm_query_batched(prompts)` functions.
    ///
    /// These let Lua code make synchronous sub-LLM calls. The functions block
    /// on a channel — the RLM engine fulfills requests asynchronously on the
    /// other end.
    pub fn register_llm_query(&self, channel: LlmQueryChannel) -> SoulResult<()> {
        let req_tx = channel.request_tx.clone();
        let resp_rx = channel.response_rx.clone();

        // llm_query(prompt) → string
        let req_tx_single = req_tx.clone();
        let resp_rx_single = resp_rx.clone();
        self.lua
            .globals()
            .set(
                "llm_query",
                self.lua
                    .create_function(move |_, prompt: String| {
                        req_tx_single
                            .send(LlmSubRequest::Single(prompt))
                            .map_err(|e| {
                                mlua::Error::external(format!("llm_query channel error: {e}"))
                            })?;
                        let response = resp_rx_single
                            .lock()
                            .unwrap()
                            .recv()
                            .map_err(|e| {
                                mlua::Error::external(format!("llm_query response error: {e}"))
                            })?;
                        Ok(response)
                    })
                    .map_err(|e| SoulError::ToolExecution {
                        tool_name: "lua_sandbox".into(),
                        message: format!("Failed to create llm_query: {e}"),
                    })?,
            )
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set llm_query: {e}"),
            })?;

        // llm_query_batched(prompts_table) → table of strings
        self.lua
            .globals()
            .set(
                "llm_query_batched",
                self.lua
                    .create_function(move |lua, prompts: Value| {
                        // Convert Lua table to Vec<String>
                        let table = match prompts {
                            Value::Table(t) => t,
                            _ => {
                                return Err(mlua::Error::external(
                                    "llm_query_batched expects a table of strings",
                                ));
                            }
                        };
                        let mut prompt_vec = Vec::new();
                        for pair in table.sequence_values::<String>() {
                            prompt_vec.push(pair.map_err(|e| {
                                mlua::Error::external(format!(
                                    "llm_query_batched: invalid prompt: {e}"
                                ))
                            })?);
                        }

                        req_tx
                            .send(LlmSubRequest::Batched(prompt_vec.clone()))
                            .map_err(|e| {
                                mlua::Error::external(format!(
                                    "llm_query_batched channel error: {e}"
                                ))
                            })?;

                        // Receive one response per prompt
                        let result_table = lua.create_table()?;
                        for i in 0..prompt_vec.len() {
                            let response =
                                resp_rx.lock().unwrap().recv().map_err(|e| {
                                    mlua::Error::external(format!(
                                        "llm_query_batched response error: {e}"
                                    ))
                                })?;
                            result_table.set(i + 1, response)?;
                        }
                        Ok(Value::Table(result_table))
                    })
                    .map_err(|e| SoulError::ToolExecution {
                        tool_name: "lua_sandbox".into(),
                        message: format!("Failed to create llm_query_batched: {e}"),
                    })?,
            )
            .map_err(|e| SoulError::ToolExecution {
                tool_name: "lua_sandbox".into(),
                message: format!("Failed to set llm_query_batched: {e}"),
            })?;

        Ok(())
    }
}

// ─── JSON ↔ Lua Conversion ──────────────────────────────────────────────────

/// Convert a serde_json::Value to a mlua::Value.
pub fn json_to_lua(lua: &Lua, value: &serde_json::Value) -> LuaResult<Value> {
    match value {
        serde_json::Value::Null => Ok(Value::Nil),
        serde_json::Value::Bool(b) => Ok(Value::Boolean(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Number(f))
            } else {
                Ok(Value::Nil)
            }
        }
        serde_json::Value::String(s) => Ok(Value::String(lua.create_string(s)?)),
        serde_json::Value::Array(arr) => {
            let table = lua.create_table()?;
            for (i, item) in arr.iter().enumerate() {
                table.set(i + 1, json_to_lua(lua, item)?)?; // Lua is 1-indexed
            }
            Ok(Value::Table(table))
        }
        serde_json::Value::Object(obj) => {
            let table = lua.create_table()?;
            for (key, val) in obj {
                table.set(key.as_str(), json_to_lua(lua, val)?)?;
            }
            Ok(Value::Table(table))
        }
    }
}

/// Convert a mlua::Value to a serde_json::Value.
pub fn lua_to_json(value: &Value) -> serde_json::Value {
    match value {
        Value::Nil => serde_json::Value::Null,
        Value::Boolean(b) => serde_json::Value::Bool(*b),
        Value::Integer(i) => serde_json::json!(*i),
        Value::Number(f) => serde_json::json!(*f),
        Value::String(s) => {
            serde_json::Value::String(
                s.to_str().map(|s| s.to_string()).unwrap_or_default(),
            )
        }
        Value::Table(t) => {
            // Check if this is an array-like table (consecutive integer keys starting at 1)
            let len = t.raw_len();
            if len > 0 {
                // Check if all keys 1..=len exist
                let mut is_array = true;
                for i in 1..=len {
                    if t.raw_get::<Value>(i).is_err() {
                        is_array = false;
                        break;
                    }
                }

                if is_array {
                    // Also check if there are any non-integer keys
                    let mut total_pairs = 0;
                    if let Ok(pairs) = t.clone().pairs::<Value, Value>().try_fold(0, |acc, r| {
                        r.map(|_| acc + 1)
                    }) {
                        total_pairs = pairs;
                    }
                    if total_pairs != len as usize {
                        is_array = false;
                    }
                }

                if is_array {
                    let arr: Vec<serde_json::Value> = (1..=len)
                        .filter_map(|i| t.raw_get::<Value>(i).ok().map(|v| lua_to_json(&v)))
                        .collect();
                    return serde_json::Value::Array(arr);
                }
            }

            // Object
            let mut map = serde_json::Map::new();
            if let Ok(pairs) = t.clone().pairs::<Value, Value>().collect::<LuaResult<Vec<_>>>()
            {
                for (k, v) in pairs {
                    let key = lua_value_to_string(&k);
                    map.insert(key, lua_to_json(&v));
                }
            }
            serde_json::Value::Object(map)
        }
        _ => serde_json::Value::Null,
    }
}

/// Convert a Lua value to its string representation.
pub(crate) fn lua_value_to_string(val: &Value) -> String {
    match val {
        Value::Nil => "nil".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        Value::Number(f) => f.to_string(),
        Value::String(s) => s.to_str().map(|s| s.to_string()).unwrap_or_default(),
        Value::Table(t) => {
            // Pretty-print table as JSON
            let json = lua_to_json(&Value::Table(t.clone()));
            serde_json::to_string_pretty(&json).unwrap_or_else(|_| "table: <?>".to_string())
        }
        _ => format!("{val:?}"),
    }
}

/// Extract ` ```lua ``` ` code blocks from text.
pub fn extract_lua_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if !in_block && trimmed.starts_with("```lua") {
            in_block = true;
            current.clear();
        } else if in_block && trimmed == "```" {
            blocks.push(current.clone());
            in_block = false;
        } else if in_block {
            current.push_str(line);
            current.push('\n');
        }
    }

    blocks
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sandbox_creation() {
        let sandbox = LuaSandbox::new().unwrap();
        // Should be able to execute basic Lua
        let result = sandbox.exec("return 2 + 2").unwrap();
        assert_eq!(result, "4");
    }

    #[test]
    fn sandbox_removes_dangerous_modules() {
        let sandbox = LuaSandbox::new().unwrap();

        // io should be nil
        let result = sandbox.exec("return type(io)").unwrap();
        assert_eq!(result, "nil");

        // os should be nil
        let result = sandbox.exec("return type(os)").unwrap();
        assert_eq!(result, "nil");

        // debug should be nil
        let result = sandbox.exec("return type(debug)").unwrap();
        assert_eq!(result, "nil");

        // require should be nil
        let result = sandbox.exec("return type(require)").unwrap();
        assert_eq!(result, "nil");
    }

    #[test]
    fn set_string_variable() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox.set_string("greeting", "hello world").unwrap();
        let result = sandbox.exec("return greeting").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn set_json_object() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox
            .set_json("data", &json!({"name": "test", "count": 42}))
            .unwrap();

        let name = sandbox.exec("return data.name").unwrap();
        assert_eq!(name, "test");

        let count = sandbox.exec("return data.count").unwrap();
        assert_eq!(count, "42");
    }

    #[test]
    fn set_json_array() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox
            .set_json("items", &json!(["a", "b", "c"]))
            .unwrap();

        let first = sandbox.exec("return items[1]").unwrap();
        assert_eq!(first, "a");

        let len = sandbox.exec("return #items").unwrap();
        assert_eq!(len, "3");
    }

    #[test]
    fn set_json_nested() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox
            .set_json(
                "config",
                &json!({"server": {"host": "localhost", "port": 8080}}),
            )
            .unwrap();

        let host = sandbox.exec("return config.server.host").unwrap();
        assert_eq!(host, "localhost");

        let port = sandbox.exec("return config.server.port").unwrap();
        assert_eq!(port, "8080");
    }

    #[test]
    fn print_var_captures_output() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox.exec("print_var('hello')").unwrap();
        sandbox.exec("print_var(42)").unwrap();
        let output = sandbox.take_output();
        assert_eq!(output, vec!["hello", "42"]);
    }

    #[test]
    fn print_captures_output() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox.exec("print('hello', 'world')").unwrap();
        let output = sandbox.take_output();
        assert_eq!(output, vec!["hello\tworld"]);
    }

    #[test]
    fn final_answer_captures() {
        let sandbox = LuaSandbox::new().unwrap();
        assert!(!sandbox.has_final_answer());

        sandbox.exec("final_answer('the answer is 42')").unwrap();
        assert!(sandbox.has_final_answer());

        let answer = sandbox.take_final_answer();
        assert_eq!(answer, Some("the answer is 42".to_string()));
    }

    #[test]
    fn exec_error_returns_soul_error() {
        let sandbox = LuaSandbox::new().unwrap();
        let result = sandbox.exec("return undefined_function()");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn exec_with_timeout_succeeds() {
        let sandbox = LuaSandbox::new().unwrap();
        let result = sandbox.exec_with_timeout("return 'ok'", 5).await.unwrap();
        assert_eq!(result, "ok");
    }

    #[test]
    fn json_to_lua_roundtrip_object() {
        let sandbox = LuaSandbox::new().unwrap();
        let original = json!({"key": "value", "num": 123, "flag": true});
        sandbox.set_json("data", &original).unwrap();

        let key = sandbox.exec("return data.key").unwrap();
        assert_eq!(key, "value");
        let num = sandbox.exec("return data.num").unwrap();
        assert_eq!(num, "123");
        let flag = sandbox.exec("return tostring(data.flag)").unwrap();
        assert_eq!(flag, "true");
    }

    #[test]
    fn json_to_lua_null() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox.set_json("x", &json!(null)).unwrap();
        let result = sandbox.exec("return type(x)").unwrap();
        assert_eq!(result, "nil");
    }

    #[test]
    fn extract_lua_blocks_basic() {
        let text = r#"Some text
```lua
local x = 1
return x
```
More text"#;
        let blocks = extract_lua_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("local x = 1"));
    }

    #[test]
    fn extract_lua_blocks_multiple() {
        let text = r#"
```lua
print("first")
```
middle
```lua
print("second")
```
"#;
        let blocks = extract_lua_blocks(text);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[0].contains("first"));
        assert!(blocks[1].contains("second"));
    }

    #[test]
    fn extract_lua_blocks_none() {
        let blocks = extract_lua_blocks("no code blocks here");
        assert!(blocks.is_empty());
    }

    #[test]
    fn extract_lua_blocks_ignores_other_languages() {
        let text = r#"
```python
print("not lua")
```
```lua
print("lua!")
```
```rust
fn main() {}
```
"#;
        let blocks = extract_lua_blocks(text);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("lua!"));
    }

    #[test]
    fn lua_string_operations() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox.set_string("text", "hello world").unwrap();
        let result = sandbox.exec("return string.upper(text)").unwrap();
        assert_eq!(result, "HELLO WORLD");
    }

    #[test]
    fn lua_table_operations() {
        let sandbox = LuaSandbox::new().unwrap();
        let result = sandbox
            .exec(
                r#"
            local t = {1, 2, 3, 4, 5}
            local sum = 0
            for _, v in ipairs(t) do
                sum = sum + v
            end
            return sum
        "#,
            )
            .unwrap();
        assert_eq!(result, "15");
    }

    #[test]
    fn lua_to_json_array() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox
            .set_json("items", &json!(["a", "b", "c"]))
            .unwrap();
        // Round-trip: Lua table -> JSON
        let lua_val: Value = sandbox.lua.globals().get("items").unwrap();
        let json_val = lua_to_json(&lua_val);
        assert_eq!(json_val, json!(["a", "b", "c"]));
    }

    #[test]
    fn lua_to_json_object() {
        let sandbox = LuaSandbox::new().unwrap();
        sandbox
            .set_json("obj", &json!({"name": "test"}))
            .unwrap();
        let lua_val: Value = sandbox.lua.globals().get("obj").unwrap();
        let json_val = lua_to_json(&lua_val);
        assert_eq!(json_val["name"], "test");
    }
}
