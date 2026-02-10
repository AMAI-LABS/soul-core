//! Soul Log — unified event logging system.
//!
//! Like syslog/journald but for agentic runtimes. Every action, tool call, LLM
//! request, permission check, and cost event flows through a single log pipeline
//! with multiple output sinks.
//!
//! ## Architecture
//!
//! ```text
//! AgentLoop / Tools / Hooks / etc.
//!           │
//!           ▼
//!      SoulLogger::log(entry)
//!           │
//!      ┌────┼────┐
//!      ▼    ▼    ▼
//!   Sink1 Sink2 Sink3
//!  (stdout)(vfs)(memory)
//! ```

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Severity levels for soul log entries (matches syslog conventions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Fatal => write!(f, "FATAL"),
        }
    }
}

/// A structured log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub level: LogLevel,
    /// Source module or component (e.g. "agent", "tool:bash", "mcp:context7").
    pub source: String,
    /// Session ID if available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Human-readable message.
    pub message: String,
    /// Optional structured payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

impl LogEntry {
    pub fn new(level: LogLevel, source: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            level,
            source: source.into(),
            session_id: None,
            message: message.into(),
            payload: None,
        }
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_payload(mut self, payload: serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }

    /// Format as a single-line log string.
    pub fn format_line(&self) -> String {
        let ts = self.timestamp.format("%Y-%m-%dT%H:%M:%S%.3fZ");
        let session = self
            .session_id
            .as_deref()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        format!(
            "{ts} {} {}{} {}",
            self.level, self.source, session, self.message
        )
    }
}

/// Trait for log output sinks.
///
/// Sinks receive log entries and write them to their target (stdout, file, memory, etc.).
/// Must be `Send + Sync` for concurrent use.
pub trait LogSink: Send + Sync {
    /// Write a log entry. Implementations should be non-blocking where possible.
    fn write(&self, entry: &LogEntry);

    /// Flush any buffered output.
    fn flush(&self) {}
}

/// The central logger that dispatches to multiple sinks.
pub struct SoulLogger {
    sinks: Vec<Arc<dyn LogSink>>,
    min_level: LogLevel,
}

impl SoulLogger {
    pub fn new() -> Self {
        Self {
            sinks: Vec::new(),
            min_level: LogLevel::Trace,
        }
    }

    pub fn with_level(mut self, level: LogLevel) -> Self {
        self.min_level = level;
        self
    }

    pub fn add_sink(&mut self, sink: Arc<dyn LogSink>) {
        self.sinks.push(sink);
    }

    /// Log an entry to all sinks.
    pub fn log(&self, entry: &LogEntry) {
        if entry.level < self.min_level {
            return;
        }
        for sink in &self.sinks {
            sink.write(entry);
        }
    }

    /// Convenience: log at info level.
    pub fn info(&self, source: &str, message: &str) {
        self.log(&LogEntry::new(LogLevel::Info, source, message));
    }

    /// Convenience: log at warn level.
    pub fn warn(&self, source: &str, message: &str) {
        self.log(&LogEntry::new(LogLevel::Warn, source, message));
    }

    /// Convenience: log at error level.
    pub fn error(&self, source: &str, message: &str) {
        self.log(&LogEntry::new(LogLevel::Error, source, message));
    }

    /// Convenience: log at debug level.
    pub fn debug(&self, source: &str, message: &str) {
        self.log(&LogEntry::new(LogLevel::Debug, source, message));
    }

    /// Flush all sinks.
    pub fn flush(&self) {
        for sink in &self.sinks {
            sink.flush();
        }
    }

    /// Number of attached sinks.
    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }
}

impl Default for SoulLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Built-in Sinks ────────────────────────────────────────────────────────

/// Sink that writes formatted lines to stdout.
pub struct StdoutSink;

impl LogSink for StdoutSink {
    fn write(&self, entry: &LogEntry) {
        println!("{}", entry.format_line());
    }
}

/// Sink that collects entries in memory (for testing / inspection).
pub struct MemorySink {
    entries: std::sync::Mutex<Vec<LogEntry>>,
}

impl MemorySink {
    pub fn new() -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn entries(&self) -> Vec<LogEntry> {
        self.entries.lock().unwrap().clone()
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.lock().unwrap().is_empty()
    }

    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for MemorySink {
    fn write(&self, entry: &LogEntry) {
        self.entries.lock().unwrap().push(entry.clone());
    }
}

/// Sink that writes JSONL to a VFS path (one JSON object per line).
///
/// Uses a callback to perform the write since we need to stay synchronous
/// in the `LogSink` trait. The callback queues writes for async processing.
pub struct CallbackSink {
    callback: Box<dyn Fn(&LogEntry) + Send + Sync>,
}

impl CallbackSink {
    pub fn new(callback: impl Fn(&LogEntry) + Send + Sync + 'static) -> Self {
        Self {
            callback: Box::new(callback),
        }
    }
}

impl LogSink for CallbackSink {
    fn write(&self, entry: &LogEntry) {
        (self.callback)(entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn log_entry_creates() {
        let entry = LogEntry::new(LogLevel::Info, "agent", "Starting loop");
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.source, "agent");
        assert_eq!(entry.message, "Starting loop");
        assert!(entry.session_id.is_none());
        assert!(entry.payload.is_none());
    }

    #[test]
    fn log_entry_with_session() {
        let entry = LogEntry::new(LogLevel::Info, "agent", "test").with_session("sess-123");
        assert_eq!(entry.session_id, Some("sess-123".to_string()));
    }

    #[test]
    fn log_entry_with_payload() {
        let entry =
            LogEntry::new(LogLevel::Info, "cost", "recorded").with_payload(json!({"usd": 0.05}));
        assert_eq!(entry.payload.unwrap()["usd"], 0.05);
    }

    #[test]
    fn log_entry_format_line() {
        let entry = LogEntry::new(LogLevel::Error, "tool:bash", "command failed");
        let line = entry.format_line();
        assert!(line.contains("ERROR"));
        assert!(line.contains("tool:bash"));
        assert!(line.contains("command failed"));
    }

    #[test]
    fn log_entry_format_line_with_session() {
        let entry = LogEntry::new(LogLevel::Info, "agent", "start").with_session("s1");
        let line = entry.format_line();
        assert!(line.contains("[s1]"));
    }

    #[test]
    fn log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn log_level_display() {
        assert_eq!(LogLevel::Info.to_string(), "INFO");
        assert_eq!(LogLevel::Error.to_string(), "ERROR");
    }

    #[test]
    fn log_level_serializes() {
        let json = serde_json::to_string(&LogLevel::Warn).unwrap();
        assert_eq!(json, "\"warn\"");
        let deser: LogLevel = serde_json::from_str("\"error\"").unwrap();
        assert_eq!(deser, LogLevel::Error);
    }

    #[test]
    fn log_entry_serializes_roundtrip() {
        let entry = LogEntry::new(LogLevel::Info, "test", "hello")
            .with_session("s1")
            .with_payload(json!({"key": "value"}));
        let json = serde_json::to_string(&entry).unwrap();
        let deser: LogEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.level, LogLevel::Info);
        assert_eq!(deser.source, "test");
        assert_eq!(deser.message, "hello");
        assert_eq!(deser.session_id, Some("s1".to_string()));
    }

    #[test]
    fn memory_sink_collects() {
        let sink = MemorySink::new();
        assert!(sink.is_empty());

        sink.write(&LogEntry::new(LogLevel::Info, "test", "msg1"));
        sink.write(&LogEntry::new(LogLevel::Warn, "test", "msg2"));

        assert_eq!(sink.len(), 2);
        let entries = sink.entries();
        assert_eq!(entries[0].message, "msg1");
        assert_eq!(entries[1].message, "msg2");
    }

    #[test]
    fn memory_sink_clear() {
        let sink = MemorySink::new();
        sink.write(&LogEntry::new(LogLevel::Info, "test", "msg"));
        assert_eq!(sink.len(), 1);
        sink.clear();
        assert!(sink.is_empty());
    }

    #[test]
    fn soul_logger_dispatches() {
        let sink = Arc::new(MemorySink::new());
        let mut logger = SoulLogger::new();
        logger.add_sink(sink.clone());

        logger.info("agent", "hello");
        logger.warn("tool", "caution");
        logger.error("cost", "budget exceeded");

        assert_eq!(sink.len(), 3);
    }

    #[test]
    fn soul_logger_level_filter() {
        let sink = Arc::new(MemorySink::new());
        let mut logger = SoulLogger::new().with_level(LogLevel::Warn);
        logger.add_sink(sink.clone());

        logger.debug("test", "ignored");
        logger.info("test", "ignored too");
        logger.warn("test", "visible");
        logger.error("test", "visible too");

        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn soul_logger_multiple_sinks() {
        let sink1 = Arc::new(MemorySink::new());
        let sink2 = Arc::new(MemorySink::new());
        let mut logger = SoulLogger::new();
        logger.add_sink(sink1.clone());
        logger.add_sink(sink2.clone());

        logger.info("test", "broadcast");

        assert_eq!(sink1.len(), 1);
        assert_eq!(sink2.len(), 1);
    }

    #[test]
    fn soul_logger_sink_count() {
        let mut logger = SoulLogger::new();
        assert_eq!(logger.sink_count(), 0);
        logger.add_sink(Arc::new(MemorySink::new()));
        assert_eq!(logger.sink_count(), 1);
    }

    #[test]
    fn callback_sink_invokes() {
        let counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = counter.clone();
        let sink = CallbackSink::new(move |_| {
            c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        });
        sink.write(&LogEntry::new(LogLevel::Info, "test", "msg"));
        sink.write(&LogEntry::new(LogLevel::Info, "test", "msg"));
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn soul_logger_convenience_methods() {
        let sink = Arc::new(MemorySink::new());
        let mut logger = SoulLogger::new();
        logger.add_sink(sink.clone());

        logger.debug("test", "d");
        logger.info("test", "i");
        logger.warn("test", "w");
        logger.error("test", "e");

        let entries = sink.entries();
        assert_eq!(entries[0].level, LogLevel::Debug);
        assert_eq!(entries[1].level, LogLevel::Info);
        assert_eq!(entries[2].level, LogLevel::Warn);
        assert_eq!(entries[3].level, LogLevel::Error);
    }
}
