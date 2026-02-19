//! Versioned snapshot log — append-only history of arbitrary state.
//!
//! Each save appends a `VersionedSnapshot<T>` to a JSONL file. The log can be
//! replayed to restore any previous version, enabling rollback after mistakes
//! or crash recovery to the last known-good state.
//!
//! Generic over any `Serialize + DeserializeOwned` type — works for planner
//! state, agent config, cost summaries, or any composite state struct.
//!
//! Storage uses `VirtualFs` — works on native filesystem or in-memory (WASM).

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::error::{SoulError, SoulResult};
use crate::vfs::VirtualFs;

/// A snapshot with version metadata, generic over the payload type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VersionedSnapshot<T> {
    /// Monotonically increasing version number (1-indexed).
    pub version: u64,
    /// When this snapshot was saved.
    pub timestamp: DateTime<Utc>,
    /// Optional label (e.g. "before deploy", "after retry").
    #[serde(default)]
    pub label: Option<String>,
    /// The state at this version.
    pub snapshot: T,
}

/// Lightweight version info without the full snapshot payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionInfo {
    pub version: u64,
    pub timestamp: DateTime<Utc>,
    pub label: Option<String>,
}

/// Append-only versioned log stored as JSONL via VirtualFs.
///
/// Works with any serializable type. Each entry is one JSON line containing
/// the version number, timestamp, optional label, and the full snapshot.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use soul_core::snapshot::SnapshotLog;
/// use soul_core::vfs::MemoryFs;
///
/// # async fn example() {
/// let fs: Arc<dyn soul_core::vfs::VirtualFs> = Arc::new(MemoryFs::new());
/// let log: SnapshotLog<String> = SnapshotLog::new(fs, "state.jsonl");
///
/// let v = log.save("hello".to_string(), Some("initial".into())).await.unwrap();
/// assert_eq!(v, 1);
///
/// let latest = log.load_latest().await.unwrap().unwrap();
/// assert_eq!(latest.snapshot, "hello");
/// # }
/// ```
pub struct SnapshotLog<T> {
    fs: Arc<dyn VirtualFs>,
    path: String,
    _marker: std::marker::PhantomData<T>,
}

impl<T> SnapshotLog<T>
where
    T: Serialize + DeserializeOwned + Clone,
{
    /// Create a new snapshot log at the given path.
    pub fn new(fs: Arc<dyn VirtualFs>, path: impl Into<String>) -> Self {
        Self {
            fs,
            path: path.into(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Save a snapshot, appending it to the log. Returns the version number.
    pub async fn save(&self, snapshot: T, label: Option<String>) -> SoulResult<u64> {
        let version = self.next_version().await?;

        let versioned = VersionedSnapshot {
            version,
            timestamp: Utc::now(),
            label,
            snapshot,
        };

        let mut line = serde_json::to_string(&versioned).map_err(SoulError::Serialization)?;
        line.push('\n');

        self.fs.append(&self.path, &line).await?;

        Ok(version)
    }

    /// Load the latest snapshot from the log.
    pub async fn load_latest(&self) -> SoulResult<Option<VersionedSnapshot<T>>> {
        let entries = self.read_all().await?;
        Ok(entries.into_iter().last())
    }

    /// Load a specific version from the log.
    pub async fn load_version(&self, version: u64) -> SoulResult<Option<VersionedSnapshot<T>>> {
        let entries = self.read_all().await?;
        Ok(entries.into_iter().find(|e| e.version == version))
    }

    /// Rollback: save a copy of a previous version as the new latest.
    ///
    /// This doesn't truncate history — it appends the old snapshot as a new
    /// version, preserving the full audit trail.
    pub async fn rollback(&self, version: u64) -> SoulResult<u64> {
        let target = self
            .load_version(version)
            .await?
            .ok_or_else(|| SoulError::Other(anyhow::anyhow!("Version {version} not found")))?;

        let label = Some(format!("rollback to v{version}"));
        self.save(target.snapshot, label).await
    }

    /// List all versions with metadata (no full snapshot data).
    pub async fn history(&self) -> SoulResult<Vec<VersionInfo>> {
        let entries = self.read_all().await?;
        Ok(entries
            .into_iter()
            .map(|e| VersionInfo {
                version: e.version,
                timestamp: e.timestamp,
                label: e.label,
            })
            .collect())
    }

    /// Number of versions in the log.
    pub async fn version_count(&self) -> SoulResult<u64> {
        let entries = self.read_all().await?;
        Ok(entries.len() as u64)
    }

    /// Read all entries from the JSONL log.
    async fn read_all(&self) -> SoulResult<Vec<VersionedSnapshot<T>>> {
        let exists = self.fs.exists(&self.path).await?;
        if !exists {
            return Ok(Vec::new());
        }

        let content = self.fs.read_to_string(&self.path).await?;
        let mut entries = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let entry: VersionedSnapshot<T> =
                serde_json::from_str(trimmed).map_err(SoulError::Serialization)?;
            entries.push(entry);
        }

        Ok(entries)
    }

    /// Compute the next version number.
    async fn next_version(&self) -> SoulResult<u64> {
        let entries = self.read_all().await?;
        Ok(entries.last().map(|e| e.version + 1).unwrap_or(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::Planner;
    use crate::vfs::MemoryFs;

    fn make_fs() -> Arc<dyn VirtualFs> {
        Arc::new(MemoryFs::new())
    }

    // --- Tests with PlannerSnapshot (the primary use case) ---

    #[tokio::test]
    async fn save_and_load_latest() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        planner.add_task("Build", None::<String>);

        let v = log.save(planner.snapshot(), None).await.unwrap();
        assert_eq!(v, 1);

        let loaded = log.load_latest().await.unwrap().unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.snapshot.tasks.len(), 1);
    }

    #[tokio::test]
    async fn versions_increment() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        let v1 = log.save(planner.snapshot(), None).await.unwrap();

        planner.add_task("B", None::<String>);
        let v2 = log.save(planner.snapshot(), None).await.unwrap();

        assert_eq!(v1, 1);
        assert_eq!(v2, 2);
    }

    #[tokio::test]
    async fn load_specific_version() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        log.save(planner.snapshot(), Some("first".into()))
            .await
            .unwrap();

        planner.add_task("B", None::<String>);
        log.save(planner.snapshot(), Some("second".into()))
            .await
            .unwrap();

        let v1 = log.load_version(1).await.unwrap().unwrap();
        assert_eq!(v1.snapshot.tasks.len(), 1);
        assert_eq!(v1.label.as_deref(), Some("first"));

        let v2 = log.load_version(2).await.unwrap().unwrap();
        assert_eq!(v2.snapshot.tasks.len(), 2);
    }

    #[tokio::test]
    async fn load_missing_version() {
        let fs = make_fs();
        let log: SnapshotLog<String> = SnapshotLog::new(fs, "plan.jsonl");
        assert!(log.load_version(999).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn load_latest_empty_log() {
        let fs = make_fs();
        let log: SnapshotLog<String> = SnapshotLog::new(fs, "plan.jsonl");
        assert!(log.load_latest().await.unwrap().is_none());
    }

    #[tokio::test]
    async fn rollback_appends_old_version() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        let t1 = planner.add_task("Good state", None::<String>);
        log.save(planner.snapshot(), Some("good".into()))
            .await
            .unwrap();

        // Make a "bad" change
        planner.start(t1).unwrap();
        planner.fail(t1).unwrap();
        log.save(planner.snapshot(), Some("bad".into()))
            .await
            .unwrap();

        // Rollback to v1
        let v3 = log.rollback(1).await.unwrap();
        assert_eq!(v3, 3);

        // Latest is now the rolled-back state
        let latest = log.load_latest().await.unwrap().unwrap();
        assert_eq!(latest.version, 3);
        assert_eq!(latest.label.as_deref(), Some("rollback to v1"));
        // Task should be back in pending state (from v1)
        let task = latest.snapshot.tasks.values().next().unwrap();
        assert_eq!(task.status, crate::planner::TaskStatus::Pending);

        // Full history preserved (3 entries)
        assert_eq!(log.version_count().await.unwrap(), 3);
    }

    #[tokio::test]
    async fn rollback_nonexistent_version_fails() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let planner = Planner::new();
        log.save(planner.snapshot(), None).await.unwrap();

        assert!(log.rollback(999).await.is_err());
    }

    #[tokio::test]
    async fn history_returns_version_info() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        log.save(planner.snapshot(), Some("initial".into()))
            .await
            .unwrap();

        planner.add_task("B", None::<String>);
        log.save(planner.snapshot(), None).await.unwrap();

        let history = log.history().await.unwrap();
        assert_eq!(history.len(), 2);

        assert_eq!(history[0].version, 1);
        assert_eq!(history[0].label.as_deref(), Some("initial"));

        assert_eq!(history[1].version, 2);
        assert!(history[1].label.is_none());
    }

    #[tokio::test]
    async fn history_empty_log() {
        let fs = make_fs();
        let log: SnapshotLog<String> = SnapshotLog::new(fs, "plan.jsonl");
        let history = log.history().await.unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn version_count() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");
        assert_eq!(log.version_count().await.unwrap(), 0);

        let planner = Planner::new();
        log.save(planner.snapshot(), None).await.unwrap();
        assert_eq!(log.version_count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn save_with_label() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let planner = Planner::new();
        log.save(planner.snapshot(), Some("before deploy".into()))
            .await
            .unwrap();

        let latest = log.load_latest().await.unwrap().unwrap();
        assert_eq!(latest.label.as_deref(), Some("before deploy"));
    }

    #[tokio::test]
    async fn snapshot_roundtrip_preserves_data() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        let t1 = planner.add_task("Build", Some("Building"));
        let t2 = planner.add_task("Test", None::<String>);
        planner.add_dependency(t2, t1).unwrap();
        planner.start(t1).unwrap();
        planner.checkpoint(t1, "compiled ok").unwrap();
        planner.complete(t1).unwrap();
        planner.add_question("Which DB?");
        planner.set_max_retries(t2, 3).unwrap();

        log.save(planner.snapshot(), Some("full state".into()))
            .await
            .unwrap();

        let loaded = log.load_latest().await.unwrap().unwrap();
        let restored = Planner::from_snapshot(loaded.snapshot);

        assert_eq!(
            restored.get(t1).unwrap().output.as_deref(),
            Some("compiled ok")
        );
        assert!(restored.is_ready(t2));
        assert_eq!(restored.get(t2).unwrap().max_retries, 3);
        assert!(restored.has_open_questions());
    }

    #[tokio::test]
    async fn multiple_rollbacks_build_history() {
        let fs = make_fs();
        let log = SnapshotLog::new(fs, "plan.jsonl");

        let mut planner = Planner::new();
        planner.add_task("A", None::<String>);
        log.save(planner.snapshot(), None).await.unwrap(); // v1

        planner.add_task("B", None::<String>);
        log.save(planner.snapshot(), None).await.unwrap(); // v2

        planner.add_task("C", None::<String>);
        log.save(planner.snapshot(), None).await.unwrap(); // v3

        // Rollback to v1 → creates v4
        log.rollback(1).await.unwrap();
        // Rollback to v2 → creates v5
        log.rollback(2).await.unwrap();

        assert_eq!(log.version_count().await.unwrap(), 5);

        let latest = log.load_latest().await.unwrap().unwrap();
        assert_eq!(latest.version, 5);
        assert_eq!(latest.snapshot.tasks.len(), 2); // v2 had A + B
    }

    // --- Tests with generic types (not just PlannerSnapshot) ---

    #[tokio::test]
    async fn works_with_simple_string() {
        let fs = make_fs();
        let log: SnapshotLog<String> = SnapshotLog::new(fs, "strings.jsonl");

        log.save("hello".into(), None).await.unwrap();
        log.save("world".into(), Some("second".into()))
            .await
            .unwrap();

        let latest = log.load_latest().await.unwrap().unwrap();
        assert_eq!(latest.snapshot, "world");
        assert_eq!(latest.version, 2);
    }

    #[tokio::test]
    async fn works_with_custom_struct() {
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        struct AppState {
            counter: u64,
            name: String,
            items: Vec<String>,
        }

        let fs = make_fs();
        let log: SnapshotLog<AppState> = SnapshotLog::new(fs, "state.jsonl");

        let state1 = AppState {
            counter: 1,
            name: "alpha".into(),
            items: vec!["a".into()],
        };
        log.save(state1.clone(), Some("initial".into()))
            .await
            .unwrap();

        let state2 = AppState {
            counter: 2,
            name: "beta".into(),
            items: vec!["a".into(), "b".into()],
        };
        log.save(state2.clone(), None).await.unwrap();

        // Rollback to v1
        log.rollback(1).await.unwrap();

        let latest = log.load_latest().await.unwrap().unwrap();
        assert_eq!(latest.version, 3);
        assert_eq!(latest.snapshot, state1);
    }

    #[tokio::test]
    async fn works_with_nested_option() {
        #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
        struct Config {
            value: Option<Vec<u32>>,
        }

        let fs = make_fs();
        let log: SnapshotLog<Config> = SnapshotLog::new(fs, "cfg.jsonl");

        log.save(Config { value: None }, None).await.unwrap();
        log.save(
            Config {
                value: Some(vec![1, 2, 3]),
            },
            None,
        )
        .await
        .unwrap();

        let v1 = log.load_version(1).await.unwrap().unwrap();
        assert_eq!(v1.snapshot.value, None);

        let v2 = log.load_version(2).await.unwrap().unwrap();
        assert_eq!(v2.snapshot.value, Some(vec![1, 2, 3]));
    }
}
