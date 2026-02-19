//! In-memory VFS implementation for testing and WASM.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::error::{SoulError, SoulResult};

use super::{VfsDirEntry, VfsMetadata, VirtualFs};

/// Serializable snapshot of a [`MemoryFs`].
///
/// Can be persisted as JSON, bincode, MessagePack, or any serde-compatible
/// format. Use [`MemoryFs::snapshot`] to capture and [`MemoryFs::from_snapshot`]
/// to restore.
///
/// ```rust
/// use soul_core::vfs::MemoryFs;
///
/// let fs = MemoryFs::new();
/// // ... write files ...
/// let snapshot = fs.snapshot();
///
/// // Serialize to JSON string
/// let json = snapshot.to_json().unwrap();
///
/// // Restore from JSON
/// let restored = MemoryFs::from_json(&json).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryFsSnapshot {
    /// All files: path -> contents
    pub files: BTreeMap<String, String>,
    /// All explicitly created directories
    pub dirs: BTreeSet<String>,
}

impl MemoryFsSnapshot {
    /// Serialize to JSON string.
    pub fn to_json(&self) -> SoulResult<String> {
        serde_json::to_string(self).map_err(SoulError::Serialization)
    }

    /// Serialize to pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> SoulResult<String> {
        serde_json::to_string_pretty(self).map_err(SoulError::Serialization)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> SoulResult<Self> {
        serde_json::from_str(json).map_err(SoulError::Serialization)
    }

    /// Serialize to binary (serde_json value encoding for portability).
    /// Returns a compact binary representation that can be stored anywhere.
    pub fn to_bytes(&self) -> SoulResult<Vec<u8>> {
        serde_json::to_vec(self).map_err(SoulError::Serialization)
    }

    /// Deserialize from binary bytes.
    pub fn from_bytes(bytes: &[u8]) -> SoulResult<Self> {
        serde_json::from_slice(bytes).map_err(SoulError::Serialization)
    }

    /// Total number of files.
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Total size of all file contents in bytes.
    pub fn total_size(&self) -> usize {
        self.files.values().map(|c| c.len()).sum()
    }
}

/// In-memory filesystem backed by a `BTreeMap`.
///
/// Thread-safe via `RwLock`. Suitable for testing and WASM environments.
///
/// Supports snapshotting and restoring via [`MemoryFsSnapshot`], enabling
/// the entire filesystem state to be serialized to JSON or binary and
/// persisted to any storage backend.
pub struct MemoryFs {
    files: RwLock<BTreeMap<String, String>>,
    dirs: RwLock<BTreeSet<String>>,
}

impl MemoryFs {
    pub fn new() -> Self {
        Self {
            files: RwLock::new(BTreeMap::new()),
            dirs: RwLock::new(BTreeSet::new()),
        }
    }

    /// Create a MemoryFs from a snapshot.
    pub fn from_snapshot(snapshot: MemoryFsSnapshot) -> Self {
        Self {
            files: RwLock::new(snapshot.files),
            dirs: RwLock::new(snapshot.dirs),
        }
    }

    /// Create a MemoryFs from a JSON string.
    pub fn from_json(json: &str) -> SoulResult<Self> {
        let snapshot = MemoryFsSnapshot::from_json(json)?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Create a MemoryFs from binary bytes.
    pub fn from_bytes(bytes: &[u8]) -> SoulResult<Self> {
        let snapshot = MemoryFsSnapshot::from_bytes(bytes)?;
        Ok(Self::from_snapshot(snapshot))
    }

    /// Capture the current state as a serializable snapshot.
    pub fn snapshot(&self) -> MemoryFsSnapshot {
        let files = self.files.read().unwrap().clone();
        let dirs = self.dirs.read().unwrap().clone();
        MemoryFsSnapshot { files, dirs }
    }

    /// Serialize the current state to a JSON string.
    pub fn to_json(&self) -> SoulResult<String> {
        self.snapshot().to_json()
    }

    /// Serialize the current state to a pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> SoulResult<String> {
        self.snapshot().to_json_pretty()
    }

    /// Serialize the current state to binary bytes.
    pub fn to_bytes(&self) -> SoulResult<Vec<u8>> {
        self.snapshot().to_bytes()
    }

    /// Replace the current state with a snapshot (restoring from backup).
    pub fn restore(&self, snapshot: MemoryFsSnapshot) {
        let mut files = self.files.write().unwrap();
        let mut dirs = self.dirs.write().unwrap();
        *files = snapshot.files;
        *dirs = snapshot.dirs;
    }

    /// Normalize path: strip leading/trailing slashes, collapse double slashes.
    fn normalize(path: &str) -> String {
        let p = path.trim_matches('/');
        if p.is_empty() {
            ".".to_string()
        } else {
            p.to_string()
        }
    }

    /// Ensure parent directories exist for a file path.
    fn ensure_parents(&self, path: &str) {
        let mut dirs = self.dirs.write().unwrap();
        let parts: Vec<&str> = path.split('/').collect();
        for i in 1..parts.len() {
            let parent = parts[..i].join("/");
            dirs.insert(parent);
        }
    }
}

impl Default for MemoryFs {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualFs for MemoryFs {
    fn read_to_string<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            let files = self.files.read().unwrap();
            files.get(&normalized).cloned().ok_or_else(|| {
                SoulError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("File not found: {normalized}"),
                ))
            })
        })
    }

    fn write<'a>(
        &'a self,
        path: &'a str,
        contents: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            self.ensure_parents(&normalized);
            let mut files = self.files.write().unwrap();
            files.insert(normalized, contents.to_string());
            Ok(())
        })
    }

    fn append<'a>(
        &'a self,
        path: &'a str,
        contents: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            self.ensure_parents(&normalized);
            let mut files = self.files.write().unwrap();
            let entry = files.entry(normalized).or_default();
            entry.push_str(contents);
            Ok(())
        })
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<bool>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            let files = self.files.read().unwrap();
            if files.contains_key(&normalized) {
                return Ok(true);
            }
            let dirs = self.dirs.read().unwrap();
            Ok(dirs.contains(&normalized))
        })
    }

    fn create_dir_all<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            let mut dirs = self.dirs.write().unwrap();
            // Insert the dir itself and all parents
            let parts: Vec<&str> = normalized.split('/').collect();
            for i in 1..=parts.len() {
                let dir = parts[..i].join("/");
                dirs.insert(dir);
            }
            Ok(())
        })
    }

    fn remove_file<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            let mut files = self.files.write().unwrap();
            files.remove(&normalized);
            Ok(())
        })
    }

    fn read_dir<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<Vec<VfsDirEntry>>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);
            let prefix = if normalized == "." {
                String::new()
            } else {
                format!("{normalized}/")
            };

            let files = self.files.read().unwrap();
            let dirs = self.dirs.read().unwrap();

            // Check that the directory actually exists (root "." always exists)
            if normalized != "." {
                let dir_exists =
                    dirs.contains(&normalized) || files.keys().any(|k| k.starts_with(&prefix));
                if !dir_exists {
                    return Err(SoulError::Io(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("Directory not found: {normalized}"),
                    )));
                }
            }

            let mut seen = BTreeSet::new();
            let mut entries = Vec::new();

            // Scan files under this prefix
            for key in files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    // Only direct children (no further slashes)
                    if let Some(name) = rest.split('/').next() {
                        if rest.contains('/') {
                            // This is a child in a subdirectory — add the subdir
                            if seen.insert(name.to_string()) {
                                entries.push(VfsDirEntry {
                                    name: name.to_string(),
                                    is_file: false,
                                    is_dir: true,
                                });
                            }
                        } else if seen.insert(name.to_string()) {
                            entries.push(VfsDirEntry {
                                name: name.to_string(),
                                is_file: true,
                                is_dir: false,
                            });
                        }
                    }
                }
            }

            // Scan explicit dirs under this prefix
            for dir in dirs.iter() {
                if let Some(rest) = dir.strip_prefix(&prefix) {
                    if !rest.is_empty() && !rest.contains('/') && seen.insert(rest.to_string()) {
                        entries.push(VfsDirEntry {
                            name: rest.to_string(),
                            is_file: false,
                            is_dir: true,
                        });
                    }
                }
            }

            entries.sort_by(|a, b| a.name.cmp(&b.name));
            Ok(entries)
        })
    }

    fn metadata<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<VfsMetadata>> + Send + 'a>> {
        Box::pin(async move {
            let normalized = Self::normalize(path);

            let files = self.files.read().unwrap();
            if let Some(content) = files.get(&normalized) {
                return Ok(VfsMetadata {
                    is_file: true,
                    is_dir: false,
                    size: content.len() as u64,
                });
            }

            let dirs = self.dirs.read().unwrap();
            if dirs.contains(&normalized) {
                return Ok(VfsMetadata {
                    is_file: false,
                    is_dir: true,
                    size: 0,
                });
            }

            Err(SoulError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Path not found: {normalized}"),
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn snapshot_roundtrip_json() {
        let fs = MemoryFs::new();
        fs.write("a.txt", "hello").await.unwrap();
        fs.write("dir/b.txt", "world").await.unwrap();
        fs.create_dir_all("empty_dir").await.unwrap();

        let json = fs.to_json().unwrap();
        let restored = MemoryFs::from_json(&json).unwrap();

        assert_eq!(restored.read_to_string("a.txt").await.unwrap(), "hello");
        assert_eq!(restored.read_to_string("dir/b.txt").await.unwrap(), "world");
        assert!(restored.exists("empty_dir").await.unwrap());
    }

    #[tokio::test]
    async fn snapshot_roundtrip_bytes() {
        let fs = MemoryFs::new();
        fs.write("file.rs", "fn main() {}").await.unwrap();

        let bytes = fs.to_bytes().unwrap();
        let restored = MemoryFs::from_bytes(&bytes).unwrap();

        assert_eq!(
            restored.read_to_string("file.rs").await.unwrap(),
            "fn main() {}"
        );
    }

    #[tokio::test]
    async fn snapshot_equality() {
        let fs = MemoryFs::new();
        fs.write("x", "1").await.unwrap();
        fs.write("y", "2").await.unwrap();

        let snap1 = fs.snapshot();
        let snap2 = fs.snapshot();
        assert_eq!(snap1, snap2);
    }

    #[tokio::test]
    async fn snapshot_restore() {
        let fs = MemoryFs::new();
        fs.write("original.txt", "data").await.unwrap();
        let checkpoint = fs.snapshot();

        // Modify
        fs.write("original.txt", "changed").await.unwrap();
        fs.write("new.txt", "extra").await.unwrap();

        // Restore
        fs.restore(checkpoint);
        assert_eq!(fs.read_to_string("original.txt").await.unwrap(), "data");
        assert!(!fs.exists("new.txt").await.unwrap());
    }

    #[test]
    fn snapshot_file_count_and_size() {
        let snap = MemoryFsSnapshot {
            files: BTreeMap::from([("a".into(), "hello".into()), ("b".into(), "world!".into())]),
            dirs: BTreeSet::new(),
        };
        assert_eq!(snap.file_count(), 2);
        assert_eq!(snap.total_size(), 11); // 5 + 6
    }

    #[test]
    fn snapshot_json_pretty() {
        let snap = MemoryFsSnapshot {
            files: BTreeMap::from([("test.txt".into(), "content".into())]),
            dirs: BTreeSet::new(),
        };
        let json = snap.to_json_pretty().unwrap();
        assert!(json.contains("test.txt"));
        assert!(json.contains('\n')); // pretty-printed
    }

    #[test]
    fn snapshot_serde_roundtrip() {
        let snap = MemoryFsSnapshot {
            files: BTreeMap::from([("a".into(), "1".into()), ("b/c".into(), "2".into())]),
            dirs: BTreeSet::from(["b".into(), "d".into()]),
        };
        let json = snap.to_json().unwrap();
        let restored = MemoryFsSnapshot::from_json(&json).unwrap();
        assert_eq!(snap, restored);

        let bytes = snap.to_bytes().unwrap();
        let restored = MemoryFsSnapshot::from_bytes(&bytes).unwrap();
        assert_eq!(snap, restored);
    }

    #[test]
    fn empty_snapshot() {
        let snap = MemoryFs::new().snapshot();
        assert_eq!(snap.file_count(), 0);
        assert_eq!(snap.total_size(), 0);
        assert!(snap.files.is_empty());
        assert!(snap.dirs.is_empty());
    }
}
