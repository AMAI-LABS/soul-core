//! Virtual Filesystem — platform-agnostic storage abstraction.
//!
//! Provides a [`VirtualFs`] trait that decouples all I/O from concrete filesystem
//! implementations. Ship with [`NativeFs`] for real OS filesystems (behind the
//! `native` feature) and [`MemoryFs`] for testing and WASM environments.

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

use crate::error::SoulResult;

/// Metadata about a virtual filesystem entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsMetadata {
    pub is_file: bool,
    pub is_dir: bool,
    pub size: u64,
}

/// A single directory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VfsDirEntry {
    pub name: String,
    pub is_file: bool,
    pub is_dir: bool,
}

/// Platform-agnostic filesystem trait.
///
/// All paths are logical strings (forward-slash separated, relative to the
/// VFS root). Implementations map them to whatever backing store they use.
pub trait VirtualFs: Send + Sync {
    /// Read the entire contents of a file as a UTF-8 string.
    fn read_to_string<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<String>> + Send + 'a>>;

    /// Write `contents` to a file, creating or overwriting it.
    fn write<'a>(
        &'a self,
        path: &'a str,
        contents: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>>;

    /// Append `contents` to a file, creating it if it doesn't exist.
    fn append<'a>(
        &'a self,
        path: &'a str,
        contents: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>>;

    /// Check whether a path exists.
    fn exists<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<bool>> + Send + 'a>>;

    /// Create a directory (and parents) at the given path.
    fn create_dir_all<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>>;

    /// Remove a file.
    fn remove_file<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>>;

    /// List entries in a directory.
    fn read_dir<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<Vec<VfsDirEntry>>> + Send + 'a>>;

    /// Get metadata for a path.
    fn metadata<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<VfsMetadata>> + Send + 'a>>;
}

// ─── MemoryFs ──────────────────────────────────────────────────────────────

mod memory;
pub use memory::MemoryFs;

// ─── NativeFs (behind `native` feature, default) ──────────────────────────

#[cfg(feature = "native")]
mod native;
#[cfg(feature = "native")]
pub use native::NativeFs;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_fs_write_and_read() {
        let fs = MemoryFs::new();
        fs.write("hello.txt", "world").await.unwrap();
        let content = fs.read_to_string("hello.txt").await.unwrap();
        assert_eq!(content, "world");
    }

    #[tokio::test]
    async fn memory_fs_overwrite() {
        let fs = MemoryFs::new();
        fs.write("f.txt", "first").await.unwrap();
        fs.write("f.txt", "second").await.unwrap();
        let content = fs.read_to_string("f.txt").await.unwrap();
        assert_eq!(content, "second");
    }

    #[tokio::test]
    async fn memory_fs_append() {
        let fs = MemoryFs::new();
        fs.append("log.txt", "line1\n").await.unwrap();
        fs.append("log.txt", "line2\n").await.unwrap();
        let content = fs.read_to_string("log.txt").await.unwrap();
        assert_eq!(content, "line1\nline2\n");
    }

    #[tokio::test]
    async fn memory_fs_exists() {
        let fs = MemoryFs::new();
        assert!(!fs.exists("nope.txt").await.unwrap());
        fs.write("yes.txt", "").await.unwrap();
        assert!(fs.exists("yes.txt").await.unwrap());
    }

    #[tokio::test]
    async fn memory_fs_remove_file() {
        let fs = MemoryFs::new();
        fs.write("del.txt", "bye").await.unwrap();
        assert!(fs.exists("del.txt").await.unwrap());
        fs.remove_file("del.txt").await.unwrap();
        assert!(!fs.exists("del.txt").await.unwrap());
    }

    #[tokio::test]
    async fn memory_fs_remove_nonexistent_ok() {
        let fs = MemoryFs::new();
        // Should not error
        fs.remove_file("ghost.txt").await.unwrap();
    }

    #[tokio::test]
    async fn memory_fs_read_nonexistent_errors() {
        let fs = MemoryFs::new();
        let result = fs.read_to_string("nope.txt").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn memory_fs_create_dir_all() {
        let fs = MemoryFs::new();
        fs.create_dir_all("a/b/c").await.unwrap();
        assert!(fs.exists("a/b/c").await.unwrap());
        assert!(fs.exists("a/b").await.unwrap());
        assert!(fs.exists("a").await.unwrap());
    }

    #[tokio::test]
    async fn memory_fs_read_dir() {
        let fs = MemoryFs::new();
        fs.write("dir/a.txt", "a").await.unwrap();
        fs.write("dir/b.txt", "b").await.unwrap();
        fs.write("dir/sub/c.txt", "c").await.unwrap();

        let entries = fs.read_dir("dir").await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"a.txt"));
        assert!(names.contains(&"b.txt"));
        assert!(names.contains(&"sub"));
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn memory_fs_read_dir_empty() {
        let fs = MemoryFs::new();
        fs.create_dir_all("empty").await.unwrap();
        let entries = fs.read_dir("empty").await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn memory_fs_metadata_file() {
        let fs = MemoryFs::new();
        fs.write("file.txt", "hello").await.unwrap();
        let meta = fs.metadata("file.txt").await.unwrap();
        assert!(meta.is_file);
        assert!(!meta.is_dir);
        assert_eq!(meta.size, 5);
    }

    #[tokio::test]
    async fn memory_fs_metadata_dir() {
        let fs = MemoryFs::new();
        fs.create_dir_all("mydir").await.unwrap();
        let meta = fs.metadata("mydir").await.unwrap();
        assert!(!meta.is_file);
        assert!(meta.is_dir);
    }

    #[tokio::test]
    async fn memory_fs_metadata_nonexistent_errors() {
        let fs = MemoryFs::new();
        let result = fs.metadata("nope").await;
        assert!(result.is_err());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_fs_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let fs = NativeFs::new(dir.path());
        fs.write("test.txt", "native content").await.unwrap();
        let content = fs.read_to_string("test.txt").await.unwrap();
        assert_eq!(content, "native content");
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_fs_append() {
        let dir = tempfile::tempdir().unwrap();
        let fs = NativeFs::new(dir.path());
        fs.append("log.txt", "a").await.unwrap();
        fs.append("log.txt", "b").await.unwrap();
        let content = fs.read_to_string("log.txt").await.unwrap();
        assert_eq!(content, "ab");
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_fs_exists() {
        let dir = tempfile::tempdir().unwrap();
        let fs = NativeFs::new(dir.path());
        assert!(!fs.exists("nope.txt").await.unwrap());
        fs.write("yes.txt", "").await.unwrap();
        assert!(fs.exists("yes.txt").await.unwrap());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_fs_remove_file() {
        let dir = tempfile::tempdir().unwrap();
        let fs = NativeFs::new(dir.path());
        fs.write("del.txt", "bye").await.unwrap();
        fs.remove_file("del.txt").await.unwrap();
        assert!(!fs.exists("del.txt").await.unwrap());
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_fs_create_dir_and_read_dir() {
        let dir = tempfile::tempdir().unwrap();
        let fs = NativeFs::new(dir.path());
        fs.create_dir_all("sub/dir").await.unwrap();
        fs.write("sub/dir/a.txt", "a").await.unwrap();
        fs.write("sub/dir/b.txt", "b").await.unwrap();

        let entries = fs.read_dir("sub/dir").await.unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_fs_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let fs = NativeFs::new(dir.path());
        fs.write("meta.txt", "12345").await.unwrap();
        let meta = fs.metadata("meta.txt").await.unwrap();
        assert!(meta.is_file);
        assert_eq!(meta.size, 5);
    }
}
