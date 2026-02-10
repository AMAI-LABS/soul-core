//! Native filesystem VFS implementation using `tokio::fs`.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

use crate::error::SoulResult;

use super::{VfsDirEntry, VfsMetadata, VirtualFs};

/// Native OS filesystem backed by `tokio::fs`.
///
/// All paths are resolved relative to a `root` directory.
pub struct NativeFs {
    root: PathBuf,
}

impl NativeFs {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, path: &str) -> PathBuf {
        self.root.join(path.trim_start_matches('/'))
    }
}

impl VirtualFs for NativeFs {
    fn read_to_string<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<String>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            let content = tokio::fs::read_to_string(&full).await?;
            Ok(content)
        })
    }

    fn write<'a>(
        &'a self,
        path: &'a str,
        contents: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            if let Some(parent) = full.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&full, contents).await?;
            Ok(())
        })
    }

    fn append<'a>(
        &'a self,
        path: &'a str,
        contents: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            if let Some(parent) = full.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            use tokio::io::AsyncWriteExt;
            let mut file = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&full)
                .await?;
            file.write_all(contents.as_bytes()).await?;
            Ok(())
        })
    }

    fn exists<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<bool>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            Ok(full.exists())
        })
    }

    fn create_dir_all<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            tokio::fs::create_dir_all(&full).await?;
            Ok(())
        })
    }

    fn remove_file<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<()>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            if full.exists() {
                tokio::fs::remove_file(&full).await?;
            }
            Ok(())
        })
    }

    fn read_dir<'a>(
        &'a self,
        path: &'a str,
    ) -> Pin<Box<dyn Future<Output = SoulResult<Vec<VfsDirEntry>>> + Send + 'a>> {
        Box::pin(async move {
            let full = self.resolve(path);
            let mut entries = Vec::new();
            let mut reader = tokio::fs::read_dir(&full).await?;
            while let Some(entry) = reader.next_entry().await? {
                let name = entry.file_name().to_string_lossy().to_string();
                let file_type = entry.file_type().await?;
                entries.push(VfsDirEntry {
                    name,
                    is_file: file_type.is_file(),
                    is_dir: file_type.is_dir(),
                });
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
            let full = self.resolve(path);
            let meta = tokio::fs::metadata(&full).await?;
            Ok(VfsMetadata {
                is_file: meta.is_file(),
                is_dir: meta.is_dir(),
                size: meta.len(),
            })
        })
    }
}
