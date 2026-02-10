//! In-memory VFS implementation for testing and WASM.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::RwLock;

use crate::error::{SoulError, SoulResult};

use super::{VfsDirEntry, VfsMetadata, VirtualFs};

/// In-memory filesystem backed by a `BTreeMap`.
///
/// Thread-safe via `RwLock`. Suitable for testing and WASM environments.
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
