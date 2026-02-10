use std::path::{Path, PathBuf};

use crate::error::SoulResult;

/// Memory hierarchy manager
///
/// L1: Current context window (volatile, in-memory)
/// L2: Session transcript (persistent, JSONL)
/// L3: Memory files (durable, semantic-searchable)
/// L4: External tools/APIs (unlimited, on-demand)
///
/// This module manages L3 — durable memory files that persist across sessions.
pub struct MemoryStore {
    base_dir: PathBuf,
}

impl MemoryStore {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn memory_file(&self) -> PathBuf {
        self.base_dir.join("MEMORY.md")
    }

    fn topic_dir(&self) -> PathBuf {
        self.base_dir.join("memory")
    }

    /// Read the main MEMORY.md file (always injected into system prompt)
    pub async fn read_main(&self) -> SoulResult<Option<String>> {
        let path = self.memory_file();
        if !path.exists() {
            return Ok(None);
        }
        let content = tokio::fs::read_to_string(&path).await?;
        Ok(Some(content))
    }

    /// Write the main MEMORY.md file
    pub async fn write_main(&self, content: &str) -> SoulResult<()> {
        tokio::fs::create_dir_all(&self.base_dir).await?;
        tokio::fs::write(self.memory_file(), content).await?;
        Ok(())
    }

    /// Read a topic-specific memory file
    pub async fn read_topic(&self, topic: &str) -> SoulResult<Option<String>> {
        let path = self.topic_dir().join(format!("{topic}.md"));
        if !path.exists() {
            return Ok(None);
        }
        let content = tokio::fs::read_to_string(&path).await?;
        Ok(Some(content))
    }

    /// Write a topic-specific memory file
    pub async fn write_topic(&self, topic: &str, content: &str) -> SoulResult<()> {
        let dir = self.topic_dir();
        tokio::fs::create_dir_all(&dir).await?;
        tokio::fs::write(dir.join(format!("{topic}.md")), content).await?;
        Ok(())
    }

    /// List all topic memory files
    pub async fn list_topics(&self) -> SoulResult<Vec<String>> {
        let dir = self.topic_dir();
        if !dir.exists() {
            return Ok(Vec::new());
        }

        let mut topics = Vec::new();
        let mut entries = tokio::fs::read_dir(&dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(topic) = name.strip_suffix(".md") {
                    topics.push(topic.to_string());
                }
            }
        }
        topics.sort();
        Ok(topics)
    }

    /// Delete a topic memory file
    pub async fn delete_topic(&self, topic: &str) -> SoulResult<()> {
        let path = self.topic_dir().join(format!("{topic}.md"));
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }
        Ok(())
    }

    /// Build the memory section for system prompt injection
    pub async fn build_prompt_section(&self) -> SoulResult<String> {
        let mut section = String::new();

        if let Some(main) = self.read_main().await? {
            section.push_str("## Memory\n\n");
            section.push_str(&main);
            section.push('\n');
        }

        let topics = self.list_topics().await?;
        if !topics.is_empty() {
            section.push_str("\n## Available memory topics\n\n");
            for topic in &topics {
                section.push_str(&format!("- {topic}\n"));
            }
        }

        Ok(section)
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

/// Bootstrap files that are read-only and injected into system prompt
pub struct BootstrapFiles {
    base_dir: PathBuf,
}

impl BootstrapFiles {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    /// Known bootstrap file names
    const FILES: &[&str] = &[
        "CLAUDE.md",
        "AGENTS.md",
        "SOUL.md",
        "TOOLS.md",
        "IDENTITY.md",
    ];

    /// Read all bootstrap files that exist
    pub async fn read_all(&self) -> SoulResult<Vec<(String, String)>> {
        let mut results = Vec::new();
        for &name in Self::FILES {
            let path = self.base_dir.join(name);
            if path.exists() {
                let content = tokio::fs::read_to_string(&path).await?;
                results.push((name.to_string(), content));
            }
        }
        Ok(results)
    }

    /// Build bootstrap section for system prompt
    pub async fn build_prompt_section(&self) -> SoulResult<String> {
        let files = self.read_all().await?;
        let mut section = String::new();

        for (name, content) in &files {
            section.push_str(&format!("\n## {name}\n\n{content}\n"));
        }

        Ok(section)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn memory_store_read_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let result = store.read_main().await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn memory_store_write_and_read_main() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());

        store
            .write_main("# Key Patterns\n\n- Pattern 1\n- Pattern 2\n")
            .await
            .unwrap();

        let content = store.read_main().await.unwrap().unwrap();
        assert!(content.contains("Pattern 1"));
        assert!(content.contains("Pattern 2"));
    }

    #[tokio::test]
    async fn memory_store_topics() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());

        store
            .write_topic("debugging", "# Debugging Notes\n\nTip 1\n")
            .await
            .unwrap();
        store
            .write_topic("architecture", "# Architecture\n\nDesign doc\n")
            .await
            .unwrap();

        let topics = store.list_topics().await.unwrap();
        assert_eq!(topics.len(), 2);
        assert!(topics.contains(&"debugging".to_string()));
        assert!(topics.contains(&"architecture".to_string()));

        let content = store.read_topic("debugging").await.unwrap().unwrap();
        assert!(content.contains("Tip 1"));
    }

    #[tokio::test]
    async fn memory_store_delete_topic() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());

        store.write_topic("temp", "temporary").await.unwrap();
        assert!(store.read_topic("temp").await.unwrap().is_some());

        store.delete_topic("temp").await.unwrap();
        assert!(store.read_topic("temp").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn memory_store_delete_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        // Should not error
        store.delete_topic("nonexistent").await.unwrap();
    }

    #[tokio::test]
    async fn memory_store_build_prompt_section() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());

        store.write_main("Main memory content").await.unwrap();
        store
            .write_topic("patterns", "Some patterns")
            .await
            .unwrap();

        let section = store.build_prompt_section().await.unwrap();
        assert!(section.contains("Main memory content"));
        assert!(section.contains("patterns"));
    }

    #[tokio::test]
    async fn memory_store_empty_prompt_section() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryStore::new(dir.path());
        let section = store.build_prompt_section().await.unwrap();
        assert!(section.is_empty());
    }

    #[tokio::test]
    async fn bootstrap_files_empty() {
        let dir = tempfile::tempdir().unwrap();
        let bootstrap = BootstrapFiles::new(dir.path());
        let files = bootstrap.read_all().await.unwrap();
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn bootstrap_files_reads_existing() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("CLAUDE.md"), "Project instructions")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("SOUL.md"), "Agent personality")
            .await
            .unwrap();

        let bootstrap = BootstrapFiles::new(dir.path());
        let files = bootstrap.read_all().await.unwrap();
        assert_eq!(files.len(), 2);
        assert!(files.iter().any(|(n, _)| n == "CLAUDE.md"));
        assert!(files.iter().any(|(n, _)| n == "SOUL.md"));
    }

    #[tokio::test]
    async fn bootstrap_build_prompt_section() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(
            dir.path().join("CLAUDE.md"),
            "# Project\n\nInstructions here",
        )
        .await
        .unwrap();

        let bootstrap = BootstrapFiles::new(dir.path());
        let section = bootstrap.build_prompt_section().await.unwrap();
        assert!(section.contains("CLAUDE.md"));
        assert!(section.contains("Instructions here"));
    }
}
