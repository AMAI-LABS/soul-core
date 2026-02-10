//! Filesystem-based skill loader — backed by VirtualFs.

use std::sync::Arc;

use dashmap::DashMap;

use super::parser::parse_skill;
use super::SkillDefinition;
use crate::error::{SoulError, SoulResult};
use crate::vfs::VirtualFs;

/// Loads skill definitions from a directory of skill files.
pub struct SkillLoader {
    fs: Arc<dyn VirtualFs>,
    skills_dir: String,
    skills: DashMap<String, SkillDefinition>,
}

impl SkillLoader {
    pub fn new(fs: Arc<dyn VirtualFs>, skills_dir: impl Into<String>) -> Self {
        Self {
            fs,
            skills_dir: skills_dir.into(),
            skills: DashMap::new(),
        }
    }

    /// Create a skill loader backed by the native filesystem.
    #[cfg(feature = "native")]
    pub fn native(skills_dir: impl Into<std::path::PathBuf>) -> Self {
        let path: std::path::PathBuf = skills_dir.into();
        let fs = Arc::new(crate::vfs::NativeFs::new(&path));
        Self {
            fs,
            skills_dir: String::new(),
            skills: DashMap::new(),
        }
    }

    fn file_path(&self, filename: &str) -> String {
        if self.skills_dir.is_empty() {
            filename.to_string()
        } else {
            format!("{}/{filename}", self.skills_dir)
        }
    }

    /// Load all `.skill` and `.md` files from the skills directory.
    pub async fn load_all(&self) -> SoulResult<usize> {
        let dir = if self.skills_dir.is_empty() {
            ".".to_string()
        } else {
            self.skills_dir.clone()
        };

        let entries = self
            .fs
            .read_dir(&dir)
            .await
            .map_err(|e| SoulError::SkillParse {
                message: format!("Failed to read skills directory {:?}: {e}", self.skills_dir),
            })?;

        let mut count = 0;
        for entry in entries {
            if entry.is_file {
                let ext = entry.name.rsplit('.').next().unwrap_or("");
                if ext == "skill" || ext == "md" {
                    match self.load_file(&entry.name).await {
                        Ok(_) => count += 1,
                        Err(e) => {
                            tracing::warn!("Failed to load skill {:?}: {e}", entry.name);
                        }
                    }
                }
            }
        }

        Ok(count)
    }

    /// Load a single skill file by name.
    pub async fn load_file(&self, filename: &str) -> SoulResult<SkillDefinition> {
        let path = self.file_path(filename);
        let content = self
            .fs
            .read_to_string(&path)
            .await
            .map_err(|e| SoulError::SkillParse {
                message: format!("Failed to read skill file {:?}: {e}", path),
            })?;

        let mut skill = parse_skill(&content)?;
        skill.source_path = Some(std::path::PathBuf::from(&path));
        self.skills.insert(skill.name.clone(), skill.clone());
        Ok(skill)
    }

    /// Get a skill by name.
    pub fn get(&self, name: &str) -> Option<SkillDefinition> {
        self.skills.get(name).map(|r| r.value().clone())
    }

    /// Get all skill names.
    pub fn names(&self) -> Vec<String> {
        self.skills.iter().map(|r| r.key().clone()).collect()
    }

    /// Get all skill definitions.
    pub fn definitions(&self) -> Vec<SkillDefinition> {
        self.skills.iter().map(|r| r.value().clone()).collect()
    }

    /// Number of loaded skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if no skills are loaded.
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    /// Register all loaded skills as tools in a ToolRegistry via SkillToolBridge.
    pub fn register_all_as_tools(
        &self,
        registry: &mut crate::tool::ToolRegistry,
        executor: Arc<dyn super::executor::SkillExecutor>,
    ) {
        for entry in self.skills.iter() {
            let skill = entry.value().clone();
            let bridge = super::bridge::SkillToolBridge::new(skill, executor.clone());
            registry.register(Box::new(bridge));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::MemoryFs;

    async fn write_skill(fs: &Arc<dyn VirtualFs>, dir: &str, filename: &str, content: &str) {
        let path = if dir.is_empty() {
            filename.to_string()
        } else {
            format!("{dir}/{filename}")
        };
        fs.write(&path, content).await.unwrap();
    }

    const SKILL_A: &str = r#"---
name: skill_a
description: Skill A
execution:
  type: shell
  command_template: "echo a"
---
Skill A body
"#;

    const SKILL_B: &str = r#"---
name: skill_b
description: Skill B
execution:
  type: llm_delegate
---
Skill B body
"#;

    #[tokio::test]
    async fn load_all_from_directory() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        write_skill(&fs, "skills", "a.skill", SKILL_A).await;
        write_skill(&fs, "skills", "b.md", SKILL_B).await;
        write_skill(&fs, "skills", "not_a_skill.txt", "ignored").await;

        let loader = SkillLoader::new(fs, "skills");
        let count = loader.load_all().await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(loader.len(), 2);
    }

    #[tokio::test]
    async fn load_single_file() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        write_skill(&fs, "skills", "a.skill", SKILL_A).await;

        let loader = SkillLoader::new(fs, "skills");
        let skill = loader.load_file("a.skill").await.unwrap();
        assert_eq!(skill.name, "skill_a");
        assert!(skill.source_path.is_some());
    }

    #[tokio::test]
    async fn get_by_name() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        write_skill(&fs, "skills", "a.skill", SKILL_A).await;

        let loader = SkillLoader::new(fs, "skills");
        loader.load_all().await.unwrap();

        assert!(loader.get("skill_a").is_some());
        assert!(loader.get("nonexistent").is_none());
    }

    #[tokio::test]
    async fn names_and_definitions() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        write_skill(&fs, "skills", "a.skill", SKILL_A).await;
        write_skill(&fs, "skills", "b.md", SKILL_B).await;

        let loader = SkillLoader::new(fs, "skills");
        loader.load_all().await.unwrap();

        let names = loader.names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"skill_a".to_string()));
        assert!(names.contains(&"skill_b".to_string()));

        let defs = loader.definitions();
        assert_eq!(defs.len(), 2);
    }

    #[tokio::test]
    async fn empty_directory() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        fs.create_dir_all("skills").await.unwrap();
        let loader = SkillLoader::new(fs, "skills");
        let count = loader.load_all().await.unwrap();
        assert_eq!(count, 0);
        assert!(loader.is_empty());
    }

    #[tokio::test]
    async fn nonexistent_directory_errors() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        let loader = SkillLoader::new(fs, "nonexistent");
        let result = loader.load_all().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn malformed_skill_skipped() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        write_skill(&fs, "skills", "good.skill", SKILL_A).await;
        write_skill(&fs, "skills", "bad.skill", "not valid skill content").await;

        let loader = SkillLoader::new(fs, "skills");
        let count = loader.load_all().await.unwrap();
        assert_eq!(count, 1); // Only the good one loaded
    }

    #[tokio::test]
    async fn concurrent_access() {
        let fs: Arc<dyn VirtualFs> = Arc::new(MemoryFs::new());
        write_skill(&fs, "skills", "a.skill", SKILL_A).await;

        let loader = SkillLoader::new(fs, "skills");
        loader.load_all().await.unwrap();

        // DashMap supports concurrent reads
        let skill = loader.get("skill_a");
        assert!(skill.is_some());
    }
}
