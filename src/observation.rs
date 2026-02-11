//! Cross-session observation storage (L3 memory layer).
//!
//! Observations are semantic records of what happened in a session: bug fixes,
//! features, discoveries, decisions. They persist across sessions via a
//! VirtualFs-backed JSONL file and can be injected into new sessions as
//! context blocks.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::SoulResult;
use crate::vfs::VirtualFs;

// ─── ObservationKind ─────────────────────────────────────────────────────────

/// Semantic classification of what was accomplished in an observation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObservationKind {
    BugFix,
    Feature,
    Refactor,
    Change,
    /// Gotchas, learnings, "how it works".
    Discovery,
    /// Architectural/design rationale.
    Decision,
}

impl ObservationKind {
    pub fn label(&self) -> &'static str {
        match self {
            ObservationKind::BugFix => "Bug Fix",
            ObservationKind::Feature => "Feature",
            ObservationKind::Refactor => "Refactor",
            ObservationKind::Change => "Change",
            ObservationKind::Discovery => "Discovery",
            ObservationKind::Decision => "Decision",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            ObservationKind::BugFix => "🔴",
            ObservationKind::Feature => "🟣",
            ObservationKind::Refactor => "🔄",
            ObservationKind::Change => "✅",
            ObservationKind::Discovery => "🔵",
            ObservationKind::Decision => "⚖️",
        }
    }
}

impl std::fmt::Display for ObservationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ─── Observation ─────────────────────────────────────────────────────────────

/// A single observed outcome from a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub id: String,
    pub session_id: String,
    pub project: String,
    pub kind: ObservationKind,
    pub title: String,
    /// Significance and context of this observation.
    pub narrative: String,
    pub facts: Vec<String>,
    pub files_modified: Vec<String>,
    pub concepts: Vec<String>,
    pub timestamp: DateTime<Utc>,
}

impl Observation {
    pub fn new(
        session_id: impl Into<String>,
        project: impl Into<String>,
        kind: ObservationKind,
        title: impl Into<String>,
        narrative: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.into(),
            project: project.into(),
            kind,
            title: title.into(),
            narrative: narrative.into(),
            facts: Vec::new(),
            files_modified: Vec::new(),
            concepts: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    pub fn with_facts(mut self, facts: Vec<String>) -> Self {
        self.facts = facts;
        self
    }

    pub fn with_files(mut self, files: Vec<String>) -> Self {
        self.files_modified = files;
        self
    }

    pub fn with_concepts(mut self, concepts: Vec<String>) -> Self {
        self.concepts = concepts;
        self
    }

    /// Render this observation as a markdown section.
    pub fn render(&self) -> String {
        let mut out = format!(
            "## {} {}: {}\n{}\n",
            self.kind.emoji(),
            self.kind.label(),
            self.title,
            self.narrative
        );

        if !self.facts.is_empty() {
            out.push_str(&format!("**Facts:** {}\n", self.facts.join(", ")));
        }
        if !self.files_modified.is_empty() {
            out.push_str(&format!("**Files:** {}\n", self.files_modified.join(", ")));
        }
        if !self.concepts.is_empty() {
            out.push_str(&format!("**Concepts:** {}\n", self.concepts.join(", ")));
        }

        out
    }
}

// ─── ObservationStore ────────────────────────────────────────────────────────

/// Append-only store for observations across sessions, backed by VirtualFs JSONL.
pub struct ObservationStore {
    fs: Arc<dyn VirtualFs>,
    path: String,
}

impl ObservationStore {
    pub fn new(fs: Arc<dyn VirtualFs>, path: impl Into<String>) -> Self {
        Self {
            fs,
            path: path.into(),
        }
    }

    /// Append one observation to the log.
    pub async fn save(&self, obs: Observation) -> SoulResult<()> {
        let line = format!("{}\n", serde_json::to_string(&obs)?);
        self.fs.append(&self.path, &line).await
    }

    /// Load all observations for a specific project.
    pub async fn load_project(&self, project: &str) -> SoulResult<Vec<Observation>> {
        let all = self.read_all().await?;
        Ok(all.into_iter().filter(|o| o.project == project).collect())
    }

    /// Load all observations for a specific session.
    pub async fn load_session(&self, session_id: &str) -> SoulResult<Vec<Observation>> {
        let all = self.read_all().await?;
        Ok(all
            .into_iter()
            .filter(|o| o.session_id == session_id)
            .collect())
    }

    /// Load the N most recent observations, optionally filtered by project.
    pub async fn load_recent(
        &self,
        limit: usize,
        project: Option<&str>,
    ) -> SoulResult<Vec<Observation>> {
        let mut all = self.read_all().await?;

        if let Some(proj) = project {
            all.retain(|o| o.project == proj);
        }

        // Sort descending by timestamp (most recent first).
        all.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        all.truncate(limit);
        Ok(all)
    }

    /// Load observations of a specific kind, optionally filtered by project.
    pub async fn load_by_kind(
        &self,
        kind: &ObservationKind,
        project: Option<&str>,
    ) -> SoulResult<Vec<Observation>> {
        let all = self.read_all().await?;
        Ok(all
            .into_iter()
            .filter(|o| o.kind == *kind && project.map_or(true, |p| o.project == p))
            .collect())
    }

    /// Search observations by keyword (case-insensitive in title + narrative + facts).
    pub async fn search(&self, query: &str, project: Option<&str>) -> SoulResult<Vec<Observation>> {
        let needle = query.to_lowercase();
        let all = self.read_all().await?;
        Ok(all
            .into_iter()
            .filter(|o| {
                let project_match = project.map_or(true, |p| o.project == p);
                if !project_match {
                    return false;
                }
                let in_title = o.title.to_lowercase().contains(&needle);
                let in_narrative = o.narrative.to_lowercase().contains(&needle);
                let in_facts = o.facts.iter().any(|f| f.to_lowercase().contains(&needle));
                in_title || in_narrative || in_facts
            })
            .collect())
    }

    /// Build a context block string for injection into session start.
    /// Returns empty string if no observations found.
    pub async fn build_context_block(&self, project: &str, limit: usize) -> SoulResult<String> {
        let recent = self.load_recent(limit, Some(project)).await?;
        if recent.is_empty() {
            return Ok(String::new());
        }

        let mut block = "## Previous Session Observations\n\n".to_string();
        for obs in &recent {
            block.push_str(&obs.render());
            block.push('\n');
        }
        Ok(block)
    }

    /// Count total observations.
    pub async fn count(&self) -> SoulResult<usize> {
        Ok(self.read_all().await?.len())
    }

    /// Read all observations from JSONL (private).
    async fn read_all(&self) -> SoulResult<Vec<Observation>> {
        let exists = self.fs.exists(&self.path).await?;
        if !exists {
            return Ok(Vec::new());
        }

        let content = self.fs.read_to_string(&self.path).await?;
        let mut observations = Vec::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let obs: Observation = serde_json::from_str(trimmed)?;
            observations.push(obs);
        }

        Ok(observations)
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::MemoryFs;

    fn make_store() -> ObservationStore {
        let fs = Arc::new(MemoryFs::new());
        ObservationStore::new(fs, "observations.jsonl")
    }

    fn obs(session: &str, project: &str, kind: ObservationKind, title: &str) -> Observation {
        Observation::new(session, project, kind, title, "some narrative")
    }

    #[tokio::test]
    async fn save_and_load_project() {
        let store = make_store();
        store
            .save(obs("s1", "proj-a", ObservationKind::Feature, "add login"))
            .await
            .unwrap();
        store
            .save(obs("s1", "proj-b", ObservationKind::BugFix, "fix crash"))
            .await
            .unwrap();
        store
            .save(obs(
                "s2",
                "proj-a",
                ObservationKind::Discovery,
                "found gotcha",
            ))
            .await
            .unwrap();

        let results = store.load_project("proj-a").await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|o| o.project == "proj-a"));
    }

    #[tokio::test]
    async fn load_session_filtered() {
        let store = make_store();
        store
            .save(obs("session-1", "proj", ObservationKind::Change, "thing 1"))
            .await
            .unwrap();
        store
            .save(obs("session-2", "proj", ObservationKind::Change, "thing 2"))
            .await
            .unwrap();
        store
            .save(obs(
                "session-1",
                "proj",
                ObservationKind::Feature,
                "thing 3",
            ))
            .await
            .unwrap();

        let results = store.load_session("session-1").await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|o| o.session_id == "session-1"));
    }

    #[tokio::test]
    async fn load_recent_limit() {
        let store = make_store();
        for i in 0..5 {
            store
                .save(obs(
                    "s",
                    "proj",
                    ObservationKind::Change,
                    &format!("obs {i}"),
                ))
                .await
                .unwrap();
        }

        let results = store.load_recent(3, None).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn load_recent_project_filter() {
        let store = make_store();
        store
            .save(obs("s", "alpha", ObservationKind::Feature, "alpha feature"))
            .await
            .unwrap();
        store
            .save(obs("s", "beta", ObservationKind::Feature, "beta feature"))
            .await
            .unwrap();
        store
            .save(obs("s", "alpha", ObservationKind::BugFix, "alpha fix"))
            .await
            .unwrap();

        let results = store.load_recent(10, Some("alpha")).await.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|o| o.project == "alpha"));
    }

    #[tokio::test]
    async fn load_recent_most_recent_first() {
        let store = make_store();
        // Save multiple with distinct timestamps by saving sequentially.
        store
            .save(obs("s", "proj", ObservationKind::Change, "first"))
            .await
            .unwrap();
        // Sleep not needed — UUIDs differ and JSONL order is deterministic,
        // but we rely on sort by timestamp. Use explicit timestamps via direct
        // construction to make test deterministic.
        let mut early = obs("s", "proj", ObservationKind::Change, "early");
        early.timestamp = DateTime::from_timestamp(1_000_000, 0).unwrap();
        let mut late = obs("s", "proj", ObservationKind::Change, "late");
        late.timestamp = DateTime::from_timestamp(2_000_000, 0).unwrap();

        let store2 = make_store();
        store2.save(early).await.unwrap();
        store2.save(late).await.unwrap();

        let results = store2.load_recent(2, None).await.unwrap();
        assert_eq!(results[0].title, "late");
        assert_eq!(results[1].title, "early");
    }

    #[tokio::test]
    async fn load_by_kind() {
        let store = make_store();
        store
            .save(obs("s", "proj", ObservationKind::BugFix, "fix 1"))
            .await
            .unwrap();
        store
            .save(obs("s", "proj", ObservationKind::Feature, "feat 1"))
            .await
            .unwrap();
        store
            .save(obs("s", "proj", ObservationKind::BugFix, "fix 2"))
            .await
            .unwrap();

        let results = store
            .load_by_kind(&ObservationKind::BugFix, None)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|o| o.kind == ObservationKind::BugFix));
    }

    #[tokio::test]
    async fn load_by_kind_project_filter() {
        let store = make_store();
        store
            .save(obs("s", "proj-a", ObservationKind::BugFix, "fix a"))
            .await
            .unwrap();
        store
            .save(obs("s", "proj-b", ObservationKind::BugFix, "fix b"))
            .await
            .unwrap();

        let results = store
            .load_by_kind(&ObservationKind::BugFix, Some("proj-a"))
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project, "proj-a");
    }

    #[tokio::test]
    async fn search_by_keyword_title() {
        let store = make_store();
        store
            .save(obs(
                "s",
                "proj",
                ObservationKind::Feature,
                "websocket handler",
            ))
            .await
            .unwrap();
        store
            .save(obs(
                "s",
                "proj",
                ObservationKind::Feature,
                "database migrations",
            ))
            .await
            .unwrap();

        let results = store.search("websocket", None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "websocket handler");
    }

    #[tokio::test]
    async fn search_by_keyword_narrative() {
        let store = make_store();
        let mut o = obs("s", "proj", ObservationKind::Discovery, "some title");
        o.narrative = "discovered that tokio channels deadlock under pressure".into();
        store.save(o).await.unwrap();
        store
            .save(obs("s", "proj", ObservationKind::Change, "unrelated"))
            .await
            .unwrap();

        let results = store.search("deadlock", None).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "some title");
    }

    #[tokio::test]
    async fn search_case_insensitive() {
        let store = make_store();
        store
            .save(obs(
                "s",
                "proj",
                ObservationKind::Feature,
                "UPPERCASE TITLE",
            ))
            .await
            .unwrap();

        let results = store.search("uppercase", None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn search_project_filter() {
        let store = make_store();
        store
            .save(obs("s", "proj-x", ObservationKind::Feature, "auth module"))
            .await
            .unwrap();
        store
            .save(obs("s", "proj-y", ObservationKind::Feature, "auth module"))
            .await
            .unwrap();

        let results = store.search("auth", Some("proj-x")).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].project, "proj-x");
    }

    #[tokio::test]
    async fn count_observations() {
        let store = make_store();
        assert_eq!(store.count().await.unwrap(), 0);

        store
            .save(obs("s", "proj", ObservationKind::Change, "one"))
            .await
            .unwrap();
        store
            .save(obs("s", "proj", ObservationKind::Change, "two"))
            .await
            .unwrap();

        assert_eq!(store.count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn build_context_block_empty() {
        let store = make_store();
        let block = store
            .build_context_block("no-such-project", 10)
            .await
            .unwrap();
        assert_eq!(block, "");
    }

    #[tokio::test]
    async fn build_context_block_with_data() {
        let store = make_store();
        store
            .save(obs(
                "s",
                "myproj",
                ObservationKind::Discovery,
                "found a gotcha",
            ))
            .await
            .unwrap();

        let block = store.build_context_block("myproj", 10).await.unwrap();
        assert!(block.contains("## Previous Session Observations"));
        assert!(block.contains("found a gotcha"));
        assert!(block.contains("Discovery"));
    }

    #[tokio::test]
    async fn observation_render_contains_kind() {
        let obs = Observation::new(
            "s",
            "proj",
            ObservationKind::BugFix,
            "null pointer",
            "fixed a null deref in handler",
        );
        let rendered = obs.render();
        assert!(rendered.contains("Bug Fix"));
        assert!(rendered.contains("🔴"));
        assert!(rendered.contains("null pointer"));
        assert!(rendered.contains("fixed a null deref"));
    }

    #[tokio::test]
    async fn observation_new_has_id() {
        let o1 = Observation::new("s", "p", ObservationKind::Change, "t", "n");
        let o2 = Observation::new("s", "p", ObservationKind::Change, "t", "n");
        assert!(!o1.id.is_empty());
        assert_ne!(o1.id, o2.id);
    }

    #[tokio::test]
    async fn builder_pattern_with_facts() {
        let obs = Observation::new(
            "s",
            "proj",
            ObservationKind::Feature,
            "new endpoint",
            "added POST /users",
        )
        .with_facts(vec!["uses Ed25519 auth".into(), "rate-limited".into()])
        .with_files(vec!["src/routes.rs".into()])
        .with_concepts(vec!["authentication".into()]);

        let rendered = obs.render();
        assert!(rendered.contains("**Facts:** uses Ed25519 auth, rate-limited"));
        assert!(rendered.contains("**Files:** src/routes.rs"));
        assert!(rendered.contains("**Concepts:** authentication"));
        assert_eq!(obs.facts.len(), 2);
        assert_eq!(obs.files_modified.len(), 1);
        assert_eq!(obs.concepts.len(), 1);
    }

    #[tokio::test]
    async fn kind_labels_and_emojis() {
        let cases = vec![
            (ObservationKind::BugFix, "Bug Fix", "🔴"),
            (ObservationKind::Feature, "Feature", "🟣"),
            (ObservationKind::Refactor, "Refactor", "🔄"),
            (ObservationKind::Change, "Change", "✅"),
            (ObservationKind::Discovery, "Discovery", "🔵"),
            (ObservationKind::Decision, "Decision", "⚖️"),
        ];
        for (kind, label, emoji) in cases {
            assert_eq!(kind.label(), label);
            assert_eq!(kind.emoji(), emoji);
            assert_eq!(kind.to_string(), label);
        }
    }

    #[tokio::test]
    async fn search_by_keyword_facts() {
        let store = make_store();
        let obs = Observation::new("s", "proj", ObservationKind::Discovery, "gotcha", "context")
            .with_facts(vec!["axum 0.7 uses :id not {id}".into()]);
        store.save(obs).await.unwrap();

        let results = store.search("axum", None).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
