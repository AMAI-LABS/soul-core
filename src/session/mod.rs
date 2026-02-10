use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::error::SoulResult;
use crate::types::Message;

/// A persistent session — stores conversation transcript
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<Message>,
    #[serde(default)]
    pub metadata: serde_json::Value,
    #[serde(default)]
    pub lane: Option<String>,
}

impl Session {
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            metadata: serde_json::Value::Null,
            lane: None,
        }
    }

    pub fn with_lane(mut self, lane: impl Into<String>) -> Self {
        self.lane = Some(lane.into());
        self
    }

    pub fn append(&mut self, message: Message) {
        self.updated_at = Utc::now();
        self.messages.push(message);
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// JSONL-based session persistence
pub struct SessionStore {
    base_dir: PathBuf,
}

impl SessionStore {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.base_dir.join(format!("{session_id}.jsonl"))
    }

    fn index_path(&self) -> PathBuf {
        self.base_dir.join("sessions.json")
    }

    /// Append a message to the session transcript (JSONL)
    pub async fn append_message(&self, session_id: &str, message: &Message) -> SoulResult<()> {
        tokio::fs::create_dir_all(&self.base_dir).await?;
        let path = self.session_path(session_id);
        let line = serde_json::to_string(message)? + "\n";

        use tokio::io::AsyncWriteExt;
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        file.write_all(line.as_bytes()).await?;
        Ok(())
    }

    /// Load all messages from a session transcript
    pub async fn load_messages(&self, session_id: &str) -> SoulResult<Vec<Message>> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }

        let content = tokio::fs::read_to_string(&path).await?;
        let messages: Vec<Message> = content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        Ok(messages)
    }

    /// Save session metadata
    pub async fn save_session(&self, session: &Session) -> SoulResult<()> {
        tokio::fs::create_dir_all(&self.base_dir).await?;
        let path = self.index_path();

        let mut sessions = self.load_sessions_index().await?;
        sessions.retain(|s: &SessionIndex| s.id != session.id);
        sessions.push(SessionIndex {
            id: session.id.clone(),
            created_at: session.created_at,
            updated_at: session.updated_at,
            message_count: session.message_count(),
            lane: session.lane.clone(),
        });

        let json = serde_json::to_string_pretty(&sessions)?;
        tokio::fs::write(&path, json).await?;
        Ok(())
    }

    /// List all sessions
    pub async fn list_sessions(&self) -> SoulResult<Vec<SessionIndex>> {
        self.load_sessions_index().await
    }

    /// Delete a session
    pub async fn delete_session(&self, session_id: &str) -> SoulResult<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            tokio::fs::remove_file(&path).await?;
        }

        let mut sessions = self.load_sessions_index().await?;
        sessions.retain(|s| s.id != session_id);
        let json = serde_json::to_string_pretty(&sessions)?;
        tokio::fs::write(self.index_path(), json).await?;
        Ok(())
    }

    async fn load_sessions_index(&self) -> SoulResult<Vec<SessionIndex>> {
        let path = self.index_path();
        if !path.exists() {
            return Ok(Vec::new());
        }
        let content = tokio::fs::read_to_string(&path).await?;
        let sessions: Vec<SessionIndex> = serde_json::from_str(&content)?;
        Ok(sessions)
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionIndex {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub message_count: usize,
    #[serde(default)]
    pub lane: Option<String>,
}

/// Lane-based serialization — prevents concurrent execution within a lane
pub struct Lane {
    semaphore: tokio::sync::Semaphore,
}

impl Lane {
    pub fn new() -> Self {
        Self {
            semaphore: tokio::sync::Semaphore::new(1),
        }
    }

    pub async fn acquire(&self) -> tokio::sync::SemaphorePermit<'_> {
        self.semaphore.acquire().await.expect("semaphore closed")
    }
}

impl Default for Lane {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_new() {
        let session = Session::new();
        assert!(!session.id.is_empty());
        assert_eq!(session.messages.len(), 0);
        assert!(session.lane.is_none());
    }

    #[test]
    fn session_with_lane() {
        let session = Session::new().with_lane("main");
        assert_eq!(session.lane, Some("main".to_string()));
    }

    #[test]
    fn session_append() {
        let mut session = Session::new();
        let before = session.updated_at;

        // Small delay so timestamps differ
        std::thread::sleep(std::time::Duration::from_millis(1));

        session.append(Message::user("hello"));
        assert_eq!(session.message_count(), 1);
        assert!(session.updated_at >= before);
    }

    #[test]
    fn session_serializes() {
        let mut session = Session::new();
        session.append(Message::user("hello"));
        session.append(Message::assistant("hi"));

        let json = serde_json::to_string(&session).unwrap();
        let deserialized: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, session.id);
        assert_eq!(deserialized.message_count(), 2);
    }

    #[tokio::test]
    async fn session_store_append_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());

        let session = Session::new();
        let msg1 = Message::user("hello");
        let msg2 = Message::assistant("hi there");

        store.append_message(&session.id, &msg1).await.unwrap();
        store.append_message(&session.id, &msg2).await.unwrap();

        let loaded = store.load_messages(&session.id).await.unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].text_content(), "hello");
        assert_eq!(loaded[1].text_content(), "hi there");
    }

    #[tokio::test]
    async fn session_store_load_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());

        let loaded = store.load_messages("nonexistent").await.unwrap();
        assert!(loaded.is_empty());
    }

    #[tokio::test]
    async fn session_store_save_and_list() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());

        let mut session1 = Session::new();
        session1.append(Message::user("hello"));

        let session2 = Session::new().with_lane("test");

        store.save_session(&session1).await.unwrap();
        store.save_session(&session2).await.unwrap();

        let list = store.list_sessions().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    #[tokio::test]
    async fn session_store_delete() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());

        let session = Session::new();
        let msg = Message::user("hello");
        store.append_message(&session.id, &msg).await.unwrap();
        store.save_session(&session).await.unwrap();

        store.delete_session(&session.id).await.unwrap();

        let loaded = store.load_messages(&session.id).await.unwrap();
        assert!(loaded.is_empty());

        let list = store.list_sessions().await.unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn lane_serialization() {
        let lane = Lane::new();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let c1 = counter.clone();
        let c2 = counter.clone();

        let permit = lane.acquire().await;
        c1.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        drop(permit);

        let permit = lane.acquire().await;
        c2.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        drop(permit);

        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), 2);
    }
}
