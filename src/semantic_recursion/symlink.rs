//! Context Symlinks — compact references to full content.
//!
//! Every message gets a 6-char hash (from SHA-256 of content).
//! The message is truncated to a summary line with the hash as reference.
//! LLM sees `[A3F2B1]: User asked about Rust async patterns` instead of the full message.
//! LLM can call `resolve_symlink` tool to expand any hash back to full content.
//! Nothing is lost — symlinks are bidirectional pointers into the graph.

use std::collections::HashMap;

use sha2::{Digest, Sha256};

use super::graph::NodeId;

/// A single symlink: hash → original content + summary
#[derive(Debug, Clone)]
pub struct Symlink {
    /// 6-char uppercase hex hash (e.g. "A3F2B1")
    pub hash: String,
    /// Graph node this symlink points to
    pub node_id: NodeId,
    /// One-line summary of the content
    pub summary: String,
    /// Full original content (never lost)
    pub original: String,
    /// Token count of the original
    pub original_tokens: usize,
    /// Token count of the symlinked form
    pub symlink_tokens: usize,
}

/// Result of resolving one or more symlinks
#[derive(Debug, Clone)]
pub struct ResolveResult {
    pub hash: String,
    pub original: String,
    pub node_id: NodeId,
    pub summary: String,
}

/// The symlink store — maps hashes to full content via the graph
pub struct SymlinkStore {
    /// hash → Symlink
    links: HashMap<String, Symlink>,
    /// node_id → hash (reverse lookup)
    node_to_hash: HashMap<NodeId, String>,
    /// Collision counter for hash uniqueness
    collision_salt: u64,
}

impl SymlinkStore {
    pub fn new() -> Self {
        Self {
            links: HashMap::new(),
            node_to_hash: HashMap::new(),
            collision_salt: 0,
        }
    }

    /// Create a symlink for a piece of content.
    /// Returns the 6-char hash.
    pub fn create(
        &mut self,
        node_id: NodeId,
        content: &str,
        summary: &str,
    ) -> String {
        // Check if already symlinked
        if let Some(hash) = self.node_to_hash.get(&node_id) {
            return hash.clone();
        }

        let hash = self.generate_hash(content);
        let original_tokens = (content.len() + 3) / 4;
        let symlink_form = format!("[{}]: {}", hash, summary);
        let symlink_tokens = (symlink_form.len() + 3) / 4;

        let symlink = Symlink {
            hash: hash.clone(),
            node_id,
            summary: summary.to_string(),
            original: content.to_string(),
            original_tokens,
            symlink_tokens,
        };

        self.links.insert(hash.clone(), symlink);
        self.node_to_hash.insert(node_id, hash.clone());

        hash
    }

    /// Resolve a hash back to full content
    pub fn resolve(&self, hash: &str) -> Option<ResolveResult> {
        let normalized = hash.to_uppercase();
        self.links.get(&normalized).map(|link| ResolveResult {
            hash: link.hash.clone(),
            original: link.original.clone(),
            node_id: link.node_id,
            summary: link.summary.clone(),
        })
    }

    /// Resolve multiple hashes at once
    pub fn resolve_batch(&self, hashes: &[&str]) -> Vec<ResolveResult> {
        hashes
            .iter()
            .filter_map(|h| self.resolve(h))
            .collect()
    }

    /// Get the symlink hash for a node ID (if it exists)
    pub fn hash_for_node(&self, node_id: NodeId) -> Option<&str> {
        self.node_to_hash.get(&node_id).map(|s| s.as_str())
    }

    /// Get all symlinks
    pub fn all_symlinks(&self) -> Vec<&Symlink> {
        self.links.values().collect()
    }

    /// Format content as a symlink reference line
    pub fn format_ref(&self, hash: &str) -> Option<String> {
        self.links
            .get(&hash.to_uppercase())
            .map(|link| format!("[{}]: {}", link.hash, link.summary))
    }

    /// How many tokens saved by symlinking a particular node
    pub fn tokens_saved(&self, hash: &str) -> Option<usize> {
        self.links.get(&hash.to_uppercase()).map(|link| {
            link.original_tokens.saturating_sub(link.symlink_tokens)
        })
    }

    /// Total tokens saved across all symlinks
    pub fn total_tokens_saved(&self) -> usize {
        self.links
            .values()
            .map(|link| link.original_tokens.saturating_sub(link.symlink_tokens))
            .sum()
    }

    /// Total number of symlinks
    pub fn len(&self) -> usize {
        self.links.len()
    }

    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }

    /// Search symlinks by summary text (keyword match)
    pub fn search(&self, query: &str) -> Vec<&Symlink> {
        let query_lower = query.to_lowercase();
        self.links
            .values()
            .filter(|link| {
                link.summary.to_lowercase().contains(&query_lower)
                    || link.original.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Extract all symlink hashes referenced in a text (pattern: [XXXXXX])
    pub fn extract_refs(text: &str) -> Vec<String> {
        let mut refs = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'[' && i + 7 < bytes.len() && bytes[i + 7] == b']' {
                let candidate = &text[i + 1..i + 7];
                if candidate.chars().all(|c| c.is_ascii_hexdigit()) {
                    refs.push(candidate.to_uppercase());
                    i += 8;
                    continue;
                }
            }
            i += 1;
        }
        refs
    }

    /// Inline-expand all symlink references in a text
    pub fn expand_refs(&self, text: &str) -> String {
        let refs = Self::extract_refs(text);
        if refs.is_empty() {
            return text.to_string();
        }

        let mut result = text.to_string();
        for hash in refs {
            if let Some(link) = self.links.get(&hash) {
                let pattern = format!("[{}]", hash);
                result = result.replace(&pattern, &link.original);
            }
        }
        result
    }

    // ─── Internal ───────────────────────────────────────────────────────────

    /// Generate a unique 6-char hex hash from content
    fn generate_hash(&mut self, content: &str) -> String {
        loop {
            let mut hasher = Sha256::new();
            hasher.update(content.as_bytes());
            if self.collision_salt > 0 {
                hasher.update(self.collision_salt.to_le_bytes());
            }
            let digest = hasher.finalize();
            // Take first 3 bytes = 6 hex chars
            let hash = format!(
                "{:02X}{:02X}{:02X}",
                digest[0], digest[1], digest[2]
            );

            if !self.links.contains_key(&hash) {
                return hash;
            }

            // Collision — add salt and retry
            self.collision_salt += 1;
        }
    }
}

impl Default for SymlinkStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a one-line summary of content (truncate intelligently)
pub fn auto_summary(content: &str, max_chars: usize) -> String {
    let content = content.trim();
    if content.is_empty() {
        return "(empty)".to_string();
    }

    // Take first line or first N chars, whichever is shorter
    let first_line = content.lines().next().unwrap_or(content);
    let text = if first_line.len() <= max_chars {
        first_line
    } else {
        content
    };

    if text.len() <= max_chars {
        text.to_string()
    } else {
        // Truncate at word boundary
        let truncated = &text[..max_chars];
        match truncated.rfind(' ') {
            Some(pos) if pos > max_chars / 2 => format!("{}...", &truncated[..pos]),
            _ => format!("{truncated}..."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_symlink() {
        let mut store = SymlinkStore::new();
        let hash = store.create(0, "Hello world, this is a test message", "User greeted");
        assert_eq!(hash.len(), 6);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn create_idempotent_same_node() {
        let mut store = SymlinkStore::new();
        let h1 = store.create(0, "content A", "summary A");
        let h2 = store.create(0, "content A", "summary A");
        assert_eq!(h1, h2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn create_different_nodes_different_hashes() {
        let mut store = SymlinkStore::new();
        let h1 = store.create(0, "content A", "summary A");
        let h2 = store.create(1, "content B", "summary B");
        assert_ne!(h1, h2);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn resolve_existing() {
        let mut store = SymlinkStore::new();
        let hash = store.create(42, "The full original content here", "User asked something");
        let result = store.resolve(&hash).unwrap();
        assert_eq!(result.original, "The full original content here");
        assert_eq!(result.node_id, 42);
        assert_eq!(result.summary, "User asked something");
    }

    #[test]
    fn resolve_case_insensitive() {
        let mut store = SymlinkStore::new();
        let hash = store.create(0, "content", "summary");
        let lower = hash.to_lowercase();
        assert!(store.resolve(&lower).is_some());
    }

    #[test]
    fn resolve_nonexistent() {
        let store = SymlinkStore::new();
        assert!(store.resolve("ABCDEF").is_none());
    }

    #[test]
    fn resolve_batch() {
        let mut store = SymlinkStore::new();
        let h1 = store.create(0, "first content", "first");
        let h2 = store.create(1, "second content", "second");

        let results = store.resolve_batch(&[&h1, &h2, "ZZZZZZ"]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn hash_for_node() {
        let mut store = SymlinkStore::new();
        let hash = store.create(7, "content", "summary");
        assert_eq!(store.hash_for_node(7), Some(hash.as_str()));
        assert_eq!(store.hash_for_node(999), None);
    }

    #[test]
    fn format_ref() {
        let mut store = SymlinkStore::new();
        let hash = store.create(0, "original text", "User asked about Rust");
        let formatted = store.format_ref(&hash).unwrap();
        assert!(formatted.starts_with('['));
        assert!(formatted.contains("User asked about Rust"));
        assert_eq!(formatted, format!("[{hash}]: User asked about Rust"));
    }

    #[test]
    fn tokens_saved() {
        let mut store = SymlinkStore::new();
        let long_content = "x".repeat(1000);
        let hash = store.create(0, &long_content, "short summary");
        let saved = store.tokens_saved(&hash).unwrap();
        assert!(saved > 200); // 1000/4 - ~8 = big savings
    }

    #[test]
    fn total_tokens_saved() {
        let mut store = SymlinkStore::new();
        store.create(0, &"x".repeat(400), "short");
        store.create(1, &"y".repeat(400), "brief");
        let total = store.total_tokens_saved();
        assert!(total > 150);
    }

    #[test]
    fn search_by_summary() {
        let mut store = SymlinkStore::new();
        store.create(0, "Full content about Rust programming", "Rust programming discussion");
        store.create(1, "Python web development guide", "Python web dev");
        store.create(2, "More Rust async patterns", "Rust async patterns");

        let results = store.search("rust");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn search_by_original_content() {
        let mut store = SymlinkStore::new();
        store.create(0, "Contains the word tokio in the body", "Some discussion");

        let results = store.search("tokio");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn extract_refs_from_text() {
        let text = "Earlier we discussed [A3F2B1] and also [00CAFE]. See those for details.";
        let refs = SymlinkStore::extract_refs(text);
        assert_eq!(refs.len(), 2);
        assert!(refs.contains(&"A3F2B1".to_string()));
        assert!(refs.contains(&"00CAFE".to_string()));
    }

    #[test]
    fn extract_refs_no_matches() {
        let text = "No symlinks here, just [short] and [toolong123]";
        let refs = SymlinkStore::extract_refs(text);
        assert!(refs.is_empty());
    }

    #[test]
    fn extract_refs_case_normalized() {
        let text = "Reference [a3f2b1] should be uppercase";
        let refs = SymlinkStore::extract_refs(text);
        assert_eq!(refs, vec!["A3F2B1"]);
    }

    #[test]
    fn expand_refs_in_text() {
        let mut store = SymlinkStore::new();
        let hash = store.create(0, "the full explanation of traits", "traits explanation");
        let text = format!("As discussed in [{hash}], traits are important.");
        let expanded = store.expand_refs(&text);
        assert!(expanded.contains("the full explanation of traits"));
        assert!(!expanded.contains(&hash));
    }

    #[test]
    fn expand_refs_no_refs() {
        let store = SymlinkStore::new();
        let text = "No refs here.";
        assert_eq!(store.expand_refs(text), text);
    }

    #[test]
    fn expand_refs_unknown_hash() {
        let store = SymlinkStore::new();
        let text = "See [FFFFFF] for more.";
        // Unknown hashes stay as-is
        assert_eq!(store.expand_refs(text), text);
    }

    #[test]
    fn auto_summary_short() {
        assert_eq!(auto_summary("Hello world", 100), "Hello world");
    }

    #[test]
    fn auto_summary_truncates() {
        let long = "This is a very long message that should be truncated at a word boundary somewhere reasonable";
        let summary = auto_summary(long, 40);
        assert!(summary.len() <= 44); // 40 + "..."
        assert!(summary.ends_with("..."));
    }

    #[test]
    fn auto_summary_first_line() {
        let multi = "First line here\nSecond line\nThird line";
        assert_eq!(auto_summary(multi, 100), "First line here");
    }

    #[test]
    fn auto_summary_empty() {
        assert_eq!(auto_summary("", 100), "(empty)");
        assert_eq!(auto_summary("   ", 100), "(empty)");
    }

    #[test]
    fn collision_handling() {
        let mut store = SymlinkStore::new();
        // Force many symlinks — even if collision happens, all should be unique
        let mut hashes = Vec::new();
        for i in 0..100 {
            let h = store.create(i, &format!("content number {i}"), &format!("summary {i}"));
            hashes.push(h);
        }
        assert_eq!(store.len(), 100);
        // All hashes unique
        let unique: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), 100);
    }

    #[test]
    fn all_symlinks() {
        let mut store = SymlinkStore::new();
        store.create(0, "a", "s1");
        store.create(1, "b", "s2");
        store.create(2, "c", "s3");
        assert_eq!(store.all_symlinks().len(), 3);
    }

    #[test]
    fn empty_store() {
        let store = SymlinkStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert_eq!(store.total_tokens_saved(), 0);
    }
}
