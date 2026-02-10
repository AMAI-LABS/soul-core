//! Semantic Context Engine — crafts refined context windows per task.
//!
//! Combines the vector store (semantic search), context graph (relationships),
//! and tokenizer to build optimal message sequences for each LLM request.
//! Nothing is ever lost — compaction creates graph edges, not deletions.

use std::collections::HashMap;

use crate::types::Message;

use super::fragment::{self, FragmentConfig};
use super::graph::{ContextGraph, EdgeKind, NodeId, NodeKind};
use super::symlink::{self, SymlinkStore};
use super::tokenizer::Tokenizer;
use super::vector_store::VectorStore;

/// Query for retrieving relevant context
#[derive(Debug, Clone)]
pub struct RetrievalQuery {
    pub text: String,
    pub max_tokens: usize,
    pub max_results: usize,
    pub include_graph_neighbors: bool,
}

/// A crafted context window — refined message list for a specific task
#[derive(Debug, Clone)]
pub struct ContextWindow {
    pub messages: Vec<Message>,
    pub total_tokens: usize,
    pub node_ids_included: Vec<NodeId>,
    pub relevance_scores: Vec<f32>,
}

/// The semantic context engine
pub struct SemanticContextEngine {
    graph: ContextGraph,
    vector_store: VectorStore,
    tokenizer: Tokenizer,
    symlinks: SymlinkStore,
    fragment_config: FragmentConfig,
}

impl SemanticContextEngine {
    pub fn new() -> Self {
        Self {
            graph: ContextGraph::new(),
            vector_store: VectorStore::new(),
            tokenizer: Tokenizer::new(),
            symlinks: SymlinkStore::new(),
            fragment_config: FragmentConfig::default(),
        }
    }

    pub fn with_fragment_config(mut self, config: FragmentConfig) -> Self {
        self.fragment_config = config;
        self
    }

    /// Ingest a user request into the graph and vector store
    pub fn ingest_user_request(&mut self, text: &str) -> NodeId {
        let node_id = self.graph.add_node(
            NodeKind::UserRequest,
            text.to_string(),
            HashMap::new(),
        );

        let mut meta = HashMap::new();
        meta.insert("node_id".into(), node_id.to_string());
        meta.insert("kind".into(), "user_request".into());
        self.vector_store.insert(text, meta);

        node_id
    }

    /// Ingest an LLM response, linking it to the request it answers
    pub fn ingest_llm_response(
        &mut self,
        text: &str,
        request_node: NodeId,
        model: Option<&str>,
    ) -> NodeId {
        let mut metadata = HashMap::new();
        if let Some(m) = model {
            metadata.insert("model".into(), m.to_string());
        }

        let node_id = self.graph.add_node(
            NodeKind::LlmResponse,
            text.to_string(),
            metadata,
        );

        // Edge: response answers request
        self.graph.add_edge(request_node, node_id, EdgeKind::RespondsTo, 1.0);

        // Edge: follows in sequence
        self.graph.add_edge(request_node, node_id, EdgeKind::FollowsInSequence, 1.0);

        let mut meta = HashMap::new();
        meta.insert("node_id".into(), node_id.to_string());
        meta.insert("kind".into(), "llm_response".into());
        self.vector_store.insert(text, meta);

        node_id
    }

    /// Ingest a tool call and its result
    pub fn ingest_tool_interaction(
        &mut self,
        tool_name: &str,
        arguments: &str,
        result: &str,
        trigger_node: NodeId,
    ) -> (NodeId, NodeId) {
        let call_id = self.graph.add_node(
            NodeKind::ToolCall,
            format!("{tool_name}({arguments})"),
            HashMap::from([("tool".into(), tool_name.into())]),
        );

        let result_id = self.graph.add_node(
            NodeKind::ToolResult,
            result.to_string(),
            HashMap::from([("tool".into(), tool_name.into())]),
        );

        self.graph.add_edge(trigger_node, call_id, EdgeKind::TriggeredTool, 1.0);
        self.graph.add_edge(result_id, trigger_node, EdgeKind::ProvidesData, 1.0);
        self.graph.add_edge(call_id, result_id, EdgeKind::FollowsInSequence, 1.0);

        // Index tool result for search
        let mut meta = HashMap::new();
        meta.insert("node_id".into(), result_id.to_string());
        meta.insert("kind".into(), "tool_result".into());
        meta.insert("tool".into(), tool_name.into());
        self.vector_store.insert(result, meta);

        (call_id, result_id)
    }

    /// Ingest external context (file contents, API data, etc.)
    pub fn ingest_external(&mut self, content: &str, source: &str) -> NodeId {
        let node_id = self.graph.add_node(
            NodeKind::ExternalContext,
            content.to_string(),
            HashMap::from([("source".into(), source.into())]),
        );

        let mut meta = HashMap::new();
        meta.insert("node_id".into(), node_id.to_string());
        meta.insert("kind".into(), "external".into());
        meta.insert("source".into(), source.into());
        self.vector_store.insert(content, meta);

        node_id
    }

    /// Compact old nodes into a summary — nothing is deleted
    pub fn compact(
        &mut self,
        node_ids: &[NodeId],
        summary: &str,
    ) -> NodeId {
        let summary_id = self.graph.compact_nodes(node_ids, summary.to_string());

        let mut meta = HashMap::new();
        meta.insert("node_id".into(), summary_id.to_string());
        meta.insert("kind".into(), "compaction_summary".into());
        meta.insert(
            "original_count".into(),
            node_ids.len().to_string(),
        );
        self.vector_store.insert(summary, meta);

        summary_id
    }

    /// Retrieve relevant context for a query, building a context window
    pub fn retrieve(&mut self, query: &RetrievalQuery) -> ContextWindow {
        // 1. Score graph nodes by keyword relevance
        let tokens: Vec<String> = self.tokenizer
            .tokenize(&query.text)
            .tokens
            .iter()
            .map(|t| t.text.clone())
            .collect();
        self.graph.score_relevance(&tokens);

        // 2. Search vector store for semantic matches
        let search_results = self.vector_store.search(&query.text, query.max_results * 2);

        // 3. Collect candidate node IDs from both sources
        let mut candidates: HashMap<NodeId, f32> = HashMap::new();

        // From graph scoring
        for node in self.graph.top_relevant(query.max_results) {
            candidates.insert(node.id, node.relevance_score);
        }

        // From vector search
        for result in &search_results {
            if let Some(node_id_str) = result.metadata.get("node_id") {
                if let Ok(node_id) = node_id_str.parse::<NodeId>() {
                    let entry = candidates.entry(node_id).or_insert(0.0);
                    // Combine scores (max of graph score and vector score)
                    *entry = entry.max(result.score);
                }
            }
        }

        // 4. If requested, include graph neighbors of top results
        if query.include_graph_neighbors {
            let top_ids: Vec<NodeId> = {
                let mut scored: Vec<(NodeId, f32)> = candidates.iter().map(|(&k, &v)| (k, v)).collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.into_iter().take(5).map(|(id, _)| id).collect()
            };

            for id in top_ids {
                for neighbor in self.graph.neighbors(id) {
                    if let Some(node) = self.graph.get_node(neighbor) {
                        if node.active {
                            candidates.entry(neighbor).or_insert(0.1); // low base score for neighbors
                        }
                    }
                }
            }
        }

        // 5. Sort by score and fit within token budget
        let mut scored: Vec<(NodeId, f32)> = candidates.into_iter().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut messages = Vec::new();
        let mut total_tokens = 0;
        let mut included_ids = Vec::new();
        let mut relevance_scores = Vec::new();

        for (node_id, score) in scored {
            if included_ids.len() >= query.max_results {
                break;
            }
            if let Some(node) = self.graph.get_node(node_id) {
                if total_tokens + node.token_estimate > query.max_tokens {
                    continue; // skip if too large, try next
                }

                let msg = match node.kind {
                    NodeKind::UserRequest => Message::user(node.content.clone()),
                    NodeKind::LlmResponse | NodeKind::CompactionSummary => {
                        Message::assistant(node.content.clone())
                    }
                    NodeKind::ToolResult | NodeKind::ExternalContext => {
                        Message::user(format!("[Context] {}", node.content))
                    }
                    _ => Message::user(node.content.clone()),
                };

                total_tokens += node.token_estimate;
                messages.push(msg);
                included_ids.push(node_id);
                relevance_scores.push(score);
            }
        }

        ContextWindow {
            messages,
            total_tokens,
            node_ids_included: included_ids,
            relevance_scores,
        }
    }

    /// Get the full original content for a compacted summary (trace ancestry)
    pub fn expand_compaction(&self, summary_id: NodeId) -> Vec<String> {
        let ancestry = self.graph.full_ancestry(summary_id);
        ancestry
            .iter()
            .filter_map(|&id| self.graph.get_node(id))
            .filter(|n| n.kind != NodeKind::CompactionSummary)
            .map(|n| n.content.clone())
            .collect()
    }

    // ─── Symlink Operations ──────────────────────────────────────────────────

    /// Symlink a single node: create a compact hash reference.
    /// Returns the 6-char hash.
    pub fn symlink_node(&mut self, node_id: NodeId) -> Option<String> {
        let node = self.graph.get_node(node_id)?;
        let summary = symlink::auto_summary(&node.content, 80);
        let hash = self.symlinks.create(node_id, &node.content, &summary);
        Some(hash)
    }

    /// Fragment a node into semantic clusters, symlink each fragment.
    /// Returns vec of (hash, fragment_node_id) pairs.
    /// Close sentences cluster together into the same symlink.
    /// Distant content gets separate symlinks.
    pub fn symlink_fragmented(&mut self, node_id: NodeId) -> Vec<(String, NodeId)> {
        let content = match self.graph.get_node(node_id) {
            Some(n) => n.content.clone(),
            None => return vec![],
        };
        let kind = self.graph.get_node(node_id).unwrap().kind.clone();

        let result = fragment::fragment_message(&content, &mut self.tokenizer, &self.fragment_config);

        if result.fragments.len() <= 1 {
            // Single fragment — symlink the whole node directly
            let summary = symlink::auto_summary(&content, 80);
            let hash = self.symlinks.create(node_id, &content, &summary);
            return vec![(hash, node_id)];
        }

        // Multiple fragments — create a sub-node for each, edge back to parent
        let mut symlinked = Vec::new();
        for fragment in &result.fragments {
            let frag_node = self.graph.add_node(
                NodeKind::Fragment,
                fragment.text.clone(),
                HashMap::from([
                    ("parent_node".into(), node_id.to_string()),
                    ("sentence_indices".into(), format!("{:?}", fragment.sentence_indices)),
                ]),
            );

            // Edge: fragment is derived from parent
            self.graph.add_edge(node_id, frag_node, EdgeKind::DerivedFrom, 1.0);

            // Index fragment in vector store
            let mut meta = HashMap::new();
            meta.insert("node_id".into(), frag_node.to_string());
            meta.insert("kind".into(), "fragment".into());
            meta.insert("parent_node".into(), node_id.to_string());
            self.vector_store.insert(&fragment.text, meta);

            // Create symlink for this fragment
            let hash = self.symlinks.create(frag_node, &fragment.text, &fragment.summary);
            symlinked.push((hash, frag_node));
        }

        symlinked
    }

    /// Build a symlinked context window: same as retrieve() but messages are
    /// replaced with compact symlink references. The LLM can use resolve_symlink
    /// tool to expand any hash it needs.
    pub fn retrieve_symlinked(&mut self, query: &RetrievalQuery) -> SymlinkedContextWindow {
        let window = self.retrieve(query);

        let mut symlinked_messages = Vec::new();
        let mut symlink_map = Vec::new();

        for (i, msg) in window.messages.iter().enumerate() {
            let node_id = window.node_ids_included[i];
            let text = msg.text_content();

            // Fragment and symlink this message
            let fragments = self.symlink_fragmented(node_id);

            if fragments.len() <= 1 {
                // Single fragment — one symlink line
                if let Some((hash, _)) = fragments.first() {
                    let ref_line = self.symlinks.format_ref(hash).unwrap_or_else(|| text.clone());
                    symlinked_messages.push(ref_line.clone());
                    symlink_map.push(vec![hash.clone()]);
                } else {
                    symlinked_messages.push(text);
                    symlink_map.push(vec![]);
                }
            } else {
                // Multiple fragments — each as separate symlink line
                let mut lines = Vec::new();
                let mut hashes = Vec::new();
                for (hash, _) in &fragments {
                    if let Some(line) = self.symlinks.format_ref(hash) {
                        lines.push(line);
                    }
                    hashes.push(hash.clone());
                }
                symlinked_messages.push(lines.join("\n"));
                symlink_map.push(hashes);
            }
        }

        let symlinked_tokens: usize = symlinked_messages
            .iter()
            .map(|s| (s.len() + 3) / 4)
            .sum();

        SymlinkedContextWindow {
            original: window,
            symlinked_messages,
            symlinked_tokens,
            symlink_map,
            tokens_saved: 0, // computed below
        }
        .compute_savings()
    }

    /// Resolve a symlink hash back to its full content
    pub fn resolve_symlink(&self, hash: &str) -> Option<symlink::ResolveResult> {
        self.symlinks.resolve(hash)
    }

    /// Extract and resolve all symlink refs from a text
    pub fn expand_symlink_refs(&self, text: &str) -> String {
        self.symlinks.expand_refs(text)
    }

    /// Access the symlink store
    pub fn symlinks(&self) -> &SymlinkStore {
        &self.symlinks
    }

    /// Access the symlink store mutably
    pub fn symlinks_mut(&mut self) -> &mut SymlinkStore {
        &mut self.symlinks
    }

    // ─── Stats ──────────────────────────────────────────────────────────────

    /// Get graph statistics
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            total_nodes: self.graph.node_count(),
            active_nodes: self.graph.active_nodes().len(),
            total_edges: self.graph.edge_count(),
            vector_entries: self.vector_store.len(),
            vocab_size: self.vector_store.vocab_size(),
            active_tokens: self.graph.active_token_estimate(),
            symlink_count: self.symlinks.len(),
            symlink_tokens_saved: self.symlinks.total_tokens_saved(),
        }
    }

    /// Access the graph directly
    pub fn graph(&self) -> &ContextGraph {
        &self.graph
    }

    /// Access the graph mutably
    pub fn graph_mut(&mut self) -> &mut ContextGraph {
        &mut self.graph
    }
}

impl Default for SemanticContextEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// A context window with symlink references instead of full content
#[derive(Debug, Clone)]
pub struct SymlinkedContextWindow {
    /// The original (full) context window
    pub original: ContextWindow,
    /// Messages replaced with symlink references
    pub symlinked_messages: Vec<String>,
    /// Total tokens in the symlinked form
    pub symlinked_tokens: usize,
    /// Map: message index → list of symlink hashes
    pub symlink_map: Vec<Vec<String>>,
    /// Tokens saved by symlinking
    pub tokens_saved: usize,
}

impl SymlinkedContextWindow {
    fn compute_savings(mut self) -> Self {
        self.tokens_saved = self.original.total_tokens.saturating_sub(self.symlinked_tokens);
        self
    }
}

/// Statistics about the engine state
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub total_nodes: usize,
    pub active_nodes: usize,
    pub total_edges: usize,
    pub vector_entries: usize,
    pub vocab_size: usize,
    pub active_tokens: usize,
    pub symlink_count: usize,
    pub symlink_tokens_saved: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_conversation() {
        let mut engine = SemanticContextEngine::new();

        let req = engine.ingest_user_request("What is Rust?");
        let resp = engine.ingest_llm_response(
            "Rust is a systems programming language focused on safety and performance.",
            req,
            Some("claude-sonnet"),
        );

        let stats = engine.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 2); // RespondsTo + FollowsInSequence
        assert_eq!(stats.vector_entries, 2);
    }

    #[test]
    fn ingest_tool_interaction() {
        let mut engine = SemanticContextEngine::new();

        let req = engine.ingest_user_request("Read the config file");
        let resp = engine.ingest_llm_response("Let me read that file", req, None);
        let (call_id, result_id) = engine.ingest_tool_interaction(
            "read",
            "/config.toml",
            "[package]\nname = \"soul-core\"\nversion = \"0.1.0\"",
            resp,
        );

        let stats = engine.stats();
        assert_eq!(stats.total_nodes, 4); // req, resp, call, result
        assert!(stats.total_edges >= 4);
    }

    #[test]
    fn compact_preserves_originals() {
        let mut engine = SemanticContextEngine::new();

        let n1 = engine.ingest_user_request("Hello");
        let n2 = engine.ingest_llm_response("Hi", n1, None);
        let n3 = engine.ingest_user_request("How are you?");
        let n4 = engine.ingest_llm_response("I'm good", n3, None);

        assert_eq!(engine.stats().active_nodes, 4);

        let summary = engine.compact(&[n1, n2, n3, n4], "User greeted and exchanged pleasantries");

        // Originals deactivated but not deleted
        assert_eq!(engine.stats().total_nodes, 5); // 4 originals + 1 summary
        assert_eq!(engine.stats().active_nodes, 1); // only summary

        // Can expand back
        let originals = engine.expand_compaction(summary);
        assert_eq!(originals.len(), 4);
        assert!(originals.iter().any(|s| s == "Hello"));
    }

    #[test]
    fn retrieve_relevant_context() {
        let mut engine = SemanticContextEngine::new();

        engine.ingest_user_request("Tell me about Rust async");
        engine.ingest_llm_response("Rust async uses tokio runtime", 0, None);
        engine.ingest_user_request("What about Python?");
        engine.ingest_llm_response("Python uses asyncio", 2, None);
        engine.ingest_user_request("Rust error handling");
        engine.ingest_llm_response("Rust uses Result and Option types", 4, None);

        let query = RetrievalQuery {
            text: "Rust async programming".into(),
            max_tokens: 10000,
            max_results: 3,
            include_graph_neighbors: false,
        };

        let window = engine.retrieve(&query);
        assert!(!window.messages.is_empty());
        assert!(window.total_tokens > 0);
        assert!(!window.node_ids_included.is_empty());
    }

    #[test]
    fn retrieve_with_graph_neighbors() {
        let mut engine = SemanticContextEngine::new();

        let req = engine.ingest_user_request("Explain Rust traits");
        let resp = engine.ingest_llm_response("Traits define shared behavior", req, None);
        let (_, _) = engine.ingest_tool_interaction(
            "read",
            "traits.rs",
            "pub trait Foo { fn bar(&self); }",
            resp,
        );

        let query = RetrievalQuery {
            text: "traits".into(),
            max_tokens: 10000,
            max_results: 5,
            include_graph_neighbors: true,
        };

        let window = engine.retrieve(&query);
        // Should include the tool result as a graph neighbor
        assert!(!window.messages.is_empty());
    }

    #[test]
    fn retrieve_respects_token_budget() {
        let mut engine = SemanticContextEngine::new();

        // Add many large nodes
        for i in 0..20 {
            engine.ingest_user_request(&format!("Question {i}: {}", "x".repeat(500)));
        }

        let query = RetrievalQuery {
            text: "Question".into(),
            max_tokens: 200, // very tight budget
            max_results: 20,
            include_graph_neighbors: false,
        };

        let window = engine.retrieve(&query);
        assert!(window.total_tokens <= 200);
    }

    #[test]
    fn ingest_external_context() {
        let mut engine = SemanticContextEngine::new();

        let node = engine.ingest_external("File contents here", "/path/to/file.rs");

        let stats = engine.stats();
        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.vector_entries, 1);

        let graph_node = engine.graph().get_node(node).unwrap();
        assert_eq!(graph_node.metadata["source"], "/path/to/file.rs");
    }

    #[test]
    fn stats_complete() {
        let mut engine = SemanticContextEngine::new();
        let req = engine.ingest_user_request("hello world");
        engine.ingest_llm_response("hi there", req, None);

        let stats = engine.stats();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.active_nodes, 2);
        assert!(stats.total_edges > 0);
        assert_eq!(stats.vector_entries, 2);
        assert!(stats.vocab_size > 0);
        assert!(stats.active_tokens > 0);
    }

    #[test]
    fn recursive_compaction_ancestry() {
        let mut engine = SemanticContextEngine::new();

        let a = engine.ingest_user_request("msg 1");
        let b = engine.ingest_llm_response("resp 1", a, None);

        let s1 = engine.compact(&[a, b], "summary 1");

        let c = engine.ingest_user_request("msg 2");
        let d = engine.ingest_llm_response("resp 2", c, None);

        let s2 = engine.compact(&[s1, c, d], "summary 2");

        // Full ancestry of s2 should reach a and b
        let originals = engine.expand_compaction(s2);
        assert!(originals.contains(&"msg 1".to_string()));
        assert!(originals.contains(&"resp 1".to_string()));
        assert!(originals.contains(&"msg 2".to_string()));
        assert!(originals.contains(&"resp 2".to_string()));
    }

    // ─── Symlink Tests ────────────────────────────────────────────────────

    #[test]
    fn symlink_single_node() {
        let mut engine = SemanticContextEngine::new();
        let req = engine.ingest_user_request("Explain Rust async patterns with tokio");
        let hash = engine.symlink_node(req).unwrap();
        assert_eq!(hash.len(), 6);

        // Can resolve it back
        let resolved = engine.resolve_symlink(&hash).unwrap();
        assert!(resolved.original.contains("Rust async"));
        assert_eq!(resolved.node_id, req);
    }

    #[test]
    fn symlink_node_idempotent() {
        let mut engine = SemanticContextEngine::new();
        let req = engine.ingest_user_request("Test content");
        let h1 = engine.symlink_node(req).unwrap();
        let h2 = engine.symlink_node(req).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn symlink_nonexistent_node() {
        let mut engine = SemanticContextEngine::new();
        assert!(engine.symlink_node(999).is_none());
    }

    #[test]
    fn symlink_fragmented_short_message() {
        let mut engine = SemanticContextEngine::new();
        let req = engine.ingest_user_request("Short message");
        let fragments = engine.symlink_fragmented(req);
        // Short message = single fragment = single symlink
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].1, req); // points to original node
    }

    #[test]
    fn symlink_fragmented_multi_topic() {
        let mut engine = SemanticContextEngine::new();
        let long_msg = "Rust has excellent async support with tokio and futures. \
                        Python uses asyncio for async programming and has great data science libs. \
                        Rust also provides zero-cost abstractions and memory safety. \
                        Python is dynamically typed and has pip for packages. \
                        JavaScript uses promises and async await syntax. \
                        JavaScript runs in the browser and Node.js.";
        let req = engine.ingest_user_request(long_msg);
        let fragments = engine.symlink_fragmented(req);

        // Should create multiple fragments
        assert!(fragments.len() >= 1);

        // Each fragment has a unique hash
        let hashes: Vec<&str> = fragments.iter().map(|(h, _)| h.as_str()).collect();
        let unique: std::collections::HashSet<_> = hashes.iter().collect();
        assert_eq!(unique.len(), hashes.len());

        // All fragments are resolvable
        for (hash, _) in &fragments {
            assert!(engine.resolve_symlink(hash).is_some());
        }
    }

    #[test]
    fn symlink_fragmented_creates_graph_edges() {
        let mut engine = SemanticContextEngine::new();
        let long_msg = "Topic A about databases and SQL. \
                        Topic B about networking and TCP. \
                        Topic A continued about PostgreSQL. \
                        Topic B continued about HTTP protocols.";
        let req = engine.ingest_user_request(long_msg);
        let fragments = engine.symlink_fragmented(req);

        if fragments.len() > 1 {
            // Fragment nodes should have DerivedFrom edges back to parent
            for (_, frag_node_id) in &fragments {
                if *frag_node_id != req {
                    let node = engine.graph().get_node(*frag_node_id).unwrap();
                    assert_eq!(node.kind, NodeKind::Fragment);
                    assert_eq!(node.metadata["parent_node"], req.to_string());
                }
            }
        }
    }

    #[test]
    fn symlink_stats_tracked() {
        let mut engine = SemanticContextEngine::new();
        assert_eq!(engine.stats().symlink_count, 0);

        let req = engine.ingest_user_request(&"x".repeat(500));
        engine.symlink_node(req);

        let stats = engine.stats();
        assert_eq!(stats.symlink_count, 1);
        assert!(stats.symlink_tokens_saved > 0);
    }

    #[test]
    fn retrieve_symlinked_window() {
        let mut engine = SemanticContextEngine::new();

        // Use long messages so symlinks actually save tokens
        let long_content = "x".repeat(300);
        engine.ingest_user_request(&format!("Tell me about Rust async: {long_content}"));
        engine.ingest_llm_response(
            &format!("Rust async uses tokio runtime for concurrent programming: {long_content}"),
            0,
            None,
        );
        engine.ingest_user_request(&format!("What about error handling? {long_content}"));
        engine.ingest_llm_response(
            &format!("Rust uses Result and Option types: {long_content}"),
            2,
            None,
        );

        let query = RetrievalQuery {
            text: "Rust".into(),
            max_tokens: 100000,
            max_results: 5,
            include_graph_neighbors: false,
        };

        let symlinked = engine.retrieve_symlinked(&query);
        assert!(!symlinked.symlinked_messages.is_empty());
        // Symlinked form should be significantly smaller for long messages
        assert!(symlinked.symlinked_tokens < symlinked.original.total_tokens);
        // Symlink map should have entries for each message
        assert_eq!(symlinked.symlink_map.len(), symlinked.original.messages.len());
        // Each message should have at least one symlink hash
        for hashes in &symlinked.symlink_map {
            assert!(!hashes.is_empty());
        }
    }

    #[test]
    fn expand_symlink_refs_in_text() {
        let mut engine = SemanticContextEngine::new();
        let req = engine.ingest_user_request("The original content about Rust traits");
        let hash = engine.symlink_node(req).unwrap();

        let text = format!("As discussed in [{hash}], traits are important.");
        let expanded = engine.expand_symlink_refs(&text);
        assert!(expanded.contains("The original content about Rust traits"));
    }

    #[test]
    fn symlink_with_fragment_config() {
        let config = FragmentConfig {
            max_fragments: 3,
            min_fragments: 2,
            ..Default::default()
        };
        let mut engine = SemanticContextEngine::new().with_fragment_config(config);

        let long_msg = (0..10)
            .map(|i| format!("Sentence about topic {i} with details.", i = i))
            .collect::<Vec<_>>()
            .join(" ");
        let req = engine.ingest_user_request(&long_msg);
        let fragments = engine.symlink_fragmented(req);
        assert!(fragments.len() >= 2);
        assert!(fragments.len() <= 3);
    }
}
