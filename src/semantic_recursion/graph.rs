//! Context Social Graph — relationships between context pieces, requests, and responses.
//!
//! Think social graph for LLM context: each piece of context, each user request,
//! each LLM response is a node. Edges represent relationships (derived_from,
//! relevant_to, compacted_from, etc.). Nothing is ever deleted — compaction
//! creates new summary nodes with edges back to originals.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique node identifier
pub type NodeId = u64;

/// Kind of node in the context graph
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    /// User request
    UserRequest,
    /// LLM response
    LlmResponse,
    /// Tool call
    ToolCall,
    /// Tool result
    ToolResult,
    /// System prompt
    SystemPrompt,
    /// Memory file content
    MemoryFile,
    /// Compacted summary of other nodes
    CompactionSummary,
    /// External context (file contents, API responses)
    ExternalContext,
    /// Arbitrary context fragment
    Fragment,
}

/// Kind of edge (relationship) between nodes
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Node B was derived from (summarized/compacted from) Node A
    DerivedFrom,
    /// Node A is relevant to Node B (semantic similarity above threshold)
    RelevantTo,
    /// Node B responds to / answers Node A
    RespondsTo,
    /// Node B references Node A (explicit reference in text)
    References,
    /// Node B follows Node A chronologically in conversation
    FollowsInSequence,
    /// Node A was compacted into Node B (original → summary)
    CompactedInto,
    /// Node A triggered tool call Node B
    TriggeredTool,
    /// Tool result Node B provides data for Node A
    ProvidesData,
}

/// A node in the context graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub token_estimate: usize,
    pub metadata: HashMap<String, String>,
    /// Whether this node is "active" (included in current context window)
    #[serde(default)]
    pub active: bool,
    /// Relevance score (updated dynamically based on current task)
    #[serde(default)]
    pub relevance_score: f32,
}

/// An edge in the context graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: EdgeKind,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
}

/// The context social graph
pub struct ContextGraph {
    nodes: HashMap<NodeId, GraphNode>,
    edges: Vec<GraphEdge>,
    next_id: NodeId,
    /// Index: node → outgoing edges
    outgoing: HashMap<NodeId, Vec<usize>>,
    /// Index: node → incoming edges
    incoming: HashMap<NodeId, Vec<usize>>,
}

impl ContextGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_id: 0,
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(
        &mut self,
        kind: NodeKind,
        content: String,
        metadata: HashMap<String, String>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;

        let token_estimate = (content.len() + 3) / 4;

        self.nodes.insert(
            id,
            GraphNode {
                id,
                kind,
                content,
                created_at: Utc::now(),
                token_estimate,
                metadata,
                active: true,
                relevance_score: 0.0,
            },
        );

        id
    }

    /// Add an edge between two nodes
    pub fn add_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        kind: EdgeKind,
        weight: f32,
    ) {
        let edge_idx = self.edges.len();
        self.edges.push(GraphEdge {
            from,
            to,
            kind,
            weight,
            created_at: Utc::now(),
        });
        self.outgoing.entry(from).or_default().push(edge_idx);
        self.incoming.entry(to).or_default().push(edge_idx);
    }

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }

    /// Get outgoing edges from a node
    pub fn outgoing_edges(&self, node: NodeId) -> Vec<&GraphEdge> {
        self.outgoing
            .get(&node)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get incoming edges to a node
    pub fn incoming_edges(&self, node: NodeId) -> Vec<&GraphEdge> {
        self.incoming
            .get(&node)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// Get all active nodes
    pub fn active_nodes(&self) -> Vec<&GraphNode> {
        self.nodes.values().filter(|n| n.active).collect()
    }

    /// Get nodes connected to a given node (1-hop neighbors)
    pub fn neighbors(&self, node: NodeId) -> Vec<NodeId> {
        let mut result: Vec<NodeId> = Vec::new();
        for edge in self.outgoing_edges(node) {
            result.push(edge.to);
        }
        for edge in self.incoming_edges(node) {
            result.push(edge.from);
        }
        result.sort();
        result.dedup();
        result
    }

    /// Compact a set of nodes into a summary node.
    /// Original nodes are deactivated but never deleted.
    /// A CompactedInto edge is created from each original to the summary.
    pub fn compact_nodes(
        &mut self,
        node_ids: &[NodeId],
        summary: String,
    ) -> NodeId {
        let summary_id = self.add_node(
            NodeKind::CompactionSummary,
            summary,
            HashMap::new(),
        );

        for &id in node_ids {
            if let Some(node) = self.nodes.get_mut(&id) {
                node.active = false;
            }
            self.add_edge(id, summary_id, EdgeKind::CompactedInto, 1.0);
        }

        summary_id
    }

    /// Find the original nodes that a compaction summary was derived from
    pub fn trace_compaction(&self, summary_id: NodeId) -> Vec<NodeId> {
        self.incoming_edges(summary_id)
            .iter()
            .filter(|e| e.kind == EdgeKind::CompactedInto)
            .map(|e| e.from)
            .collect()
    }

    /// Get the full ancestry chain for a compacted node (recursive)
    pub fn full_ancestry(&self, node_id: NodeId) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut stack = vec![node_id];
        let mut visited = std::collections::HashSet::new();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            let originals = self.trace_compaction(id);
            for &orig in &originals {
                result.push(orig);
                stack.push(orig);
            }
        }

        result
    }

    /// Score nodes by relevance to a query (using simple keyword overlap)
    pub fn score_relevance(&mut self, query_tokens: &[String]) {
        for node in self.nodes.values_mut() {
            let content_lower = node.content.to_lowercase();
            let matches: usize = query_tokens
                .iter()
                .filter(|t| content_lower.contains(&t.to_lowercase()))
                .count();
            let total = query_tokens.len().max(1);
            node.relevance_score = matches as f32 / total as f32;
        }
    }

    /// Get top-k most relevant active nodes
    pub fn top_relevant(&self, k: usize) -> Vec<&GraphNode> {
        let mut active: Vec<&GraphNode> = self.active_nodes();
        active.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        active.into_iter().take(k).collect()
    }

    /// Total number of nodes (including inactive)
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Total number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Total token estimate for active nodes
    pub fn active_token_estimate(&self) -> usize {
        self.active_nodes()
            .iter()
            .map(|n| n.token_estimate)
            .sum()
    }
}

impl Default for ContextGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_node_and_get() {
        let mut graph = ContextGraph::new();
        let id = graph.add_node(NodeKind::UserRequest, "hello".into(), HashMap::new());
        let node = graph.get_node(id).unwrap();
        assert_eq!(node.content, "hello");
        assert_eq!(node.kind, NodeKind::UserRequest);
        assert!(node.active);
    }

    #[test]
    fn add_edge_and_query() {
        let mut graph = ContextGraph::new();
        let a = graph.add_node(NodeKind::UserRequest, "question".into(), HashMap::new());
        let b = graph.add_node(NodeKind::LlmResponse, "answer".into(), HashMap::new());
        graph.add_edge(a, b, EdgeKind::RespondsTo, 1.0);

        let out = graph.outgoing_edges(a);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to, b);
        assert_eq!(out[0].kind, EdgeKind::RespondsTo);

        let inc = graph.incoming_edges(b);
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].from, a);
    }

    #[test]
    fn neighbors() {
        let mut graph = ContextGraph::new();
        let a = graph.add_node(NodeKind::UserRequest, "q".into(), HashMap::new());
        let b = graph.add_node(NodeKind::LlmResponse, "a".into(), HashMap::new());
        let c = graph.add_node(NodeKind::ToolCall, "t".into(), HashMap::new());
        graph.add_edge(a, b, EdgeKind::RespondsTo, 1.0);
        graph.add_edge(b, c, EdgeKind::TriggeredTool, 1.0);

        let neighbors = graph.neighbors(b);
        assert_eq!(neighbors.len(), 2); // a and c
    }

    #[test]
    fn compact_nodes() {
        let mut graph = ContextGraph::new();
        let a = graph.add_node(NodeKind::UserRequest, "msg 1".into(), HashMap::new());
        let b = graph.add_node(NodeKind::LlmResponse, "msg 2".into(), HashMap::new());
        let c = graph.add_node(NodeKind::UserRequest, "msg 3".into(), HashMap::new());

        assert_eq!(graph.active_nodes().len(), 3);

        let summary_id = graph.compact_nodes(&[a, b, c], "Summary of msgs 1-3".into());

        // Originals deactivated
        assert!(!graph.get_node(a).unwrap().active);
        assert!(!graph.get_node(b).unwrap().active);
        assert!(!graph.get_node(c).unwrap().active);

        // Summary active
        assert!(graph.get_node(summary_id).unwrap().active);
        assert_eq!(graph.active_nodes().len(), 1);

        // Edges exist
        let originals = graph.trace_compaction(summary_id);
        assert_eq!(originals.len(), 3);

        // Nothing deleted
        assert_eq!(graph.node_count(), 4);
    }

    #[test]
    fn full_ancestry_recursive() {
        let mut graph = ContextGraph::new();
        let a = graph.add_node(NodeKind::UserRequest, "original 1".into(), HashMap::new());
        let b = graph.add_node(NodeKind::LlmResponse, "original 2".into(), HashMap::new());

        // First compaction
        let summary1 = graph.compact_nodes(&[a, b], "summary 1".into());

        let c = graph.add_node(NodeKind::UserRequest, "original 3".into(), HashMap::new());

        // Second compaction (summary1 + c → summary2)
        let summary2 = graph.compact_nodes(&[summary1, c], "summary 2".into());

        // Full ancestry should include a, b, c, summary1
        let ancestry = graph.full_ancestry(summary2);
        assert!(ancestry.contains(&a));
        assert!(ancestry.contains(&b));
        assert!(ancestry.contains(&c));
        assert!(ancestry.contains(&summary1));
    }

    #[test]
    fn score_relevance() {
        let mut graph = ContextGraph::new();
        graph.add_node(NodeKind::UserRequest, "rust programming language".into(), HashMap::new());
        graph.add_node(NodeKind::LlmResponse, "python is great for data science".into(), HashMap::new());
        graph.add_node(NodeKind::UserRequest, "rust async tokio".into(), HashMap::new());

        graph.score_relevance(&["rust".into(), "async".into()]);

        let top = graph.top_relevant(2);
        // "rust async tokio" should be most relevant (2/2 matches)
        assert!(top[0].content.contains("rust"));
        assert!(top[0].relevance_score > top[1].relevance_score);
    }

    #[test]
    fn active_token_estimate() {
        let mut graph = ContextGraph::new();
        graph.add_node(NodeKind::UserRequest, "hello".into(), HashMap::new()); // ~2 tokens
        graph.add_node(NodeKind::LlmResponse, "world".into(), HashMap::new()); // ~2 tokens

        let estimate = graph.active_token_estimate();
        assert!(estimate > 0);
    }

    #[test]
    fn edge_count() {
        let mut graph = ContextGraph::new();
        let a = graph.add_node(NodeKind::UserRequest, "q".into(), HashMap::new());
        let b = graph.add_node(NodeKind::LlmResponse, "a".into(), HashMap::new());
        graph.add_edge(a, b, EdgeKind::RespondsTo, 1.0);
        graph.add_edge(a, b, EdgeKind::RelevantTo, 0.5);

        assert_eq!(graph.edge_count(), 2);
        assert_eq!(graph.node_count(), 2);
    }

    #[test]
    fn node_metadata() {
        let mut graph = ContextGraph::new();
        let mut meta = HashMap::new();
        meta.insert("model".into(), "claude-sonnet".into());
        meta.insert("turn".into(), "3".into());

        let id = graph.add_node(NodeKind::LlmResponse, "response".into(), meta);
        let node = graph.get_node(id).unwrap();
        assert_eq!(node.metadata["model"], "claude-sonnet");
    }
}
