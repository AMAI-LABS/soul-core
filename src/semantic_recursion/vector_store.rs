//! In-memory vector store for semantic search.
//! Uses cosine similarity over TF-IDF-like embeddings.
//! No external dependencies — pure Rust computation.

use std::collections::HashMap;

use super::tokenizer::{TokenizedText, Tokenizer};

/// An embedding vector
#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Vec<f32>,
    pub magnitude: f32,
}

impl Embedding {
    pub fn new(vector: Vec<f32>) -> Self {
        let magnitude = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        Self { vector, magnitude }
    }

    /// Cosine similarity between two embeddings
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.magnitude == 0.0 || other.magnitude == 0.0 {
            return 0.0;
        }
        let dot: f32 = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        dot / (self.magnitude * other.magnitude)
    }

    /// Dimensionality
    pub fn dim(&self) -> usize {
        self.vector.len()
    }
}

/// Search result from the vector store
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: usize,
    pub text: String,
    pub score: f32,
    pub metadata: HashMap<String, String>,
}

/// Entry in the vector store
#[derive(Debug, Clone)]
struct VectorEntry {
    id: usize,
    text: String,
    embedding: Embedding,
    metadata: HashMap<String, String>,
}

/// In-memory vector store with TF-IDF embeddings
pub struct VectorStore {
    entries: Vec<VectorEntry>,
    tokenizer: Tokenizer,
    idf: HashMap<u32, f32>,
    next_id: usize,
    total_docs: usize,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            tokenizer: Tokenizer::new(),
            idf: HashMap::new(),
            next_id: 0,
            total_docs: 0,
        }
    }

    /// Insert text into the store
    pub fn insert(&mut self, text: &str, metadata: HashMap<String, String>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.total_docs += 1;

        let tokenized = self.tokenizer.tokenize(text);
        self.update_idf(&tokenized);
        let embedding = self.compute_tfidf_embedding(&tokenized);

        self.entries.push(VectorEntry {
            id,
            text: text.to_string(),
            embedding,
            metadata,
        });

        // Recompute embeddings when IDF changes significantly
        if self.total_docs.is_multiple_of(10) {
            self.reindex();
        }

        id
    }

    /// Search by text query, return top-k results
    pub fn search(&mut self, query: &str, top_k: usize) -> Vec<SearchResult> {
        let tokenized = self.tokenizer.tokenize(query);
        let query_embedding = self.compute_tfidf_embedding(&tokenized);

        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let score = query_embedding.cosine_similarity(&entry.embedding);
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .filter(|(_, score)| *score > 0.0)
            .map(|(idx, score)| {
                let entry = &self.entries[idx];
                SearchResult {
                    id: entry.id,
                    text: entry.text.clone(),
                    score,
                    metadata: entry.metadata.clone(),
                }
            })
            .collect()
    }

    /// Search by pre-computed embedding
    pub fn search_by_embedding(&self, query: &Embedding, top_k: usize) -> Vec<SearchResult> {
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let score = query.cosine_similarity(&entry.embedding);
                (idx, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(top_k)
            .filter(|(_, score)| *score > 0.0)
            .map(|(idx, score)| {
                let entry = &self.entries[idx];
                SearchResult {
                    id: entry.id,
                    text: entry.text.clone(),
                    score,
                    metadata: entry.metadata.clone(),
                }
            })
            .collect()
    }

    /// Get entry by ID
    pub fn get(&self, id: usize) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.text.as_str())
    }

    /// Total number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Vocabulary size of the internal tokenizer
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    /// Recompute all embeddings with current IDF values
    fn reindex(&mut self) {
        // Collect texts first to avoid borrow conflict with &mut self.tokenizer
        let texts: Vec<String> = self.entries.iter().map(|e| e.text.clone()).collect();
        let tokenized: Vec<TokenizedText> =
            texts.iter().map(|t| self.tokenizer.tokenize(t)).collect();
        let embeddings: Vec<Embedding> = tokenized
            .iter()
            .map(|t| self.compute_tfidf_embedding(t))
            .collect();
        for (entry, emb) in self.entries.iter_mut().zip(embeddings) {
            entry.embedding = emb;
        }
    }

    /// Update IDF (inverse document frequency) counts
    fn update_idf(&mut self, tokenized: &TokenizedText) {
        // Count which tokens appear in this document (not frequency, just presence)
        let unique_tokens: std::collections::HashSet<u32> =
            tokenized.token_ids.iter().copied().collect();
        for token_id in unique_tokens {
            *self.idf.entry(token_id).or_insert(0.0) += 1.0;
        }
    }

    /// Compute TF-IDF embedding for tokenized text
    fn compute_tfidf_embedding(&self, tokenized: &TokenizedText) -> Embedding {
        let vocab_size = self.tokenizer.vocab_size().max(1);
        let tf = tokenized.tf_vector(vocab_size);

        let total_docs = self.total_docs.max(1) as f32;
        let mut tfidf = vec![0.0f32; vocab_size];

        for (i, &tf_val) in tf.iter().enumerate() {
            if tf_val > 0.0 {
                let df = self.idf.get(&(i as u32)).copied().unwrap_or(1.0);
                let idf = (total_docs / df).ln() + 1.0;
                tfidf[i] = tf_val * idf;
            }
        }

        Embedding::new(tfidf)
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_cosine_identical() {
        let e = Embedding::new(vec![1.0, 0.0, 0.0]);
        assert!((e.cosine_similarity(&e) - 1.0).abs() < 0.001);
    }

    #[test]
    fn embedding_cosine_orthogonal() {
        let e1 = Embedding::new(vec![1.0, 0.0]);
        let e2 = Embedding::new(vec![0.0, 1.0]);
        assert!(e1.cosine_similarity(&e2).abs() < 0.001);
    }

    #[test]
    fn embedding_cosine_zero() {
        let e1 = Embedding::new(vec![0.0, 0.0]);
        let e2 = Embedding::new(vec![1.0, 1.0]);
        assert_eq!(e1.cosine_similarity(&e2), 0.0);
    }

    #[test]
    fn vector_store_insert_and_search() {
        let mut store = VectorStore::new();
        store.insert("rust programming language", HashMap::new());
        store.insert("python data science", HashMap::new());
        store.insert("rust async tokio runtime", HashMap::new());

        let results = store.search("rust async", 2);
        assert!(!results.is_empty());
        // "rust async tokio runtime" should be most relevant
        assert!(results[0].text.contains("rust"));
    }

    #[test]
    fn vector_store_search_relevance() {
        let mut store = VectorStore::new();
        store.insert("the cat sat on the mat", HashMap::new());
        store.insert("dogs are loyal animals", HashMap::new());
        store.insert("the cat chased the mouse", HashMap::new());

        let results = store.search("cat", 3);
        // Cat-related entries should score higher
        assert!(results.len() >= 2);
        assert!(results[0].text.contains("cat"));
    }

    #[test]
    fn vector_store_with_metadata() {
        let mut store = VectorStore::new();
        let mut meta = HashMap::new();
        meta.insert("source".into(), "file.txt".into());
        meta.insert("line".into(), "42".into());

        let id = store.insert("some text", meta);

        let results = store.search("text", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].metadata["source"], "file.txt");
        assert_eq!(results[0].id, id);
    }

    #[test]
    fn vector_store_get_by_id() {
        let mut store = VectorStore::new();
        let id = store.insert("hello world", HashMap::new());
        assert_eq!(store.get(id), Some("hello world"));
        assert_eq!(store.get(999), None);
    }

    #[test]
    fn vector_store_empty_search() {
        let mut store = VectorStore::new();
        let results = store.search("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn vector_store_len() {
        let mut store = VectorStore::new();
        assert!(store.is_empty());
        store.insert("a", HashMap::new());
        store.insert("b", HashMap::new());
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn vector_store_many_entries() {
        let mut store = VectorStore::new();
        for i in 0..50 {
            store.insert(
                &format!("document number {i} about various topics"),
                HashMap::new(),
            );
        }
        assert_eq!(store.len(), 50);

        let results = store.search("document topics", 5);
        assert!(!results.is_empty());
    }
}
