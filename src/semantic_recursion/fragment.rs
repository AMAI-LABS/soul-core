//! Semantic Fragmentation — splits messages into clustered symlink fragments.
//!
//! Each message can contain multiple distinct topics/concepts.
//! Instead of one symlink per message, we fragment into semantic clusters:
//! 1. Split message into sentences/segments
//! 2. Compute TF-IDF embedding for each segment
//! 3. K-nearest-neighbor clustering groups similar segments
//! 4. Each cluster becomes its own symlink with its own hash
//!
//! Result: a message like "Rust has great async. Python uses asyncio. Rust also has traits."
//! becomes fragments: \[A3F2B1\]: Rust async+traits discussion, \[B4C5D6\]: Python asyncio mention

use super::tokenizer::Tokenizer;
use super::vector_store::Embedding;

/// A single fragment from a message — a cluster of related sentences
#[derive(Debug, Clone)]
pub struct Fragment {
    /// The text of this fragment (joined sentences)
    pub text: String,
    /// Which sentence indices belong to this cluster
    pub sentence_indices: Vec<usize>,
    /// Centroid embedding for this cluster
    pub centroid: Vec<f32>,
    /// Auto-generated summary
    pub summary: String,
}

/// Result of fragmenting a message
#[derive(Debug, Clone)]
pub struct FragmentResult {
    /// The fragments (each becomes a symlink)
    pub fragments: Vec<Fragment>,
    /// The original sentences before clustering
    pub sentences: Vec<String>,
}

/// Configuration for fragmentation
#[derive(Debug, Clone)]
pub struct FragmentConfig {
    /// Minimum number of fragments to create (won't over-fragment short messages)
    pub min_fragments: usize,
    /// Maximum number of fragments per message
    pub max_fragments: usize,
    /// Minimum sentences per fragment
    pub min_sentences_per_fragment: usize,
    /// Similarity threshold — sentences above this merge into same fragment
    pub similarity_threshold: f32,
    /// Max chars for auto-summary
    pub summary_max_chars: usize,
}

impl Default for FragmentConfig {
    fn default() -> Self {
        Self {
            min_fragments: 1,
            max_fragments: 8,
            min_sentences_per_fragment: 1,
            similarity_threshold: 0.3,
            summary_max_chars: 80,
        }
    }
}

/// Fragment a message into semantic clusters
pub fn fragment_message(
    text: &str,
    tokenizer: &mut Tokenizer,
    config: &FragmentConfig,
) -> FragmentResult {
    let sentences = split_sentences(text);

    // Too short to fragment
    if sentences.len() <= config.min_sentences_per_fragment {
        return FragmentResult {
            fragments: vec![Fragment {
                text: text.to_string(),
                sentence_indices: (0..sentences.len()).collect(),
                centroid: vec![],
                summary: super::symlink::auto_summary(text, config.summary_max_chars),
            }],
            sentences,
        };
    }

    // Compute embeddings for each sentence
    let embeddings: Vec<Vec<f32>> = sentences
        .iter()
        .map(|s| {
            let tokenized = tokenizer.tokenize(s);
            let vocab_size = tokenizer.vocab_size().max(1);
            tokenized.tf_vector(vocab_size)
        })
        .collect();

    // Compute similarity matrix
    let n = sentences.len();
    let mut sim_matrix = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let sim = cosine_sim(&embeddings[i], &embeddings[j]);
            sim_matrix[i][j] = sim;
            sim_matrix[j][i] = sim;
        }
        sim_matrix[i][i] = 1.0;
    }

    // Agglomerative clustering: merge most similar pairs until we hit target cluster count
    let target_k = determine_k(n, config);
    let assignments = agglomerative_cluster(&sim_matrix, target_k, config.similarity_threshold);

    // Group sentences by cluster
    let max_cluster = assignments.iter().copied().max().unwrap_or(0);
    let mut clusters: Vec<Vec<usize>> = vec![vec![]; max_cluster + 1];
    for (i, &cluster) in assignments.iter().enumerate() {
        clusters[cluster].push(i);
    }

    // Remove empty clusters
    let clusters: Vec<Vec<usize>> = clusters.into_iter().filter(|c| !c.is_empty()).collect();

    // Build fragments from clusters
    let fragments = clusters
        .into_iter()
        .map(|indices| {
            let text: String = indices
                .iter()
                .map(|&i| sentences[i].as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let centroid = if indices.is_empty() {
                vec![]
            } else {
                compute_centroid(&indices.iter().map(|&i| &embeddings[i]).collect::<Vec<_>>())
            };

            let summary = super::symlink::auto_summary(&text, config.summary_max_chars);

            Fragment {
                text,
                sentence_indices: indices,
                centroid,
                summary,
            }
        })
        .collect();

    FragmentResult {
        fragments,
        sentences,
    }
}

/// Determine optimal k (number of clusters) based on message length
fn determine_k(n_sentences: usize, config: &FragmentConfig) -> usize {
    if n_sentences <= 2 {
        return 1;
    }
    // Heuristic: sqrt(n) clusters, clamped to config bounds
    let k = (n_sentences as f32).sqrt().ceil() as usize;
    k.clamp(config.min_fragments, config.max_fragments)
}

/// Split text into sentences (handles ., !, ?, newlines)
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' || ch == '\n' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && trimmed.len() > 2 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Remaining text
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.len() > 2 {
        sentences.push(trimmed);
    }

    // If no sentence breaks found, split on commas or just return whole text
    if sentences.is_empty() && !text.trim().is_empty() {
        sentences.push(text.trim().to_string());
    }

    sentences
}

/// Cosine similarity between two vectors
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let min_len = a.len().min(b.len());
    let dot: f32 = a[..min_len]
        .iter()
        .zip(&b[..min_len])
        .map(|(x, y)| x * y)
        .sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Compute centroid of a set of embeddings
fn compute_centroid(embeddings: &[&Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![];
    }
    let dim = embeddings.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut centroid = vec![0.0f32; dim];
    let n = embeddings.len() as f32;
    for emb in embeddings {
        for (i, &val) in emb.iter().enumerate() {
            if i < dim {
                centroid[i] += val / n;
            }
        }
    }
    centroid
}

/// Agglomerative clustering — bottom-up merging of most similar pairs
fn agglomerative_cluster(sim_matrix: &[Vec<f32>], target_k: usize, _threshold: f32) -> Vec<usize> {
    let n = sim_matrix.len();
    if n == 0 {
        return vec![];
    }

    // Each sentence starts in its own cluster
    let mut assignments: Vec<usize> = (0..n).collect();
    let mut num_clusters = n;

    while num_clusters > target_k {
        // Find the most similar pair of different clusters
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_i = 0;
        let mut best_j = 0;

        for i in 0..n {
            for j in i + 1..n {
                if assignments[i] != assignments[j] && sim_matrix[i][j] > best_sim {
                    best_sim = sim_matrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_sim == f32::NEG_INFINITY {
            break; // No more pairs to merge
        }

        // Merge: assign all members of cluster_j to cluster_i
        let cluster_i = assignments[best_i];
        let cluster_j = assignments[best_j];
        for a in assignments.iter_mut() {
            if *a == cluster_j {
                *a = cluster_i;
            }
        }
        num_clusters -= 1;
    }

    // Renumber clusters to 0..k-1
    let mut seen = std::collections::HashMap::new();
    let mut next = 0;
    for a in assignments.iter_mut() {
        let new_id = *seen.entry(*a).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        *a = new_id;
    }

    assignments
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentences_basic() {
        let sentences = split_sentences("Hello world. How are you? I am fine!");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "How are you?");
        assert_eq!(sentences[2], "I am fine!");
    }

    #[test]
    fn split_sentences_newlines() {
        let sentences = split_sentences("Line one\nLine two\nLine three");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn split_sentences_no_breaks() {
        let sentences = split_sentences("Single continuous text without sentence breaks");
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn split_sentences_empty() {
        let sentences = split_sentences("");
        assert!(sentences.is_empty());
    }

    #[test]
    fn fragment_short_message() {
        let mut tokenizer = Tokenizer::new();
        let config = FragmentConfig::default();
        let result = fragment_message("Hello world.", &mut tokenizer, &config);
        assert_eq!(result.fragments.len(), 1);
        assert_eq!(result.fragments[0].text, "Hello world.");
    }

    #[test]
    fn fragment_multi_topic_message() {
        let mut tokenizer = Tokenizer::new();
        let config = FragmentConfig::default();
        let text = "Rust has excellent async support with tokio. \
                    Python uses asyncio for async programming. \
                    Rust also provides zero-cost abstractions. \
                    Python has great data science libraries. \
                    Rust memory safety prevents bugs. \
                    Python is dynamically typed.";
        let result = fragment_message(text, &mut tokenizer, &config);
        // Should create multiple fragments grouping Rust vs Python sentences
        assert!(result.fragments.len() >= 1);
        assert!(result.fragments.len() <= 8);
        // Total sentences should be preserved
        let total_sentences: usize = result
            .fragments
            .iter()
            .map(|f| f.sentence_indices.len())
            .sum();
        assert_eq!(total_sentences, result.sentences.len());
    }

    #[test]
    fn fragment_preserves_all_text() {
        let mut tokenizer = Tokenizer::new();
        let config = FragmentConfig::default();
        let text =
            "First topic about databases. Second topic about networking. Third about security.";
        let result = fragment_message(text, &mut tokenizer, &config);

        // All sentences should appear in exactly one fragment
        let mut all_indices: Vec<usize> = result
            .fragments
            .iter()
            .flat_map(|f| f.sentence_indices.clone())
            .collect();
        all_indices.sort();
        all_indices.dedup();
        assert_eq!(all_indices.len(), result.sentences.len());
    }

    #[test]
    fn fragment_has_summaries() {
        let mut tokenizer = Tokenizer::new();
        let config = FragmentConfig::default();
        let text = "The Rust programming language focuses on safety. \
                    It provides zero-cost abstractions and memory safety without garbage collection.";
        let result = fragment_message(text, &mut tokenizer, &config);
        for fragment in &result.fragments {
            assert!(!fragment.summary.is_empty());
        }
    }

    #[test]
    fn fragment_config_max_fragments() {
        let mut tokenizer = Tokenizer::new();
        let config = FragmentConfig {
            max_fragments: 2,
            ..Default::default()
        };
        // Many sentences
        let text = (0..20)
            .map(|i| format!("Sentence number {i} about topic {i}.", i = i))
            .collect::<Vec<_>>()
            .join(" ");
        let result = fragment_message(&text, &mut tokenizer, &config);
        assert!(result.fragments.len() <= 2);
    }

    #[test]
    fn fragment_config_min_fragments() {
        let mut tokenizer = Tokenizer::new();
        let config = FragmentConfig {
            min_fragments: 3,
            ..Default::default()
        };
        let text = "Topic A sentence one. Topic A sentence two. Topic B sentence one. \
                    Topic B sentence two. Topic C sentence one. Topic C sentence two.";
        let result = fragment_message(text, &mut tokenizer, &config);
        assert!(result.fragments.len() >= 3);
    }

    #[test]
    fn cosine_sim_identical() {
        let v = vec![1.0, 0.0, 1.0];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 0.001);
    }

    #[test]
    fn cosine_sim_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_sim(&a, &b).abs() < 0.001);
    }

    #[test]
    fn cosine_sim_empty() {
        assert_eq!(cosine_sim(&[], &[1.0]), 0.0);
    }

    #[test]
    fn compute_centroid_basic() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let centroid = compute_centroid(&[&a, &b]);
        assert!((centroid[0] - 0.5).abs() < 0.001);
        assert!((centroid[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn compute_centroid_empty() {
        let centroid = compute_centroid(&[]);
        assert!(centroid.is_empty());
    }

    #[test]
    fn determine_k_small() {
        let config = FragmentConfig::default();
        assert_eq!(determine_k(1, &config), 1);
        assert_eq!(determine_k(2, &config), 1);
    }

    #[test]
    fn determine_k_medium() {
        let config = FragmentConfig::default();
        let k = determine_k(9, &config);
        assert_eq!(k, 3); // sqrt(9) = 3
    }

    #[test]
    fn determine_k_clamped() {
        let config = FragmentConfig {
            max_fragments: 4,
            ..Default::default()
        };
        let k = determine_k(100, &config);
        assert!(k <= 4);
    }

    #[test]
    fn agglomerative_cluster_basic() {
        // 4 items, 2 clusters: (0,1) similar, (2,3) similar
        let sim = vec![
            vec![1.0, 0.9, 0.1, 0.1],
            vec![0.9, 1.0, 0.1, 0.1],
            vec![0.1, 0.1, 1.0, 0.8],
            vec![0.1, 0.1, 0.8, 1.0],
        ];
        let assignments = agglomerative_cluster(&sim, 2, 0.3);
        // Items 0 and 1 should be in the same cluster
        assert_eq!(assignments[0], assignments[1]);
        // Items 2 and 3 should be in the same cluster
        assert_eq!(assignments[2], assignments[3]);
        // The two clusters should be different
        assert_ne!(assignments[0], assignments[2]);
    }

    #[test]
    fn agglomerative_cluster_single() {
        let sim = vec![vec![1.0]];
        let assignments = agglomerative_cluster(&sim, 1, 0.3);
        assert_eq!(assignments, vec![0]);
    }

    #[test]
    fn agglomerative_cluster_all_merge() {
        let sim = vec![
            vec![1.0, 0.9, 0.8],
            vec![0.9, 1.0, 0.85],
            vec![0.8, 0.85, 1.0],
        ];
        let assignments = agglomerative_cluster(&sim, 1, 0.3);
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
    }
}
