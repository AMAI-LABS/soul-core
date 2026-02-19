//! Simple tokenizer for embedding generation.
//! Uses whitespace + punctuation splitting for fast, deterministic tokenization.
//! No external model dependency — pure Rust computation.

use std::collections::HashMap;

/// A single token
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
    pub text: String,
    pub id: u32,
}

/// Tokenized text with frequency information
#[derive(Debug, Clone)]
pub struct TokenizedText {
    pub tokens: Vec<Token>,
    pub token_ids: Vec<u32>,
    pub frequencies: HashMap<u32, usize>,
}

impl TokenizedText {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Generate a TF (term frequency) vector — used for TF-IDF embedding
    pub fn tf_vector(&self, vocab_size: usize) -> Vec<f32> {
        let mut vec = vec![0.0f32; vocab_size];
        let total = self.tokens.len() as f32;
        if total == 0.0 {
            return vec;
        }
        for (&id, &count) in &self.frequencies {
            if (id as usize) < vocab_size {
                vec[id as usize] = count as f32 / total;
            }
        }
        vec
    }
}

/// Tokenizer with vocabulary management
pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    reverse_vocab: HashMap<u32, String>,
    next_id: u32,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            reverse_vocab: HashMap::new(),
            next_id: 0,
        }
    }

    /// Tokenize text into tokens, building vocabulary as needed
    pub fn tokenize(&mut self, text: &str) -> TokenizedText {
        let words = Self::split_words(text);
        let mut tokens = Vec::new();
        let mut token_ids = Vec::new();
        let mut frequencies: HashMap<u32, usize> = HashMap::new();

        for word in words {
            let normalized = word.to_lowercase();
            if normalized.is_empty() {
                continue;
            }

            let id = if let Some(&id) = self.vocab.get(&normalized) {
                id
            } else {
                let id = self.next_id;
                self.vocab.insert(normalized.clone(), id);
                self.reverse_vocab.insert(id, normalized.clone());
                self.next_id += 1;
                id
            };

            tokens.push(Token {
                text: normalized,
                id,
            });
            token_ids.push(id);
            *frequencies.entry(id).or_insert(0) += 1;
        }

        TokenizedText {
            tokens,
            token_ids,
            frequencies,
        }
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Look up a token by ID
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.reverse_vocab.get(&id).map(|s| s.as_str())
    }

    /// Split text into words on whitespace and punctuation boundaries
    fn split_words(text: &str) -> Vec<&str> {
        let mut words = Vec::new();
        let mut start: Option<usize> = None;

        for (i, c) in text.char_indices() {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                if start.is_none() {
                    start = Some(i);
                }
            } else if let Some(s) = start {
                words.push(&text[s..i]);
                start = None;
            }
        }
        if let Some(s) = start {
            words.push(&text[s..]);
        }
        words
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_simple() {
        let mut t = Tokenizer::new();
        let result = t.tokenize("Hello World");
        assert_eq!(result.len(), 2);
        assert_eq!(result.tokens[0].text, "hello");
        assert_eq!(result.tokens[1].text, "world");
    }

    #[test]
    fn tokenize_with_punctuation() {
        let mut t = Tokenizer::new();
        let result = t.tokenize("Hello, world! How are you?");
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn tokenize_empty() {
        let mut t = Tokenizer::new();
        let result = t.tokenize("");
        assert!(result.is_empty());
    }

    #[test]
    fn vocabulary_builds() {
        let mut t = Tokenizer::new();
        t.tokenize("hello world");
        assert_eq!(t.vocab_size(), 2);
        t.tokenize("hello rust");
        assert_eq!(t.vocab_size(), 3); // "hello" already exists
    }

    #[test]
    fn same_word_same_id() {
        let mut t = Tokenizer::new();
        let r1 = t.tokenize("hello");
        let r2 = t.tokenize("Hello"); // case-insensitive
        assert_eq!(r1.token_ids[0], r2.token_ids[0]);
    }

    #[test]
    fn frequencies_correct() {
        let mut t = Tokenizer::new();
        let result = t.tokenize("the cat sat on the mat the");
        assert_eq!(result.frequencies[&result.tokens[0].id], 3); // "the" appears 3 times
    }

    #[test]
    fn tf_vector() {
        let mut t = Tokenizer::new();
        let result = t.tokenize("hello world hello");
        let tf = result.tf_vector(t.vocab_size());
        assert_eq!(tf.len(), 2);
        // "hello" = 2/3 frequency, "world" = 1/3
        assert!((tf[0] - 2.0 / 3.0).abs() < 0.01);
        assert!((tf[1] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn reverse_vocab() {
        let mut t = Tokenizer::new();
        t.tokenize("hello world");
        assert_eq!(t.token_text(0), Some("hello"));
        assert_eq!(t.token_text(1), Some("world"));
        assert_eq!(t.token_text(999), None);
    }
}
