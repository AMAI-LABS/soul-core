//! Glob-style pattern matching for tool names and argument values.
//!
//! Supports `*` (match any sequence), `?` (match single char), and literal characters.

/// Matches a string against a glob pattern.
///
/// `*` matches zero or more characters, `?` matches exactly one character.
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let pattern_bytes = pattern.as_bytes();
    let name_bytes = name.as_bytes();
    let (plen, nlen) = (pattern_bytes.len(), name_bytes.len());

    // dp[i][j] = pattern[..i] matches name[..j]
    let mut dp = vec![vec![false; nlen + 1]; plen + 1];
    dp[0][0] = true;

    // Leading `*` can match empty
    for i in 1..=plen {
        if pattern_bytes[i - 1] == b'*' {
            dp[i][0] = dp[i - 1][0];
        }
    }

    for i in 1..=plen {
        for j in 1..=nlen {
            match pattern_bytes[i - 1] {
                b'*' => {
                    // * matches zero chars (dp[i-1][j]) or one more char (dp[i][j-1])
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
                }
                b'?' => {
                    dp[i][j] = dp[i - 1][j - 1];
                }
                c => {
                    dp[i][j] = dp[i - 1][j - 1] && c == name_bytes[j - 1];
                }
            }
        }
    }

    dp[plen][nlen]
}

/// Matches a value within a JSON object at a given JSON pointer path against a glob pattern.
///
/// The `path` uses JSON pointer syntax (e.g., `/command`, `/nested/field`).
/// The value at the path is converted to a string and matched against the pattern.
pub fn match_argument(arguments: &serde_json::Value, path: &str, pattern: &str) -> bool {
    match arguments.pointer(path) {
        Some(serde_json::Value::String(s)) => glob_match(pattern, s),
        Some(serde_json::Value::Number(n)) => glob_match(pattern, &n.to_string()),
        Some(serde_json::Value::Bool(b)) => glob_match(pattern, &b.to_string()),
        Some(serde_json::Value::Null) => glob_match(pattern, "null"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Glob matching: exact ─────────────────────────────────────────

    #[test]
    fn exact_match() {
        assert!(glob_match("bash", "bash"));
    }

    #[test]
    fn exact_no_match() {
        assert!(!glob_match("bash", "read"));
    }

    #[test]
    fn empty_pattern_empty_name() {
        assert!(glob_match("", ""));
    }

    #[test]
    fn empty_pattern_nonempty_name() {
        assert!(!glob_match("", "bash"));
    }

    #[test]
    fn nonempty_pattern_empty_name() {
        assert!(!glob_match("bash", ""));
    }

    // ── Glob matching: star ──────────────────────────────────────────

    #[test]
    fn star_matches_everything() {
        assert!(glob_match("*", "anything"));
    }

    #[test]
    fn star_matches_empty() {
        assert!(glob_match("*", ""));
    }

    #[test]
    fn star_prefix() {
        assert!(glob_match("*_tool", "read_tool"));
        assert!(glob_match("*_tool", "my_fancy_tool"));
        assert!(!glob_match("*_tool", "read_tools"));
    }

    #[test]
    fn star_suffix() {
        assert!(glob_match("read_*", "read_file"));
        assert!(glob_match("read_*", "read_"));
        assert!(!glob_match("read_*", "write_file"));
    }

    #[test]
    fn star_middle() {
        assert!(glob_match("mcp_*_call", "mcp_tool_call"));
        assert!(glob_match("mcp_*_call", "mcp__call"));
        assert!(!glob_match("mcp_*_call", "mcp_tool_exec"));
    }

    #[test]
    fn double_star() {
        assert!(glob_match("**", "anything"));
        assert!(glob_match("a**b", "ab"));
        assert!(glob_match("a**b", "aXYZb"));
    }

    // ── Glob matching: question mark ─────────────────────────────────

    #[test]
    fn question_mark_single_char() {
        assert!(glob_match("ba?h", "bash"));
        assert!(glob_match("ba?h", "bath"));
        assert!(!glob_match("ba?h", "baash"));
    }

    #[test]
    fn question_mark_at_end() {
        assert!(glob_match("rea?", "read"));
        assert!(!glob_match("rea?", "rea"));
    }

    // ── Glob matching: combined ──────────────────────────────────────

    #[test]
    fn star_and_question() {
        assert!(glob_match("*_?ool", "read_tool"));
        assert!(glob_match("*_?ool", "my_pool"));
        assert!(!glob_match("*_?ool", "my_ool"));
    }

    // ── Argument matching ────────────────────────────────────────────

    #[test]
    fn argument_string_match() {
        let args = json!({"command": "rm -rf /"});
        assert!(match_argument(&args, "/command", "rm *"));
        assert!(!match_argument(&args, "/command", "ls *"));
    }

    #[test]
    fn argument_nested_path() {
        let args = json!({"config": {"path": "/etc/passwd"}});
        assert!(match_argument(&args, "/config/path", "/etc/*"));
    }

    #[test]
    fn argument_number_match() {
        let args = json!({"timeout": 30});
        assert!(match_argument(&args, "/timeout", "30"));
        assert!(!match_argument(&args, "/timeout", "60"));
    }

    #[test]
    fn argument_bool_match() {
        let args = json!({"force": true});
        assert!(match_argument(&args, "/force", "true"));
    }

    #[test]
    fn argument_null_match() {
        let args = json!({"value": null});
        assert!(match_argument(&args, "/value", "null"));
    }

    #[test]
    fn argument_missing_path() {
        let args = json!({"foo": "bar"});
        assert!(!match_argument(&args, "/missing", "*"));
    }

    #[test]
    fn argument_array_no_match() {
        let args = json!({"items": [1, 2, 3]});
        assert!(!match_argument(&args, "/items", "*"));
    }

    #[test]
    fn argument_object_no_match() {
        let args = json!({"nested": {"a": 1}});
        assert!(!match_argument(&args, "/nested", "*"));
    }
}
