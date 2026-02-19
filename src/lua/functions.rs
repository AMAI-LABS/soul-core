//! Registered Lua functions for skill execution and RLM REPL.

use mlua::{Function, Lua, Value};

use super::{json_to_lua, lua_to_json};
use crate::error::{SoulError, SoulResult};

/// Register skill-specific functions on a Lua sandbox.
///
/// Functions: `json_decode`, `json_encode`, `url_encode`, `fetch`, `post`, `fetch_json`
pub fn register_skill_functions(lua: &Lua) -> SoulResult<()> {
    let globals = lua.globals();

    // json_decode(s) → Lua table
    globals
        .set("json_decode", create_json_decode(lua)?)
        .map_err(lua_err)?;

    // json_encode(t) → JSON string
    globals
        .set("json_encode", create_json_encode(lua)?)
        .map_err(lua_err)?;

    // url_encode(s) → percent-encoded string
    globals
        .set("url_encode", create_url_encode(lua)?)
        .map_err(lua_err)?;

    // fetch(url) → body string (sync HTTP GET)
    globals
        .set("fetch", create_fetch(lua)?)
        .map_err(lua_err)?;

    // fetch_json(url) → Lua table (sync HTTP GET + JSON parse)
    globals
        .set("fetch_json", create_fetch_json(lua)?)
        .map_err(lua_err)?;

    // post(url, body, content_type?) → body string (sync HTTP POST)
    globals
        .set("post", create_post(lua)?)
        .map_err(lua_err)?;

    // Crypto, encoding, time, and random functions
    super::crypto::register_crypto_functions(lua)?;

    Ok(())
}

/// Register RLM context functions on a Lua sandbox.
///
/// Functions: `chunk_by_lines`, `chunk_by_chars`, `chunk_by_regex`, `slice`, `search`
pub fn register_rlm_functions(lua: &Lua) -> SoulResult<()> {
    let globals = lua.globals();

    // chunk_by_lines(text, n) → table of chunks
    globals
        .set("chunk_by_lines", create_chunk_by_lines(lua)?)
        .map_err(lua_err)?;

    // chunk_by_chars(text, n) → table of chunks
    globals
        .set("chunk_by_chars", create_chunk_by_chars(lua)?)
        .map_err(lua_err)?;

    // chunk_by_regex(text, pattern) → table of chunks
    globals
        .set("chunk_by_regex", create_chunk_by_regex(lua)?)
        .map_err(lua_err)?;

    // slice(text, start, len) → substring
    globals
        .set("slice", create_slice(lua)?)
        .map_err(lua_err)?;

    // search(text, query, top_k?) → table of {text, score, line} results
    globals
        .set("search", create_search(lua)?)
        .map_err(lua_err)?;

    Ok(())
}

// ─── Function Constructors ──────────────────────────────────────────────────

fn create_json_decode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, s: String| {
        let value: serde_json::Value = serde_json::from_str(&s).map_err(|e| {
            mlua::Error::external(format!("json_decode error: {e}"))
        })?;
        json_to_lua(lua, &value)
    })
    .map_err(lua_err)
}

fn create_json_encode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, val: Value| {
        let json = lua_to_json(&val);
        serde_json::to_string(&json)
            .map_err(|e| mlua::Error::external(format!("json_encode error: {e}")))
    })
    .map_err(lua_err)
}

fn create_url_encode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, s: String| {
        Ok(percent_encode(&s))
    })
    .map_err(lua_err)
}

fn create_fetch(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, url: String| {
        // Use a blocking HTTP request — Lua execution is synchronous.
        // This is safe because the outer LuaSandbox wraps execution in a timeout.
        let body = blocking_get(&url)
            .map_err(|e| mlua::Error::external(format!("fetch error: {e}")))?;
        Ok(body)
    })
    .map_err(lua_err)
}

fn create_fetch_json(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, url: String| {
        let body = blocking_get(&url)
            .map_err(|e| mlua::Error::external(format!("fetch_json error: {e}")))?;
        let value: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| mlua::Error::external(format!("fetch_json parse error: {e}")))?;
        json_to_lua(lua, &value)
    })
    .map_err(lua_err)
}

fn create_post(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, (url, body, content_type): (String, String, Option<String>)| {
        let ct = content_type.unwrap_or_else(|| "application/json".to_string());
        let result = blocking_post(&url, &body, &ct)
            .map_err(|e| mlua::Error::external(format!("post error: {e}")))?;
        Ok(result)
    })
    .map_err(lua_err)
}

fn create_chunk_by_lines(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, (text, n): (String, usize)| {
        let n = n.max(1);
        let lines: Vec<&str> = text.lines().collect();
        let chunks: Vec<String> = lines.chunks(n).map(|c| c.join("\n")).collect();

        let table = lua.create_table()?;
        for (i, chunk) in chunks.iter().enumerate() {
            table.set(i + 1, chunk.as_str())?;
        }
        Ok(Value::Table(table))
    })
    .map_err(lua_err)
}

fn create_chunk_by_chars(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, (text, n): (String, usize)| {
        let n = n.max(1);
        let chars: Vec<char> = text.chars().collect();
        let chunks: Vec<String> = chars.chunks(n).map(|c| c.iter().collect()).collect();

        let table = lua.create_table()?;
        for (i, chunk) in chunks.iter().enumerate() {
            table.set(i + 1, chunk.as_str())?;
        }
        Ok(Value::Table(table))
    })
    .map_err(lua_err)
}

fn create_chunk_by_regex(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, (text, pattern): (String, String)| {
        let re = regex::Regex::new(&pattern)
            .map_err(|e| mlua::Error::external(format!("Invalid regex '{pattern}': {e}")))?;

        let chunks: Vec<&str> = re.split(&text).collect();

        let table = lua.create_table()?;
        for (i, chunk) in chunks.iter().enumerate() {
            table.set(i + 1, *chunk)?;
        }
        Ok(Value::Table(table))
    })
    .map_err(lua_err)
}

fn create_slice(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, (text, start, len): (String, usize, usize)| {
        let chars: Vec<char> = text.chars().collect();
        let start = start.min(chars.len());
        let end = (start + len).min(chars.len());
        let result: String = chars[start..end].iter().collect();
        Ok(result)
    })
    .map_err(lua_err)
}

fn create_search(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, (text, query, top_k): (String, String, Option<usize>)| {
        let top_k = top_k.unwrap_or(5);

        // Tokenize query into lowercase terms (skip short noise words)
        let query_terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| s.len() >= 2)
            .map(|s| s.to_lowercase())
            .collect();

        if query_terms.is_empty() {
            return Ok(Value::Table(lua.create_table()?));
        }

        // Split text into sections at blank lines or markdown headers
        let lines: Vec<&str> = text.lines().collect();
        let mut sections: Vec<(usize, String)> = Vec::new();
        let mut section_start = 0;

        for (i, line) in lines.iter().enumerate() {
            if line.trim().is_empty() || line.starts_with("## ") || line.starts_with("### ") {
                if i > section_start {
                    let section_text = lines[section_start..i].join("\n");
                    if !section_text.trim().is_empty() {
                        sections.push((section_start + 1, section_text));
                    }
                }
                if line.starts_with('#') {
                    section_start = i; // include header in next section
                } else {
                    section_start = i + 1;
                }
            }
        }
        // Last section
        if section_start < lines.len() {
            let section_text = lines[section_start..].join("\n");
            if !section_text.trim().is_empty() {
                sections.push((section_start + 1, section_text));
            }
        }

        if sections.is_empty() {
            return Ok(Value::Table(lua.create_table()?));
        }

        // IDF: log(N / df) for each query term
        let n = sections.len() as f64;
        let mut idf_map: Vec<f64> = Vec::with_capacity(query_terms.len());
        for term in &query_terms {
            let df = sections
                .iter()
                .filter(|(_, s)| s.to_lowercase().contains(term.as_str()))
                .count();
            let idf = if df > 0 { (n / df as f64).ln() + 1.0 } else { 0.0 };
            idf_map.push(idf);
        }

        // Score each section: BM25-lite (tf * idf, length-normalized)
        let mut scored: Vec<(f64, usize, usize)> = sections
            .iter()
            .enumerate()
            .map(|(idx, (line, section))| {
                let lower = section.to_lowercase();
                let word_count = lower.split_whitespace().count().max(1) as f64;
                let mut score = 0.0f64;

                for (ti, term) in query_terms.iter().enumerate() {
                    let tf = lower.matches(term.as_str()).count() as f64 / word_count;
                    score += tf * idf_map[ti];
                }

                (score, *line, idx)
            })
            .filter(|(score, _, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // Build result table: [{text=..., score=..., line=...}, ...]
        let result_table = lua.create_table()?;
        for (i, (score, line, idx)) in scored.iter().enumerate() {
            let entry = lua.create_table()?;
            entry.set("text", sections[*idx].1.as_str())?;
            entry.set("score", *score)?;
            entry.set("line", *line)?;
            result_table.set(i + 1, entry)?;
        }
        Ok(Value::Table(result_table))
    })
    .map_err(lua_err)
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Simple percent-encoding (RFC 3986 unreserved characters).
fn percent_encode(s: &str) -> String {
    let mut encoded = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push_str(&format!("%{byte:02X}"));
            }
        }
    }
    encoded
}

/// Blocking HTTP GET. Uses reqwest blocking client.
fn blocking_get(url: &str) -> Result<String, String> {
    // Build a new tokio runtime is not feasible here since we're already in one.
    // Use reqwest's blocking client which uses its own thread pool.
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let response = client
        .get(url)
        .header("User-Agent", "amai-agent/lua-sandbox")
        .send()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("HTTP {status} from {url}"));
    }

    response
        .text()
        .map_err(|e| format!("Failed to read response: {e}"))
}

/// Blocking HTTP POST.
fn blocking_post(url: &str, body: &str, content_type: &str) -> Result<String, String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let response = client
        .post(url)
        .header("User-Agent", "amai-agent/lua-sandbox")
        .header("Content-Type", content_type)
        .body(body.to_string())
        .send()
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        return Err(format!("HTTP {status} from {url}"));
    }

    response
        .text()
        .map_err(|e| format!("Failed to read response: {e}"))
}

/// Convert mlua::Error to SoulError.
fn lua_err(e: mlua::Error) -> SoulError {
    SoulError::ToolExecution {
        tool_name: "lua_sandbox".into(),
        message: format!("Lua function registration error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lua::LuaSandbox;

    fn sandbox_with_skill_fns() -> LuaSandbox {
        let sb = LuaSandbox::new().unwrap();
        register_skill_functions(sb.lua()).unwrap();
        sb
    }

    fn sandbox_with_rlm_fns() -> LuaSandbox {
        let sb = LuaSandbox::new().unwrap();
        register_rlm_functions(sb.lua()).unwrap();
        sb
    }

    // ─── JSON ─────────────────────────────────────────────────────────

    #[test]
    fn json_decode_object() {
        let sb = sandbox_with_skill_fns();
        let result = sb
            .exec(r#"local t = json_decode('{"name":"test","count":3}'); return t.name"#)
            .unwrap();
        assert_eq!(result, "test");
    }

    #[test]
    fn json_decode_array() {
        let sb = sandbox_with_skill_fns();
        let result = sb
            .exec(r#"local t = json_decode('[1,2,3]'); return #t"#)
            .unwrap();
        assert_eq!(result, "3");
    }

    #[test]
    fn json_decode_invalid() {
        let sb = sandbox_with_skill_fns();
        let result = sb.exec(r#"json_decode('not json')"#);
        assert!(result.is_err());
    }

    #[test]
    fn json_encode_table() {
        let sb = sandbox_with_skill_fns();
        let result = sb
            .exec(r#"return json_encode({name = "test"})"#)
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["name"], "test");
    }

    #[test]
    fn json_encode_decode_roundtrip() {
        let sb = sandbox_with_skill_fns();
        let result = sb
            .exec(
                r#"
            local obj = {name = "test", count = 42}
            local encoded = json_encode(obj)
            local decoded = json_decode(encoded)
            return decoded.name .. ":" .. decoded.count
        "#,
            )
            .unwrap();
        assert_eq!(result, "test:42");
    }

    // ─── URL Encode ──────────────────────────────────────────────────

    #[test]
    fn url_encode_basic() {
        let sb = sandbox_with_skill_fns();
        let result = sb
            .exec(r#"return url_encode("hello world")"#)
            .unwrap();
        assert_eq!(result, "hello%20world");
    }

    #[test]
    fn url_encode_special_chars() {
        let sb = sandbox_with_skill_fns();
        let result = sb
            .exec(r#"return url_encode("a=b&c=d")"#)
            .unwrap();
        assert_eq!(result, "a%3Db%26c%3Dd");
    }

    // ─── Chunk Functions ─────────────────────────────────────────────

    #[test]
    fn chunk_by_lines_basic() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "line1\nline2\nline3\nline4\nline5")
            .unwrap();
        let result = sb.exec("local chunks = chunk_by_lines(text, 2); return #chunks").unwrap();
        assert_eq!(result, "3"); // 2+2+1
    }

    #[test]
    fn chunk_by_lines_content() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "a\nb\nc\nd").unwrap();
        let result = sb
            .exec("local chunks = chunk_by_lines(text, 2); return chunks[1]")
            .unwrap();
        assert_eq!(result, "a\nb");
    }

    #[test]
    fn chunk_by_chars_basic() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "abcdefghij").unwrap();
        let result = sb
            .exec("local chunks = chunk_by_chars(text, 3); return #chunks")
            .unwrap();
        assert_eq!(result, "4"); // 3+3+3+1
    }

    #[test]
    fn chunk_by_regex_basic() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "part1---part2---part3").unwrap();
        let result = sb
            .exec("local chunks = chunk_by_regex(text, '---'); return table.concat(chunks, ',')")
            .unwrap();
        assert_eq!(result, "part1,part2,part3");
    }

    #[test]
    fn slice_basic() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "hello world").unwrap();
        let result = sb.exec("return slice(text, 0, 5)").unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn slice_out_of_bounds() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "hello").unwrap();
        let result = sb.exec("return slice(text, 3, 100)").unwrap();
        assert_eq!(result, "lo");
    }

    // ─── Search (BM25) ────────────────────────────────────────────

    #[test]
    fn search_basic_keyword() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string(
            "text",
            "## TURN 1\n### USER\nHow do I use async rust?\n### ASSISTANT\nUse tokio and async/await\n\n## TURN 2\n### USER\nWrite a test for the server\n### ASSISTANT\nHere is the test code\n\n## TURN 3\n### USER\nFix the database error\n### ASSISTANT\nThe error was in the connection pool",
        ).unwrap();
        let result = sb
            .exec(r#"local r = search(text, "error database", 2); return #r"#)
            .unwrap();
        let count: usize = result.parse().unwrap();
        assert!(count >= 1); // Should find the database error section
    }

    #[test]
    fn search_returns_scored_results() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string(
            "text",
            "## Section A\nThe cat sat on the mat\n\n## Section B\nThe dog ran in the park\n\n## Section C\nThe cat chased the dog across the mat",
        ).unwrap();
        let result = sb
            .exec(r#"local r = search(text, "cat mat"); return r[1].score > 0"#)
            .unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn search_respects_top_k() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string(
            "text",
            "## A\nalpha bravo\n\n## B\ncharlie bravo\n\n## C\ndelta bravo\n\n## D\necho bravo",
        ).unwrap();
        let result = sb
            .exec(r#"local r = search(text, "bravo", 2); return #r"#)
            .unwrap();
        assert_eq!(result, "2");
    }

    #[test]
    fn search_empty_query() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "some text here").unwrap();
        // Single char query terms are filtered out
        let result = sb
            .exec(r#"local r = search(text, "a b"); return #r"#)
            .unwrap();
        assert_eq!(result, "0");
    }

    #[test]
    fn search_no_matches() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "## Section\nThe quick brown fox").unwrap();
        let result = sb
            .exec(r#"local r = search(text, "zebra elephant"); return #r"#)
            .unwrap();
        assert_eq!(result, "0");
    }

    #[test]
    fn search_result_has_line_numbers() {
        let sb = sandbox_with_rlm_fns();
        sb.set_string("text", "## First\nHello world\n\n## Second\nGoodbye world").unwrap();
        let result = sb
            .exec(r#"local r = search(text, "goodbye"); return r[1].line"#)
            .unwrap();
        let line: usize = result.parse().unwrap();
        assert!(line >= 4); // "## Second" starts at line 4
    }

    // ─── percent_encode ─────────────────────────────────────────────

    #[test]
    fn percent_encode_passthrough() {
        assert_eq!(percent_encode("hello"), "hello");
        assert_eq!(percent_encode("a-b_c.d~e"), "a-b_c.d~e");
    }

    #[test]
    fn percent_encode_spaces_and_specials() {
        assert_eq!(percent_encode("hello world"), "hello%20world");
        assert_eq!(percent_encode("/path?q=1"), "%2Fpath%3Fq%3D1");
    }
}
