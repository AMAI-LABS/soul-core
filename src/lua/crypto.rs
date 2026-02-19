//! Crypto, encoding, and utility functions for the Lua sandbox.
//!
//! Exposes Ed25519 keygen/sign/verify, SHA-256 hashing, base64/hex encoding,
//! timestamps, and random byte generation to Lua skills.

use base64::Engine;
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use mlua::{Function, Lua, Value};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};

use crate::error::{SoulError, SoulResult};

/// Register all crypto and utility functions on a Lua VM.
pub fn register_crypto_functions(lua: &Lua) -> SoulResult<()> {
    let globals = lua.globals();

    globals
        .set("ed25519_keygen", create_ed25519_keygen(lua)?)
        .map_err(lua_err)?;
    globals
        .set("ed25519_sign", create_ed25519_sign(lua)?)
        .map_err(lua_err)?;
    globals
        .set("ed25519_verify", create_ed25519_verify(lua)?)
        .map_err(lua_err)?;
    globals
        .set("sha256", create_sha256(lua)?)
        .map_err(lua_err)?;
    globals
        .set("sha256_bytes", create_sha256_bytes(lua)?)
        .map_err(lua_err)?;
    globals
        .set("base64_encode", create_base64_encode(lua)?)
        .map_err(lua_err)?;
    globals
        .set("base64_decode", create_base64_decode(lua)?)
        .map_err(lua_err)?;
    globals
        .set("hex_encode", create_hex_encode(lua)?)
        .map_err(lua_err)?;
    globals
        .set("hex_decode", create_hex_decode(lua)?)
        .map_err(lua_err)?;
    globals
        .set("timestamp", create_timestamp(lua)?)
        .map_err(lua_err)?;
    globals
        .set("timestamp_unix", create_timestamp_unix(lua)?)
        .map_err(lua_err)?;
    globals
        .set("random_bytes", create_random_bytes(lua)?)
        .map_err(lua_err)?;
    globals
        .set("random_hex", create_random_hex(lua)?)
        .map_err(lua_err)?;

    Ok(())
}

// ─── Ed25519 ─────────────────────────────────────────────────────────────────

fn create_ed25519_keygen(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, ()| {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        let secret_hex = hex::encode(signing_key.to_bytes());
        let public_hex = hex::encode(verifying_key.to_bytes());
        let public_b64 =
            base64::engine::general_purpose::STANDARD.encode(verifying_key.to_bytes());

        // PEM-like format for the public key (not full X.509, just base64-wrapped)
        let pem = format!(
            "-----BEGIN ED25519 PUBLIC KEY-----\n{}\n-----END ED25519 PUBLIC KEY-----",
            public_b64
        );

        let table = lua.create_table()?;
        table.set("public_key", public_hex)?;
        table.set("secret_key", secret_hex)?;
        table.set("public_key_b64", public_b64)?;
        table.set("public_key_pem", pem)?;

        Ok(Value::Table(table))
    })
    .map_err(lua_err)
}

fn create_ed25519_sign(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, (secret_key_hex, message): (String, String)| {
        let key_bytes: [u8; 32] = hex::decode(&secret_key_hex)
            .map_err(|e| mlua::Error::external(format!("ed25519_sign: invalid hex key: {e}")))?
            .try_into()
            .map_err(|_| {
                mlua::Error::external("ed25519_sign: secret key must be 32 bytes (64 hex chars)")
            })?;

        let signing_key = SigningKey::from_bytes(&key_bytes);
        let signature = signing_key.sign(message.as_bytes());
        let sig_b64 = base64::engine::general_purpose::STANDARD.encode(signature.to_bytes());

        Ok(sig_b64)
    })
    .map_err(lua_err)
}

fn create_ed25519_verify(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(
        |_, (public_key_hex, message, signature_b64): (String, String, String)| {
            let key_bytes: [u8; 32] = hex::decode(&public_key_hex)
                .map_err(|e| {
                    mlua::Error::external(format!("ed25519_verify: invalid hex key: {e}"))
                })?
                .try_into()
                .map_err(|_| {
                    mlua::Error::external(
                        "ed25519_verify: public key must be 32 bytes (64 hex chars)",
                    )
                })?;

            let sig_bytes = base64::engine::general_purpose::STANDARD
                .decode(&signature_b64)
                .map_err(|e| {
                    mlua::Error::external(format!("ed25519_verify: invalid base64 signature: {e}"))
                })?;

            let sig_array: [u8; 64] = sig_bytes.try_into().map_err(|_| {
                mlua::Error::external("ed25519_verify: signature must be 64 bytes")
            })?;

            let verifying_key = VerifyingKey::from_bytes(&key_bytes).map_err(|e| {
                mlua::Error::external(format!("ed25519_verify: invalid public key: {e}"))
            })?;

            let signature = ed25519_dalek::Signature::from_bytes(&sig_array);
            Ok(verifying_key.verify(message.as_bytes(), &signature).is_ok())
        },
    )
    .map_err(lua_err)
}

// ─── Hashing ─────────────────────────────────────────────────────────────────

fn create_sha256(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, data: String| {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        Ok(hex::encode(hasher.finalize()))
    })
    .map_err(lua_err)
}

fn create_sha256_bytes(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|lua, data: String| {
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        let hash = hasher.finalize();
        lua.create_string(hash.as_slice())
    })
    .map_err(lua_err)
}

// ─── Encoding ────────────────────────────────────────────────────────────────

fn create_base64_encode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, data: String| {
        Ok(base64::engine::general_purpose::STANDARD.encode(data.as_bytes()))
    })
    .map_err(lua_err)
}

fn create_base64_decode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, b64: String| {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .map_err(|e| mlua::Error::external(format!("base64_decode: invalid base64: {e}")))?;
        String::from_utf8(bytes)
            .map_err(|e| mlua::Error::external(format!("base64_decode: invalid UTF-8: {e}")))
    })
    .map_err(lua_err)
}

fn create_hex_encode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, data: String| Ok(hex::encode(data.as_bytes())))
        .map_err(lua_err)
}

fn create_hex_decode(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, hex_str: String| {
        let bytes = hex::decode(&hex_str)
            .map_err(|e| mlua::Error::external(format!("hex_decode: invalid hex: {e}")))?;
        String::from_utf8(bytes)
            .map_err(|e| mlua::Error::external(format!("hex_decode: invalid UTF-8: {e}")))
    })
    .map_err(lua_err)
}

// ─── Time ────────────────────────────────────────────────────────────────────

fn create_timestamp(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, ()| {
        Ok(chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string())
    })
    .map_err(lua_err)
}

fn create_timestamp_unix(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, ()| Ok(chrono::Utc::now().timestamp()))
        .map_err(lua_err)
}

// ─── Random ──────────────────────────────────────────────────────────────────

fn create_random_bytes(lua: &Lua) -> SoulResult<Function> {
    lua.create_function(|_, n: usize| {
        if n > 1024 {
            return Err(mlua::Error::external(
                "random_bytes: max 1024 bytes per call",
            ));
        }
        let mut buf = vec![0u8; n];
        rand::RngCore::fill_bytes(&mut OsRng, &mut buf);
        Ok(hex::encode(buf))
    })
    .map_err(lua_err)
}

fn create_random_hex(lua: &Lua) -> SoulResult<Function> {
    // Alias for random_bytes — clearer name for nonce generation
    lua.create_function(|_, n: usize| {
        if n > 1024 {
            return Err(mlua::Error::external(
                "random_hex: max 1024 bytes per call",
            ));
        }
        let mut buf = vec![0u8; n];
        rand::RngCore::fill_bytes(&mut OsRng, &mut buf);
        Ok(hex::encode(buf))
    })
    .map_err(lua_err)
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn lua_err(e: mlua::Error) -> SoulError {
    SoulError::ToolExecution {
        tool_name: "lua_sandbox".into(),
        message: format!("Lua crypto function registration error: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lua::LuaSandbox;

    fn sandbox_with_crypto() -> LuaSandbox {
        let sb = LuaSandbox::new().unwrap();
        register_crypto_functions(sb.lua()).unwrap();
        sb
    }

    #[test]
    fn ed25519_keygen() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(
                r#"
            local keys = ed25519_keygen()
            -- public_key is 32 bytes = 64 hex chars
            -- secret_key is 32 bytes = 64 hex chars
            return string.len(keys.public_key) .. ":" .. string.len(keys.secret_key)
        "#,
            )
            .unwrap();
        assert_eq!(result, "64:64");
    }

    #[test]
    fn ed25519_sign_verify_roundtrip() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(
                r#"
            local keys = ed25519_keygen()
            local sig = ed25519_sign(keys.secret_key, "hello world")
            local valid = ed25519_verify(keys.public_key, "hello world", sig)
            return tostring(valid)
        "#,
            )
            .unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn ed25519_verify_bad_signature() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(
                r#"
            local keys = ed25519_keygen()
            local sig = ed25519_sign(keys.secret_key, "hello world")
            local valid = ed25519_verify(keys.public_key, "wrong message", sig)
            return tostring(valid)
        "#,
            )
            .unwrap();
        assert_eq!(result, "false");
    }

    #[test]
    fn sha256_known_vector() {
        let sb = sandbox_with_crypto();
        let result = sb.exec(r#"return sha256("hello")"#).unwrap();
        // Known SHA-256 of "hello"
        assert_eq!(
            result,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn base64_roundtrip() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(
                r#"
            local encoded = base64_encode("hello world")
            local decoded = base64_decode(encoded)
            return decoded
        "#,
            )
            .unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn hex_roundtrip() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(
                r#"
            local encoded = hex_encode("hello world")
            local decoded = hex_decode(encoded)
            return decoded
        "#,
            )
            .unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn timestamp_format() {
        let sb = sandbox_with_crypto();
        let result = sb.exec(r#"return timestamp()"#).unwrap();
        // ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        assert!(
            result.len() == 20,
            "Expected 20 chars (ISO 8601), got {}: {}",
            result.len(),
            result
        );
        assert!(result.ends_with('Z'));
        assert!(result.contains('T'));
    }

    #[test]
    fn timestamp_unix_reasonable() {
        let sb = sandbox_with_crypto();
        let result = sb.exec(r#"return timestamp_unix()"#).unwrap();
        let ts: i64 = result.parse().expect("should be a number");
        assert!(ts > 1_700_000_000, "Timestamp too low: {ts}");
    }

    #[test]
    fn random_bytes_length() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(r#"return string.len(random_bytes(16))"#)
            .unwrap();
        // 16 bytes = 32 hex chars
        assert_eq!(result, "32");
    }

    #[test]
    fn ed25519_pem_format() {
        let sb = sandbox_with_crypto();
        let result = sb
            .exec(r#"return ed25519_keygen().public_key_pem"#)
            .unwrap();
        assert!(result.starts_with("-----BEGIN ED25519 PUBLIC KEY-----"));
        assert!(result.ends_with("-----END ED25519 PUBLIC KEY-----"));
    }
}
