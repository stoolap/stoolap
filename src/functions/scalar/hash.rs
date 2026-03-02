// Copyright 2025 Stoolap Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Hash Functions
//!
//! Cryptographic and checksum hash functions for SQL queries:
//!
//! - [`Md5Function`] - MD5(value) → hex string
//! - [`Sha1Function`] - SHA1(value) → hex string
//! - [`Sha256Function`] - SHA256(value) → hex string
//! - [`Sha384Function`] - SHA384(value) → hex string
//! - [`Sha512Function`] - SHA512(value) → hex string
//! - [`Crc32Function`] - CRC32(value) → integer

use md5::Md5;
use sha1::Sha1;
use sha2::{Digest, Sha256, Sha384, Sha512};

use crate::core::Value;
use crate::functions::{
    FunctionDataType, FunctionInfo, FunctionSignature, FunctionType, NativeFn1, ScalarFunction,
};
use crate::validate_arg_count;

use super::value_to_string;

/// Format a digest as a lowercase hex string without allocation overhead.
/// Pre-sizes the string to exact length (2 hex chars per byte).
#[inline]
fn hex_encode(bytes: &[u8]) -> String {
    const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";
    let mut hex = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        hex.push(HEX_CHARS[(b >> 4) as usize] as char);
        hex.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
    hex
}

/// Compute a digest over a Value without intermediate allocation for Text.
/// Text values pass their UTF-8 bytes directly to the hasher (zero-copy).
/// Other types are converted to their string representation first.
#[inline]
fn digest_value<D: Digest>(value: &Value) -> sha2::digest::Output<D> {
    match value {
        Value::Text(s) => D::digest(s.as_bytes()),
        _ => D::digest(value_to_string(value).as_bytes()),
    }
}

/// Compute CRC32 over a Value without intermediate allocation for Text.
#[inline]
fn crc32_value(value: &Value) -> u32 {
    match value {
        Value::Text(s) => crc32fast::hash(s.as_bytes()),
        _ => crc32fast::hash(value_to_string(value).as_bytes()),
    }
}

// ---------------------------------------------------------------------------
// MD5
// ---------------------------------------------------------------------------

/// MD5 hash function - returns lowercase hex string
#[derive(Default)]
pub struct Md5Function;

impl Md5Function {
    #[inline]
    fn native(v: &mut Value) {
        if v.is_null() {
            return;
        }
        let hash = digest_value::<Md5>(v);
        *v = Value::text(hex_encode(hash.as_slice()));
    }
}

impl ScalarFunction for Md5Function {
    fn name(&self) -> &str {
        "MD5"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "MD5",
            FunctionType::Scalar,
            "Returns MD5 hash as a hex string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> crate::core::Result<Value> {
        validate_arg_count!(args, "MD5", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let hash = digest_value::<Md5>(&args[0]);
        Ok(Value::text(hex_encode(hash.as_slice())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Md5Function)
    }

    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(Self::native)
    }
}

// ---------------------------------------------------------------------------
// SHA1
// ---------------------------------------------------------------------------

/// SHA1 hash function - returns lowercase hex string
#[derive(Default)]
pub struct Sha1Function;

impl Sha1Function {
    #[inline]
    fn native(v: &mut Value) {
        if v.is_null() {
            return;
        }
        let hash = digest_value::<Sha1>(v);
        *v = Value::text(hex_encode(hash.as_slice()));
    }
}

impl ScalarFunction for Sha1Function {
    fn name(&self) -> &str {
        "SHA1"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SHA1",
            FunctionType::Scalar,
            "Returns SHA-1 hash as a hex string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> crate::core::Result<Value> {
        validate_arg_count!(args, "SHA1", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let hash = digest_value::<Sha1>(&args[0]);
        Ok(Value::text(hex_encode(hash.as_slice())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Sha1Function)
    }

    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(Self::native)
    }
}

// ---------------------------------------------------------------------------
// SHA256
// ---------------------------------------------------------------------------

/// SHA256 hash function - returns lowercase hex string
#[derive(Default)]
pub struct Sha256Function;

impl Sha256Function {
    #[inline]
    fn native(v: &mut Value) {
        if v.is_null() {
            return;
        }
        let hash = digest_value::<Sha256>(v);
        *v = Value::text(hex_encode(hash.as_slice()));
    }
}

impl ScalarFunction for Sha256Function {
    fn name(&self) -> &str {
        "SHA256"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SHA256",
            FunctionType::Scalar,
            "Returns SHA-256 hash as a hex string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> crate::core::Result<Value> {
        validate_arg_count!(args, "SHA256", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let hash = digest_value::<Sha256>(&args[0]);
        Ok(Value::text(hex_encode(hash.as_slice())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Sha256Function)
    }

    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(Self::native)
    }
}

// ---------------------------------------------------------------------------
// SHA384
// ---------------------------------------------------------------------------

/// SHA384 hash function - returns lowercase hex string
#[derive(Default)]
pub struct Sha384Function;

impl Sha384Function {
    #[inline]
    fn native(v: &mut Value) {
        if v.is_null() {
            return;
        }
        let hash = digest_value::<Sha384>(v);
        *v = Value::text(hex_encode(hash.as_slice()));
    }
}

impl ScalarFunction for Sha384Function {
    fn name(&self) -> &str {
        "SHA384"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SHA384",
            FunctionType::Scalar,
            "Returns SHA-384 hash as a hex string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> crate::core::Result<Value> {
        validate_arg_count!(args, "SHA384", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let hash = digest_value::<Sha384>(&args[0]);
        Ok(Value::text(hex_encode(hash.as_slice())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Sha384Function)
    }

    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(Self::native)
    }
}

// ---------------------------------------------------------------------------
// SHA512
// ---------------------------------------------------------------------------

/// SHA512 hash function - returns lowercase hex string
#[derive(Default)]
pub struct Sha512Function;

impl Sha512Function {
    #[inline]
    fn native(v: &mut Value) {
        if v.is_null() {
            return;
        }
        let hash = digest_value::<Sha512>(v);
        *v = Value::text(hex_encode(hash.as_slice()));
    }
}

impl ScalarFunction for Sha512Function {
    fn name(&self) -> &str {
        "SHA512"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "SHA512",
            FunctionType::Scalar,
            "Returns SHA-512 hash as a hex string",
            FunctionSignature::new(FunctionDataType::String, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> crate::core::Result<Value> {
        validate_arg_count!(args, "SHA512", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let hash = digest_value::<Sha512>(&args[0]);
        Ok(Value::text(hex_encode(hash.as_slice())))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Sha512Function)
    }

    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(Self::native)
    }
}

// ---------------------------------------------------------------------------
// CRC32
// ---------------------------------------------------------------------------

/// CRC32 checksum function - returns integer
#[derive(Default)]
pub struct Crc32Function;

impl Crc32Function {
    #[inline]
    fn native(v: &mut Value) {
        if v.is_null() {
            return;
        }
        let checksum = crc32_value(v);
        *v = Value::Integer(checksum as i64);
    }
}

impl ScalarFunction for Crc32Function {
    fn name(&self) -> &str {
        "CRC32"
    }

    fn info(&self) -> FunctionInfo {
        FunctionInfo::new(
            "CRC32",
            FunctionType::Scalar,
            "Returns CRC32 checksum as an integer",
            FunctionSignature::new(FunctionDataType::Integer, vec![FunctionDataType::Any], 1, 1),
        )
    }

    fn evaluate(&self, args: &[Value]) -> crate::core::Result<Value> {
        validate_arg_count!(args, "CRC32", 1);
        if args[0].is_null() {
            return Ok(Value::null_unknown());
        }
        let checksum = crc32_value(&args[0]);
        Ok(Value::Integer(checksum as i64))
    }

    fn clone_box(&self) -> Box<dyn ScalarFunction> {
        Box::new(Crc32Function)
    }

    fn native_fn1(&self) -> Option<NativeFn1> {
        Some(Self::native)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5() {
        let f = Md5Function;
        // Known MD5 of empty string
        assert_eq!(
            f.evaluate(&[Value::text("")]).unwrap(),
            Value::text("d41d8cd98f00b204e9800998ecf8427e")
        );
        // Known MD5 of "hello"
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("5d41402abc4b2a76b9719d911017c592")
        );
        // NULL returns NULL
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
        // Integer input
        assert_eq!(
            f.evaluate(&[Value::Integer(42)]).unwrap(),
            Value::text("a1d0c6e83f027327d8461063f4ac58a6")
        );
    }

    #[test]
    fn test_sha1() {
        let f = Sha1Function;
        // Known SHA1 of "hello"
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d")
        );
        // NULL returns NULL
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_sha256() {
        let f = Sha256Function;
        // Known SHA256 of "hello"
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
        );
        // NULL returns NULL
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_sha384() {
        let f = Sha384Function;
        // Known SHA384 of "hello"
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("59e1748777448c69de6b800d7a33bbfb9ff1b463e44354c3553bcdb9c666fa90125a3c79f90397bdf5f6a13de828684f")
        );
        // NULL returns NULL
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_sha512() {
        let f = Sha512Function;
        // Known SHA512 of "hello"
        assert_eq!(
            f.evaluate(&[Value::text("hello")]).unwrap(),
            Value::text("9b71d224bd62f3785d96d46ad3ea3d73319bfbc2890caadae2dff72519673ca72323c3d99ba5c11d7c7acc6e14b8c5da0c4663475c2e5c3adef46f73bcdec043")
        );
        // NULL returns NULL
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
    }

    #[test]
    fn test_crc32() {
        let f = Crc32Function;
        // CRC32 of "hello"
        let result = f.evaluate(&[Value::text("hello")]).unwrap();
        assert_eq!(result, Value::Integer(907060870));
        // NULL returns NULL
        assert!(f.evaluate(&[Value::null_unknown()]).unwrap().is_null());
        // Empty string
        let result = f.evaluate(&[Value::text("")]).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn test_hash_integer_input() {
        let md5 = Md5Function;
        let sha256 = Sha256Function;
        let crc32 = Crc32Function;
        // All hash functions should handle integer input
        assert!(matches!(
            md5.evaluate(&[Value::Integer(123)]).unwrap(),
            Value::Text(_)
        ));
        assert!(matches!(
            sha256.evaluate(&[Value::Integer(123)]).unwrap(),
            Value::Text(_)
        ));
        assert!(matches!(
            crc32.evaluate(&[Value::Integer(123)]).unwrap(),
            Value::Integer(_)
        ));
    }

    #[test]
    fn test_hash_float_input() {
        let md5 = Md5Function;
        // Float input should be converted to string then hashed
        assert!(matches!(
            md5.evaluate(&[Value::Float(1.5)]).unwrap(),
            Value::Text(_)
        ));
    }

    #[test]
    fn test_hash_boolean_input() {
        let md5 = Md5Function;
        assert!(matches!(
            md5.evaluate(&[Value::Boolean(true)]).unwrap(),
            Value::Text(_)
        ));
    }

    #[test]
    fn test_wrong_arg_count() {
        let md5 = Md5Function;
        assert!(md5.evaluate(&[]).is_err());
        assert!(md5.evaluate(&[Value::text("a"), Value::text("b")]).is_err());
    }

    #[test]
    fn test_native_fn1() {
        // Verify native function pointers work correctly
        let md5 = Md5Function;
        assert!(md5.native_fn1().is_some());
        let sha1 = Sha1Function;
        assert!(sha1.native_fn1().is_some());
        let sha256 = Sha256Function;
        assert!(sha256.native_fn1().is_some());
        let crc32 = Crc32Function;
        assert!(crc32.native_fn1().is_some());

        // Test native mutation
        let mut v = Value::text("hello");
        Md5Function::native(&mut v);
        assert_eq!(v, Value::text("5d41402abc4b2a76b9719d911017c592"));
    }

    #[test]
    fn test_hex_encode() {
        assert_eq!(hex_encode(&[]), "");
        assert_eq!(hex_encode(&[0x00]), "00");
        assert_eq!(hex_encode(&[0xff]), "ff");
        assert_eq!(hex_encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }
}
