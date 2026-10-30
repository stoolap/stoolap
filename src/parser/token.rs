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

//! Token types for SQL lexer
//!
//! This module defines the token types used by the SQL lexer and parser.

use crate::common::smart_string::MAX_INLINE_LEN;
use crate::common::SmartString;
use std::fmt;

/// Position represents a position in the input source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    /// Byte offset, starting at 0
    pub offset: u32,
    /// Line number, starting at 1
    pub line: u32,
    /// Column number, starting at 1
    pub column: u32,
}

impl Position {
    /// Create a new position
    pub fn new(offset: u32, line: u32, column: u32) -> Self {
        Self {
            offset,
            line,
            column,
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}, column {}", self.line, self.column)
    }
}

/// TokenType represents the type of a token
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    /// Error token
    Error,
    /// End of file
    Eof,
    /// Identifier (table name, column name, etc.)
    Identifier,
    /// SQL keyword (SELECT, FROM, WHERE, etc.)
    Keyword,
    /// String literal ('hello')
    String,
    /// Integer number (123)
    Integer,
    /// Floating point number (123.45)
    Float,
    /// Operator (=, <, >, +, -, etc.)
    Operator,
    /// Punctuator (comma, semicolon, parentheses, etc.)
    Punctuator,
    /// Comment (-- or /* */)
    Comment,
    /// Date literal
    Date,
    /// Time literal
    Time,
    /// Timestamp literal
    Timestamp,
    /// Parameter ($1, ?)
    Parameter,
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenType::Error => write!(f, "ERROR"),
            TokenType::Eof => write!(f, "EOF"),
            TokenType::Identifier => write!(f, "IDENTIFIER"),
            TokenType::Keyword => write!(f, "KEYWORD"),
            TokenType::String => write!(f, "STRING"),
            TokenType::Integer => write!(f, "INTEGER"),
            TokenType::Float => write!(f, "FLOAT"),
            TokenType::Operator => write!(f, "OPERATOR"),
            TokenType::Punctuator => write!(f, "PUNCTUATOR"),
            TokenType::Comment => write!(f, "COMMENT"),
            TokenType::Date => write!(f, "DATE"),
            TokenType::Time => write!(f, "TIME"),
            TokenType::Timestamp => write!(f, "TIMESTAMP"),
            TokenType::Parameter => write!(f, "PARAMETER"),
        }
    }
}

/// Token represents a lexical token
///
/// Uses SmartString for literal to avoid heap allocation for tokens up to 15 bytes
/// (covers most SQL tokens: keywords, operators, short identifiers, numbers).
/// For error tokens, the literal field contains the error message.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The type of the token
    pub token_type: TokenType,
    /// The literal string value (SmartString inlines strings up to 15 bytes)
    /// For error tokens, this contains the error message instead.
    pub literal: SmartString,
    /// The position in the source
    pub position: Position,
    /// Whether this token was double-quoted (identifier with string fallback)
    pub quoted: bool,
    /// The keyword a Keyword token spells
    pub keyword: Option<Keyword>,
}

impl Token {
    /// Create a new token.
    ///
    /// A `SmartString` the lexer has already built moves in as is; taking
    /// `AsRef<str>` here used to copy it into a second one, which for a long
    /// comment or identifier meant a second heap string and Arc.
    #[inline]
    pub fn new(token_type: TokenType, literal: impl Into<SmartString>, position: Position) -> Self {
        let literal = literal.into();
        let keyword = if token_type == TokenType::Keyword {
            lookup_keyword(&literal)
        } else {
            None
        };
        Self {
            token_type,
            literal,
            position,
            quoted: false,
            keyword,
        }
    }

    /// Create a keyword token, spelled in upper case
    #[inline]
    pub fn from_keyword(keyword: Keyword, position: Position) -> Self {
        let text = keyword.as_str();
        let literal = match text.len() {
            len if len <= MAX_INLINE_LEN => {
                let mut data = [0; MAX_INLINE_LEN];
                data.copy_from_slice(&SPELLINGS[keyword as usize][..MAX_INLINE_LEN]);
                SmartString::inline_bytes(data, len)
            }
            _ => SmartString::new(text),
        };
        Self {
            token_type: TokenType::Keyword,
            literal,
            position,
            quoted: false,
            keyword: Some(keyword),
        }
    }

    /// Create a new double-quoted token (identifier with string fallback)
    #[inline]
    pub fn new_quoted(
        token_type: TokenType,
        literal: impl Into<SmartString>,
        position: Position,
    ) -> Self {
        Self {
            token_type,
            literal: literal.into(),
            position,
            quoted: true,
            keyword: None,
        }
    }

    /// Create an error token (error message is stored in literal field)
    pub fn error(message: impl AsRef<str>, _literal: impl AsRef<str>, position: Position) -> Self {
        Self {
            token_type: TokenType::Error,
            literal: SmartString::from(message.as_ref()),
            position,
            quoted: false,
            keyword: None,
        }
    }

    /// Create an EOF token
    #[inline]
    pub fn eof(position: Position) -> Self {
        Self {
            token_type: TokenType::Eof,
            literal: SmartString::const_new(""),
            position,
            quoted: false,
            keyword: None,
        }
    }

    /// Check if this is an EOF token
    pub fn is_eof(&self) -> bool {
        self.token_type == TokenType::Eof
    }

    /// Check if this is an error token
    pub fn is_error(&self) -> bool {
        self.token_type == TokenType::Error
    }

    /// Check if this is the given keyword
    pub fn is_keyword(&self, keyword: Keyword) -> bool {
        self.keyword == Some(keyword)
    }

    /// Check if this is an operator with the given value
    pub fn is_operator(&self, op: &str) -> bool {
        self.token_type == TokenType::Operator && self.literal == op
    }

    /// Check if this is a punctuator with the given value
    pub fn is_punctuator(&self, punct: &str) -> bool {
        self.token_type == TokenType::Punctuator && self.literal == punct
    }
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.token_type == TokenType::Error {
            // For error tokens, literal contains the error message
            write!(
                f,
                "{}: {} at {}",
                self.token_type, self.literal, self.position
            )
        } else if self.token_type == TokenType::Keyword {
            write!(
                f,
                "{}: {} at {}",
                self.token_type, self.literal, self.position
            )
        } else {
            write!(
                f,
                "{}: '{}' at {}",
                self.token_type, self.literal, self.position
            )
        }
    }
}

macro_rules! keywords {
    ($($variant:ident = $text:literal,)*) => {
        /// An SQL keyword, matched in any letter case
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Keyword {
            $($variant,)*
        }

        impl Keyword {
            /// Every keyword
            pub const ALL: &'static [Keyword] = &[$(Keyword::$variant,)*];

            /// The keyword in upper case, as the lexer reports it
            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Keyword::$variant => $text,)*
                }
            }
        }
    };
}

keywords! {
    Select = "SELECT",
    From = "FROM",
    Where = "WHERE",
    Insert = "INSERT",
    Into = "INTO",
    Values = "VALUES",
    Update = "UPDATE",
    Set = "SET",
    Delete = "DELETE",
    Create = "CREATE",
    Table = "TABLE",
    Drop = "DROP",
    Alter = "ALTER",
    Add = "ADD",
    Column = "COLUMN",
    And = "AND",
    Or = "OR",
    Xor = "XOR",
    Not = "NOT",
    Null = "NULL",
    Primary = "PRIMARY",
    Pragma = "PRAGMA",
    Key = "KEY",
    AutoIncrement = "AUTO_INCREMENT",
    Autoincrement = "AUTOINCREMENT",
    Default = "DEFAULT",
    As = "AS",
    Of = "OF",
    Distinct = "DISTINCT",
    Order = "ORDER",
    By = "BY",
    Asc = "ASC",
    Desc = "DESC",
    Limit = "LIMIT",
    Offset = "OFFSET",
    Group = "GROUP",
    Having = "HAVING",
    Join = "JOIN",
    Inner = "INNER",
    Outer = "OUTER",
    Left = "LEFT",
    Right = "RIGHT",
    Full = "FULL",
    On = "ON",
    Duplicate = "DUPLICATE",
    Conflict = "CONFLICT",
    Do = "DO",
    Nothing = "NOTHING",
    Using = "USING",
    Cross = "CROSS",
    Natural = "NATURAL",
    True = "TRUE",
    False = "FALSE",
    Integer = "INTEGER",
    Float = "FLOAT",
    Text = "TEXT",
    Boolean = "BOOLEAN",
    Bool = "BOOL",
    Timestamp = "TIMESTAMP",
    Timestamptz = "TIMESTAMPTZ",
    Datetime = "DATETIME",
    Date = "DATE",
    Time = "TIME",
    Json = "JSON",
    Vector = "VECTOR",
    Case = "CASE",
    Cast = "CAST",
    Extract = "EXTRACT",
    When = "WHEN",
    Then = "THEN",
    Else = "ELSE",
    End = "END",
    Between = "BETWEEN",
    In = "IN",
    Is = "IS",
    Like = "LIKE",
    Ilike = "ILIKE",
    Escape = "ESCAPE",
    Glob = "GLOB",
    Regexp = "REGEXP",
    Rlike = "RLIKE",
    Exists = "EXISTS",
    All = "ALL",
    Any = "ANY",
    Some = "SOME",
    If = "IF",
    Union = "UNION",
    Intersect = "INTERSECT",
    Except = "EXCEPT",
    With = "WITH",
    Unique = "UNIQUE",
    Check = "CHECK",
    Foreign = "FOREIGN",
    References = "REFERENCES",
    Show = "SHOW",
    Describe = "DESCRIBE",
    Tables = "TABLES",
    Views = "VIEWS",
    Indexes = "INDEXES",
    Cascade = "CASCADE",
    Restrict = "RESTRICT",
    Index = "INDEX",
    View = "VIEW",
    Trigger = "TRIGGER",
    Procedure = "PROCEDURE",
    Function = "FUNCTION",
    Returning = "RETURNING",
    Over = "OVER",
    Partition = "PARTITION",
    Range = "RANGE",
    Rows = "ROWS",
    Window = "WINDOW",
    Unbounded = "UNBOUNDED",
    Begin = "BEGIN",
    Transaction = "TRANSACTION",
    Commit = "COMMIT",
    Rollback = "ROLLBACK",
    Savepoint = "SAVEPOINT",
    Preceding = "PRECEDING",
    Following = "FOLLOWING",
    Current = "CURRENT",
    Row = "ROW",
    Modify = "MODIFY",
    Rename = "RENAME",
    To = "TO",
    Varchar = "VARCHAR",
    Char = "CHAR",
    String = "STRING",
    Bigint = "BIGINT",
    Tinyint = "TINYINT",
    Smallint = "SMALLINT",
    Real = "REAL",
    Double = "DOUBLE",
    Decimal = "DECIMAL",
    Numeric = "NUMERIC",
    Int = "INT",
    Isolation = "ISOLATION",
    Level = "LEVEL",
    Read = "READ",
    Committed = "COMMITTED",
    Uncommitted = "UNCOMMITTED",
    Interval = "INTERVAL",
    Recursive = "RECURSIVE",
    Nulls = "NULLS",
    First = "FIRST",
    Last = "LAST",
    Truncate = "TRUNCATE",
    Filter = "FILTER",
    Explain = "EXPLAIN",
    Analyze = "ANALYZE",
    Fetch = "FETCH",
    Next = "NEXT",
    Only = "ONLY",
    Vacuum = "VACUUM",
    Copy = "COPY",
    Format = "FORMAT",
    Header = "HEADER",
    Delimiter = "DELIMITER",
}

/// Longer than every keyword
const KEYWORD_MAX_LEN: usize = 16;

/// More than the keywords that share any one bucket
const KEYWORD_BUCKET_WIDTH: usize = 8;

/// A word's bucket: its length and first letter, the same in either case
const fn keyword_bucket(word: &[u8]) -> usize {
    word.len() * 32 + (word[0] & 31) as usize
}

/// Keywords by bucket, each with its spelling packed as `packed_upper`
/// packs a word, built at compile time, so a lookup compares a few numbers
static KEYWORD_BUCKETS: [[(u128, Option<Keyword>); KEYWORD_BUCKET_WIDTH];
    (KEYWORD_MAX_LEN + 1) * 32] = {
    let mut buckets = [[(0, None); KEYWORD_BUCKET_WIDTH]; (KEYWORD_MAX_LEN + 1) * 32];
    let mut i = 0;
    while i < Keyword::ALL.len() {
        let keyword = Keyword::ALL[i];
        let bucket = keyword_bucket(keyword.as_str().as_bytes());
        let mut slot = 0;
        while buckets[bucket][slot].1.is_some() {
            slot += 1;
        }
        buckets[bucket][slot] = (u128::from_le_bytes(SPELLINGS[i]), Some(keyword));
        i += 1;
    }
    buckets
};

/// The ASCII letters of the eight bytes of `x` in upper case, every other
/// byte as it is
#[inline]
const fn upper8(x: u64) -> u64 {
    const ONES: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = ONES * 0x80;
    // Below 0x80, a byte gains its high bit from b'a' on, and again past b'z'
    let low = x & !HIGH;
    let from_a = low + ONES * (0x80 - b'a' as u64);
    let past_z = low + ONES * (0x80 - b'z' as u64 - 1);
    x - ((from_a & !past_z & !x & HIGH) >> 2)
}

/// `bytes`, 1 to 16 of them, little-endian in a number and zero past their
/// end, their ASCII letters in upper case; read in at most four loads
#[inline]
fn packed_upper(bytes: &[u8]) -> u128 {
    let len = bytes.len();
    let load = |at: usize, width: usize| {
        let mut word = [0; 8];
        word[..width].copy_from_slice(&bytes[at..at + width]);
        u64::from_le_bytes(word)
    };
    let (low, high) = match len {
        8.. => (
            load(0, 8),
            (load(len - 8, 8) >> (8 * (8 - (len - 8)) % 64)) * u64::from(len > 8),
        ),
        4.. => (
            load(0, 4) | ((load(len - 4, 4) >> (8 * (8 - len))) << 32),
            0,
        ),
        _ => (
            bytes
                .iter()
                .rev()
                .fold(0, |word, b| word << 8 | u64::from(*b)),
            0,
        ),
    };
    u128::from(upper8(high)) << 64 | u128::from(upper8(low))
}

/// Each keyword's upper-case spelling, by keyword, zero past its end
static SPELLINGS: [[u8; KEYWORD_MAX_LEN]; Keyword::ALL.len()] = {
    let mut spellings = [[0; KEYWORD_MAX_LEN]; Keyword::ALL.len()];
    let mut i = 0;
    while i < Keyword::ALL.len() {
        let text = Keyword::ALL[i].as_str().as_bytes();
        let mut j = 0;
        while j < text.len() {
            spellings[i][j] = text[j];
            j += 1;
        }
        i += 1;
    }
    spellings
};

/// The keyword `word` spells, in any letter case
#[inline]
pub fn lookup_keyword(word: &str) -> Option<Keyword> {
    let bytes = word.as_bytes();
    if bytes.is_empty() || bytes.len() > KEYWORD_MAX_LEN {
        return None;
    }
    let word = packed_upper(bytes);
    for &(spelled, keyword) in &KEYWORD_BUCKETS[keyword_bucket(bytes)] {
        match keyword {
            Some(keyword) if spelled == word => return Some(keyword),
            Some(_) => {}
            None => return None,
        }
    }
    None
}

/// SQL operators, the ones `spells_operator` matches
pub static OPERATORS: &[&str] = &[
    "=", ">", "<", ">=", "<=", "<>", "!=", "+", "-", "*", "/", "%", "||", "->", "->>", "#>", "#>>",
    "@>", "<@", "?", "?|", "?&", "&", "|", "^", "~", "<<", ">>", "<=>",
];

/// Whether `bytes` spell an SQL operator
#[inline]
fn spells_operator(bytes: &[u8]) -> bool {
    matches!(
        bytes,
        b"=" | b">" | b"<" | b">=" | b"<=" | b"<>" | b"!=" | b"+" | b"-" | b"*" | b"/" | b"%"
            // String concatenation
            | b"||"
            // JSON operators, paths, containment and existence
            | b"->" | b"->>" | b"#>" | b"#>>" | b"@>" | b"<@" | b"?" | b"?|" | b"?&"
            // Bitwise operators
            | b"&" | b"|" | b"^" | b"~" | b"<<" | b">>"
            // Vector distance operator
            | b"<=>"
    )
}

/// How many bytes of the operator `bytes` start with, read greedily: three
/// when their first two and three spell operators, two when their first two
/// do, one when their first does, else none
#[inline]
pub fn operator_len(bytes: &[u8]) -> usize {
    let spells = |len: usize| bytes.len() >= len && spells_operator(&bytes[..len]);
    match (spells(2), spells(3)) {
        (true, true) => 3,
        (true, false) => 2,
        _ => usize::from(spells(1)),
    }
}

/// Check if a string is an SQL operator
#[inline]
pub fn is_operator(s: &str) -> bool {
    spells_operator(s.as_bytes())
}

/// SQL punctuators
pub static PUNCTUATORS: &[char] = &[',', ';', '(', ')', '.', ':', '[', ']'];

/// Check if a character is an SQL punctuator
pub fn is_punctuator(c: char) -> bool {
    PUNCTUATORS.contains(&c)
}

/// Get a static string for a punctuator character (avoids allocation)
/// Returns None if the character is not a punctuator
#[inline]
pub fn punctuator_str(c: char) -> Option<&'static str> {
    match c {
        ',' => Some(","),
        ';' => Some(";"),
        '(' => Some("("),
        ')' => Some(")"),
        '.' => Some("."),
        ':' => Some(":"),
        '[' => Some("["),
        ']' => Some("]"),
        _ => None,
    }
}

/// Characters that can be part of an operator
pub fn is_operator_char(c: char) -> bool {
    // Don't include # as an operator character since it's used for comments
    matches!(
        c,
        '=' | '<'
            | '>'
            | '!'
            | '+'
            | '-'
            | '*'
            | '/'
            | '%'
            | '|'
            | '&'
            | '^'
            | '~'
            | '?'
            | '@'
            | ':'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_display() {
        let pos = Position::new(10, 2, 5);
        assert_eq!(pos.to_string(), "line 2, column 5");
    }

    #[test]
    fn test_every_keyword_looks_up_in_any_case_and_no_near_word_does() {
        for keyword in Keyword::ALL {
            let text = keyword.as_str();
            let mixed: String = text
                .chars()
                .enumerate()
                .map(|(i, c)| match i % 2 {
                    0 => c.to_ascii_lowercase(),
                    _ => c,
                })
                .collect();
            for spelled in [text.to_string(), text.to_ascii_lowercase(), mixed] {
                assert_eq!(lookup_keyword(&spelled), Some(*keyword), "{spelled}");
            }
            let near = [
                text[..text.len() - 1].to_string(),
                format!("{text}X"),
                format!("{}Q", &text[..text.len() - 1]),
                format!("{text}é"),
            ];
            for word in near {
                assert!(
                    lookup_keyword(&word).is_none_or(|found| found.as_str() == word),
                    "{word} looked up as a keyword"
                );
            }
        }
    }

    #[test]
    fn test_a_word_packs_its_bytes_in_order_in_upper_case() {
        let text = "abcdefghijklmnop";
        for len in 1..=16 {
            let word = &text[..len];
            let mut want = [0u8; 16];
            want[..len].copy_from_slice(word.to_ascii_uppercase().as_bytes());
            assert_eq!(
                packed_upper(word.as_bytes()),
                u128::from_le_bytes(want),
                "{word}"
            );
        }
        assert_eq!(
            packed_upper("a_Z{@`".as_bytes()),
            u128::from_le_bytes(*b"A_Z{@`\0\0\0\0\0\0\0\0\0\0")
        );
    }

    #[test]
    fn test_the_operator_match_spells_the_operators_listed() {
        let chars = "=<>!+-*/%|&^~?@#:".as_bytes();
        let mut spelled = Vec::new();
        for len in 1..=3 {
            let mut word = vec![0; len];
            for code in 0..chars.len().pow(len as u32) {
                let mut code = code;
                for byte in word.iter_mut() {
                    *byte = chars[code % chars.len()];
                    code /= chars.len();
                }
                if spells_operator(&word) {
                    spelled.push(String::from_utf8(word.clone()).unwrap());
                }
            }
        }
        let mut listed: Vec<String> = OPERATORS.iter().map(|s| s.to_string()).collect();
        spelled.sort();
        listed.sort();
        assert_eq!(spelled, listed);
    }

    #[test]
    fn test_token_type_display() {
        assert_eq!(TokenType::Keyword.to_string(), "KEYWORD");
        assert_eq!(TokenType::Identifier.to_string(), "IDENTIFIER");
        assert_eq!(TokenType::String.to_string(), "STRING");
        assert_eq!(TokenType::Eof.to_string(), "EOF");
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1));
        assert_eq!(token.token_type, TokenType::Keyword);
        assert_eq!(token.literal, "SELECT");
        assert!(token.is_keyword(Keyword::Select));
        assert!(!token.is_keyword(Keyword::From));
    }

    #[test]
    fn test_error_token() {
        let token = Token::error("unexpected character", "x", Position::new(5, 1, 6));
        assert!(token.is_error());
        // Error message is stored in literal field
        assert_eq!(token.literal.as_str(), "unexpected character");
    }

    #[test]
    fn test_eof_token() {
        let token = Token::eof(Position::new(100, 5, 10));
        assert!(token.is_eof());
        assert_eq!(token.literal, "");
    }

    #[test]
    fn test_lookup_keyword() {
        assert_eq!(lookup_keyword("SELECT"), Some(Keyword::Select));
        assert_eq!(lookup_keyword("select"), Some(Keyword::Select));
        assert_eq!(lookup_keyword("Select"), Some(Keyword::Select));
        assert_eq!(
            lookup_keyword("auto_increment"),
            Some(Keyword::AutoIncrement)
        );
        assert_eq!(lookup_keyword("SELEC"), None);
        assert_eq!(lookup_keyword("mycolumn"), None);
        assert_eq!(lookup_keyword("ın"), None);
        assert_eq!(lookup_keyword(""), None);
        for &keyword in Keyword::ALL {
            assert!(keyword.as_str().len() <= KEYWORD_MAX_LEN, "{keyword:?}");
            assert_eq!(lookup_keyword(keyword.as_str()), Some(keyword));
            let lower = keyword.as_str().to_lowercase();
            assert_eq!(lookup_keyword(&lower), Some(keyword));
        }
    }

    #[test]
    fn test_is_operator() {
        assert!(is_operator("="));
        assert!(is_operator(">="));
        assert!(is_operator("->"));
        assert!(is_operator("->>"));
        assert!(!is_operator("==="));
    }

    #[test]
    fn test_is_punctuator() {
        assert!(is_punctuator(','));
        assert!(is_punctuator(';'));
        assert!(is_punctuator('('));
        assert!(is_punctuator(')'));
        assert!(!is_punctuator('x'));
    }

    #[test]
    fn test_is_operator_char() {
        assert!(is_operator_char('='));
        assert!(is_operator_char('+'));
        assert!(is_operator_char('-'));
        assert!(is_operator_char('|'));
        assert!(!is_operator_char('a'));
        assert!(!is_operator_char('#')); // # is for comments, not operators
    }

    #[test]
    fn test_token_display() {
        let keyword = Token::new(TokenType::Keyword, "SELECT", Position::new(0, 1, 1));
        assert!(keyword.to_string().contains("KEYWORD: SELECT"));

        let string = Token::new(TokenType::String, "hello", Position::new(7, 1, 8));
        assert!(string.to_string().contains("STRING: 'hello'"));

        let error = Token::error("bad token", "x", Position::new(0, 1, 1));
        assert!(error.to_string().contains("ERROR: bad token"));
    }
}
