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

//! Operator precedence levels for the Pratt parser

/// Precedence levels (higher number = higher precedence)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
#[derive(Default)]
pub enum Precedence {
    /// Lowest precedence
    #[default]
    Lowest = 1,
    /// Logical operators (OR)
    Or = 2,
    /// Logical operators (AND)
    And = 3,
    /// NOT operator
    Not = 4,
    /// Comparison operators (=, <>, !=, IS, LIKE, IN, BETWEEN)
    Equals = 5,
    /// Comparison operators (<, >, <=, >=)
    LessGreater = 6,
    /// Bitwise OR (|)
    BitwiseOr = 7,
    /// Bitwise XOR (^)
    BitwiseXor = 8,
    /// Bitwise AND (&)
    BitwiseAnd = 9,
    /// Bitwise shift (<<, >>)
    BitwiseShift = 10,
    /// Addition and subtraction (+, -, ||)
    Sum = 11,
    /// Multiplication and division (*, /, %)
    Product = 12,
    /// Prefix operators (-, +, NOT, ~)
    Prefix = 13,
    /// Function calls
    Call = 14,
    /// Array/JSON index access ([])
    Index = 15,
    /// Dot notation (table.column)
    Dot = 16,
}

impl Precedence {
    /// Get precedence for an operator string
    pub fn for_operator(op: &str) -> Precedence {
        match op.to_uppercase().as_str() {
            // Logical
            "OR" => Precedence::Or,
            "XOR" => Precedence::Or, // XOR has same precedence as OR
            "AND" => Precedence::And,
            "NOT" => Precedence::Not,

            // Comparison
            "=" | "<>" | "!=" | "IS" | "LIKE" | "ILIKE" | "GLOB" | "REGEXP" | "RLIKE" | "IN"
            | "BETWEEN" => Precedence::Equals,
            // AS has lowest precedence to prevent it from being consumed by NOT/AND/OR etc.
            // e.g., "NOT TRUE AS alias" should parse as "(NOT TRUE) AS alias"
            "AS" => Precedence::Lowest,
            "<" | ">" | "<=" | ">=" => Precedence::LessGreater,

            // Bitwise operators (lower to higher precedence)
            "|" => Precedence::BitwiseOr,
            "^" => Precedence::BitwiseXor,
            "&" => Precedence::BitwiseAnd,
            "<<" | ">>" => Precedence::BitwiseShift,

            // Arithmetic
            "+" | "-" | "||" => Precedence::Sum,
            "*" | "/" | "%" => Precedence::Product,

            // Access
            "." => Precedence::Dot,
            "(" => Precedence::Call,
            "[" => Precedence::Index,

            // JSON operators
            "->" | "->>" => Precedence::Index,

            _ => Precedence::Lowest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precedence_ordering() {
        assert!(Precedence::Product > Precedence::Sum);
        assert!(Precedence::Sum > Precedence::LessGreater);
        assert!(Precedence::LessGreater > Precedence::Equals);
        assert!(Precedence::And > Precedence::Or);
        assert!(Precedence::Dot > Precedence::Call);
    }

    #[test]
    fn test_operator_precedence() {
        assert_eq!(Precedence::for_operator("+"), Precedence::Sum);
        assert_eq!(Precedence::for_operator("*"), Precedence::Product);
        assert_eq!(Precedence::for_operator("AND"), Precedence::And);
        assert_eq!(Precedence::for_operator("OR"), Precedence::Or);
        assert_eq!(Precedence::for_operator("="), Precedence::Equals);
        assert_eq!(Precedence::for_operator("."), Precedence::Dot);
    }
}
