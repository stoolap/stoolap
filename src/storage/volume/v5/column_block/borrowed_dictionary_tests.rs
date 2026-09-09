// Copyright 2026 Stoolap Contributors
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

use super::*;

fn expected(rows: u32) -> ColumnExpectation {
    ColumnExpectation {
        identity: ColumnIdentity {
            physical_column: 2,
            group: 9,
            row_start: 41,
            row_count: rows,
            data_type: DataType::Text,
        },
        vector_dimensions: None,
    }
}
fn encode(expect: ColumnExpectation, nulls: &[bool], input: ColumnInput<'_>) -> Vec<u8> {
    let plan = ColumnEncodePlan::new(expect, nulls, input, ColumnLimits::default()).unwrap();
    let mut output = vec![0xa5; plan.encoded_len() + 8];
    assert_eq!(plan.encode_into(&mut output).unwrap(), plan.encoded_len());
    assert_eq!(output[plan.encoded_len()..], [0xa5; 8]);
    output.truncate(plan.encoded_len());
    output
}

#[test]
fn borrowed_dictionary_matches_existing_wire_and_preserves_remapped_ids() {
    let texts = [
        "",
        "\0",
        "a",
        "long heap string used repeatedly",
        "é",
        "東京",
        "🦀",
    ];
    let owned: Vec<SmartString> = texts.iter().map(|s| SmartString::from(*s)).collect();
    let mut blob = Vec::new();
    let mut offsets = vec![0];
    for text in texts {
        blob.extend_from_slice(text.as_bytes());
        offsets.push(blob.len() as u32);
    }
    let dictionary = BorrowedTextDictionary::new(&blob, &offsets).unwrap();
    assert_eq!(dictionary.len(), texts.len());
    assert!(!dictionary.is_empty());
    assert_eq!(dictionary.iter().collect::<Vec<_>>(), texts);
    for (i, text) in texts.iter().enumerate() {
        assert_eq!(dictionary.get(i), Some(*text));
        assert_eq!(
            dictionary.get(i).unwrap().as_ptr(),
            blob[offsets[i] as usize..].as_ptr()
        );
    }
    assert_eq!(dictionary.get(dictionary.len()), None);
    assert_eq!(dictionary.get(usize::MAX), None);
    // IDs are already remapped to this sorted group-local dictionary. Null
    // source IDs may be junk and become the same canonical zero as before.
    let ids = [6, 0, 3, 1, 5, 2, 4, u32::MAX];
    let nulls = [false, false, false, false, false, false, false, true];
    let expect = expected(ids.len() as u32);
    let before = encode(
        expect,
        &nulls,
        ColumnInput::DictionaryText {
            ids: &ids,
            dictionary: &owned,
        },
    );
    let after = encode(
        expect,
        &nulls,
        ColumnInput::BorrowedDictionaryText {
            ids: &ids,
            dictionary: &dictionary,
        },
    );
    assert_eq!(after, before);
    let view = ColumnBlockRef::parse(&after, expect, ColumnLimits::default()).unwrap();
    for (i, &id) in ids.iter().enumerate() {
        if nulls[i] {
            assert_eq!(view.dictionary_id(i).unwrap(), Some(None));
            assert_eq!(view.cell(i), Some(ColumnCell::Null(DataType::Text)));
        } else {
            assert_eq!(view.dictionary_id(i).unwrap(), Some(Some(id)));
            assert_eq!(view.cell(i), Some(ColumnCell::Text(texts[id as usize])));
        }
    }
    for (i, text) in texts.iter().enumerate() {
        assert_eq!(view.dictionary_lookup(text).unwrap(), Some(i as u32));
    }
    assert_eq!(view.dictionary_lookup("absent").unwrap(), None);
}

#[test]
fn borrowed_dictionary_checks_offsets_utf8_and_strict_order_once() {
    for (blob, offsets) in [
        (&b""[..], &[][..]),
        (&b"a"[..], &[1, 1][..]),
        (&b"a"[..], &[0][..]),
        (&b"a"[..], &[0, 0][..]),
        (&b"a"[..], &[0, 2][..]),
        (&b"abc"[..], &[0, 2, 1, 3][..]),
        (&b"a"[..], &[0, u32::MAX, 1][..]),
    ] {
        assert_eq!(
            BorrowedTextDictionary::new(blob, offsets).unwrap_err(),
            ColumnError::Offsets
        );
    }
    for (blob, offsets) in [
        (&b"\xff"[..], &[0, 1][..]),
        ("é".as_bytes(), &[0, 1, 2][..]),
    ] {
        assert_eq!(
            BorrowedTextDictionary::new(blob, offsets).unwrap_err(),
            ColumnError::Utf8
        );
    }
    for (blob, offsets) in [
        (&b"aa"[..], &[0, 1, 2][..]),
        (&b"ba"[..], &[0, 1, 2][..]),
        (&b"a"[..], &[0, 1, 1][..]),
        (&b""[..], &[0, 0, 0][..]),
    ] {
        assert_eq!(
            BorrowedTextDictionary::new(blob, offsets).unwrap_err(),
            ColumnError::DictionaryOrder
        );
    }
    assert_eq!(
        BorrowedTextDictionary::new(b"", &vec![0; MAX_BLOCK_ROWS as usize + 2]).unwrap_err(),
        ColumnError::DictionaryLimit
    );
    assert_eq!(
        BorrowedTextDictionary::new(
            &vec![b'a'; MAX_DECODED_BYTES + 1],
            &[0, (MAX_DECODED_BYTES + 1) as u32]
        )
        .unwrap_err(),
        ColumnError::ByteLimit
    );
}

#[test]
fn borrowed_dictionary_nulls_limits_and_output_preflight_match_existing() {
    let empty = BorrowedTextDictionary::new(b"", &[0]).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.iter().next(), None);
    let input = ColumnInput::BorrowedDictionaryText {
        ids: &[u32::MAX],
        dictionary: &empty,
    };
    let after = encode(expected(1), &[true], input);
    let before = encode(
        expected(1),
        &[true],
        ColumnInput::DictionaryText {
            ids: &[u32::MAX],
            dictionary: &[],
        },
    );
    assert_eq!(after, before);
    assert!(matches!(
        ColumnEncodePlan::new(expected(1), &[false], input, ColumnLimits::default()),
        Err(ColumnError::DictionaryId)
    ));
    let dictionary = BorrowedTextDictionary::new(b"azzz", &[0, 1, 4]).unwrap();
    let ids = [0, 1];
    let nulls = [false; 2];
    let input = ColumnInput::BorrowedDictionaryText {
        ids: &ids,
        dictionary: &dictionary,
    };
    let valid = ColumnEncodePlan::new(expected(2), &nulls, input, ColumnLimits::default()).unwrap();
    for (limits, error) in [
        (
            ColumnLimits {
                dictionary_entries: 1,
                ..ColumnLimits::default()
            },
            ColumnError::DictionaryLimit,
        ),
        (
            ColumnLimits {
                cell_bytes: 2,
                ..ColumnLimits::default()
            },
            ColumnError::CellLimit,
        ),
        (
            ColumnLimits {
                decoded_bytes: valid.encoded_len() - 1,
                ..ColumnLimits::default()
            },
            ColumnError::ByteLimit,
        ),
        (
            ColumnLimits {
                rows: 1,
                ..ColumnLimits::default()
            },
            ColumnError::RowLimit,
        ),
    ] {
        assert!(
            matches!(ColumnEncodePlan::new(expected(2), &nulls, input, limits), Err(found) if found==error)
        );
    }
    assert!(matches!(
        ColumnEncodePlan::new(
            expected(1),
            &[false],
            ColumnInput::BorrowedDictionaryText {
                ids: &[0],
                dictionary: &dictionary
            },
            ColumnLimits::default()
        ),
        Err(ColumnError::DictionaryLimit)
    ));
    assert!(matches!(
        ColumnEncodePlan::new(
            expected(2),
            &nulls,
            ColumnInput::BorrowedDictionaryText {
                ids: &[0, u32::MAX],
                dictionary: &dictionary
            },
            ColumnLimits::default()
        ),
        Err(ColumnError::DictionaryId)
    ));
    let mut short = vec![0xa5; valid.encoded_len() - 1];
    assert_eq!(
        valid.encode_into(&mut short).unwrap_err(),
        ColumnError::OutputTooShort
    );
    assert!(short.iter().all(|&byte| byte == 0xa5));
    let wrong_type = ColumnExpectation {
        identity: ColumnIdentity {
            data_type: DataType::Json,
            ..expected(2).identity
        },
        ..expected(2)
    };
    assert!(matches!(
        ColumnEncodePlan::new(wrong_type, &nulls, input, ColumnLimits::default()),
        Err(ColumnError::Type)
    ));
}

#[test]
fn borrowed_dictionary_rows_and_large_repeated_cells_stay_group_bounded() {
    for rows in [4095, 4096] {
        let text = "x".repeat(256 * 1024);
        let offsets = [0, text.len() as u32];
        let dictionary = BorrowedTextDictionary::new(text.as_bytes(), &offsets).unwrap();
        let ids = vec![0; rows];
        let nulls = vec![false; rows];
        let input = ColumnInput::BorrowedDictionaryText {
            ids: &ids,
            dictionary: &dictionary,
        };
        let plan = ColumnEncodePlan::new(
            expected(rows as u32),
            &nulls,
            input,
            ColumnLimits::default(),
        )
        .unwrap();
        // The large entry is stored once, independent of its row multiplicity.
        assert_eq!(
            plan.encoded_len(),
            HEADER_BYTES + rows.div_ceil(8) + 4 * rows + 8 + text.len()
        );
        let encoded = encode(expected(rows as u32), &nulls, input);
        let block = ColumnBlockRef::parse(&encoded, expected(rows as u32), ColumnLimits::default())
            .unwrap();
        assert_eq!(block.cell(rows - 1), Some(ColumnCell::Text(&text)));
    }
    // A borrowed proof pointer keeps the input enum's largest payload equal to
    // the existing Variable { data, offsets } payload, with one discriminant.
    #[cfg(target_pointer_width = "64")]
    assert_eq!(std::mem::size_of::<ColumnInput<'_>>(), 40);
}
