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

//! Tests live in a few binaries, one per directory under tests/, each file a
//! module its main.rs declares. A file of its own at the top is a binary of
//! its own: only a test that sets the global allocator, or one listed here

use std::fs;
use std::path::Path;

/// Top-level test files that are binaries of their own without setting the
/// global allocator: a feature of their own, or a long stress run
const OWN_BINARIES: &[&str] = &[
    "concurrency_history_test.rs",
    "crash_soak_test.rs",
    "ffi_test.rs",
    "heap_profile_test.rs",
    "layout_test.rs",
    "metamorphic_test.rs",
];

fn tests_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

#[test]
fn every_file_in_a_group_is_a_module_of_its_main() {
    let mut missing = Vec::new();
    for group in fs::read_dir(tests_dir()).unwrap() {
        let dir = group.unwrap().path();
        let main = dir.join("main.rs");
        if !main.is_file() {
            continue;
        }
        let declared = fs::read_to_string(&main).unwrap();
        for file in fs::read_dir(&dir).unwrap() {
            let path = file.unwrap().path();
            let name = path.file_name().unwrap().to_str().unwrap().to_string();
            let Some(stem) = name.strip_suffix(".rs") else {
                continue;
            };
            if stem != "main" && !declared.lines().any(|l| l == format!("mod {stem};")) {
                missing.push(path.display().to_string());
            }
        }
    }
    assert!(
        missing.is_empty(),
        "never compiled: add `mod <name>;` to the group's main.rs for {missing:?}"
    );
}

/// The literal `memory://` names and registered function names a file uses
fn shared_names(text: &str) -> Vec<String> {
    let mut names = Vec::new();
    for (start, _) in text.match_indices("\"memory://") {
        let rest = &text[start + 1..];
        let name = &rest[..rest.find('"').unwrap()];
        if !name.contains('{') {
            names.push(name.to_string());
        }
    }
    for (start, _) in text.match_indices("fn name(&self)") {
        let rest = &text[start..];
        let open = rest.find('"').unwrap() + 1;
        let close = open + rest[open..].find('"').unwrap();
        names.push(format!("function {}", rest[open..close].to_uppercase()));
    }
    names
}

#[test]
fn no_two_files_of_a_group_share_a_database_or_function_name() {
    let mut shared = Vec::new();
    for group in fs::read_dir(tests_dir()).unwrap() {
        let dir = group.unwrap().path();
        if !dir.join("main.rs").is_file() {
            continue;
        }
        let mut owner = std::collections::HashMap::new();
        for file in fs::read_dir(&dir).unwrap() {
            let path = file.unwrap().path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            for name in shared_names(&fs::read_to_string(&path).unwrap()) {
                let first = owner.entry(name.clone()).or_insert_with(|| path.clone());
                if *first != path {
                    shared.push(format!(
                        "{name}: {} and {}",
                        first.display(),
                        path.display()
                    ));
                }
            }
        }
    }
    shared.sort();
    shared.dedup();
    assert!(
        shared.is_empty(),
        "tests of one group run in one process under cargo test, rename one side: {shared:#?}"
    );
}

#[test]
fn a_top_level_test_file_is_one_that_needs_a_binary_of_its_own() {
    let mut extra = Vec::new();
    for file in fs::read_dir(tests_dir()).unwrap() {
        let path = file.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_string();
        if !name.ends_with(".rs") || OWN_BINARIES.contains(&name.as_str()) {
            continue;
        }
        // CI's allocation step finds these files by this line, at its start
        if !fs::read_to_string(&path)
            .unwrap()
            .lines()
            .any(|l| l.starts_with("#[global_allocator]"))
        {
            extra.push(name);
        }
    }
    assert!(
        extra.is_empty(),
        "each top-level file links a binary of its own: move {extra:?} into the group under tests/ it tests"
    );
}
