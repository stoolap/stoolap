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

//! V5 format building blocks. Production emission remains disabled until the
//! durable identity, catalog bootstrap and remover protocol land together.

pub mod column_block;
pub mod compression;
pub mod directory;
pub mod directory_reader;
pub mod directory_writer;
pub mod envelope;
pub mod file_backed;
pub mod group_metadata;
pub mod metadata_runs;
pub mod page_io;
pub mod payload_writer;
pub mod row_identity;
pub mod row_spool;
