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

//! Error mapping from TiKV errors to Stoolap errors

use crate::core::Error;

/// Convert a tikv-client error to a Stoolap error
pub fn from_tikv_error(e: tikv_client::Error) -> Error {
    Error::internal(format!("TiKV error: {e}"))
}
