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

//! Holds on the keys a foreign key references. A child row's check shares
//! its parent key and a parent's delete or key change holds the key alone,
//! each until its transaction ends, so the two never both commit

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::common::{new_i64_map, I64Map, SmartString};
use crate::core::{Error, Result, Value};

/// The transactions holding one key
#[derive(Default)]
struct Hold {
    /// The transaction that deletes or changes the key, 0 for none
    alone: i64,
    shared: SmallVec<[i64; 2]>,
}

struct Holds {
    /// Each referenced column with the holds on its keys
    columns: SmallVec<[(SmartString, FxHashMap<Value, Hold>); 1]>,
    /// The keys each transaction holds, by column position, for its release
    by_txn: I64Map<SmallVec<[(usize, Value); 1]>>,
}

/// The key holds of one table
pub struct KeyHolds(Mutex<Holds>);

impl Default for KeyHolds {
    fn default() -> Self {
        Self(Mutex::new(Holds {
            columns: SmallVec::new(),
            by_txn: new_i64_map(),
        }))
    }
}

impl KeyHolds {
    /// Holds `value` of `column` for `txn_id`, alone or shared. A hold that
    /// conflicts refuses it while `open` says its transaction has not ended.
    /// True when it is the transaction's first hold on this table
    pub fn hold(
        &self,
        table: &str,
        column: &str,
        value: &Value,
        txn_id: i64,
        alone: bool,
        open: impl Fn(i64) -> bool,
    ) -> Result<bool> {
        let mut holds = self.0.lock();
        let holds = &mut *holds;
        let slot = match holds.columns.iter().position(|(c, _)| c.as_str() == column) {
            Some(slot) => slot,
            None => {
                holds
                    .columns
                    .push((SmartString::from(column), FxHashMap::default()));
                holds.columns.len() - 1
            }
        };
        let hold = holds.columns[slot].1.entry(value.clone()).or_default();
        if hold.alone != 0 && hold.alone != txn_id {
            if open(hold.alone) {
                return Err(Self::conflict(table, column, value, hold.alone));
            }
            hold.alone = 0;
        }
        if alone {
            hold.shared.retain(|h| *h == txn_id || open(*h));
            if let Some(&other) = hold.shared.iter().find(|&&h| h != txn_id) {
                return Err(Self::conflict(table, column, value, other));
            }
        }
        let held = hold.alone == txn_id || hold.shared.contains(&txn_id);
        if alone {
            hold.alone = txn_id;
        } else if !held {
            hold.shared.push(txn_id);
        }
        if held {
            return Ok(false);
        }
        let keys = holds.by_txn.entry(txn_id).or_default();
        keys.push((slot, value.clone()));
        Ok(keys.len() == 1)
    }

    /// The holds on column `old` stay its holds under the name `new` it is
    /// renamed to; holds left under `new` by a column gone since name none
    pub fn rename_column(&self, old: &str, new: &str) {
        let (old, new_lower) = (old.to_lowercase(), new.to_lowercase());
        let mut holds = self.0.lock();
        for (column, _) in holds.columns.iter_mut() {
            let lower = column.to_lowercase();
            if lower == old {
                *column = SmartString::from(new);
            } else if lower == new_lower {
                *column = SmartString::default();
            }
        }
    }

    /// Lets go of every key `txn_id` holds
    pub fn release(&self, txn_id: i64) {
        let mut holds = self.0.lock();
        let Some(keys) = holds.by_txn.remove(txn_id) else {
            return;
        };
        for (slot, value) in keys {
            let keys = &mut holds.columns[slot].1;
            let Some(hold) = keys.get_mut(&value) else {
                continue;
            };
            if hold.alone == txn_id {
                hold.alone = 0;
            }
            hold.shared.retain(|h| *h != txn_id);
            if hold.alone == 0 && hold.shared.is_empty() {
                keys.remove(&value);
            }
        }
    }

    fn conflict(table: &str, column: &str, value: &Value, holder: i64) -> Error {
        Error::internal(format!(
            "write conflict: key {column} = {value} of table '{table}' is held by transaction {holder}"
        ))
    }
}
