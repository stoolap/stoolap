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

//! Durability: the WAL, snapshots, recovery and reopening a database

mod auto_increment_reopen_test;
mod bug_manifest_version_fail_closed_test;
mod bug_unique_lost_after_reopen_test;
mod checkpoint_catalog_copy_test;
mod dml_lsn_receipt_test;
mod drop_rename_replay_test;
mod durability_test;
mod failpoint_io_test;
mod file_lock_integration_test;
mod index_replay_test;
mod migration_v037_test;
mod new_table_index_replay_test;
mod open_session_test;
mod open_unreadable_volume_test;
mod persistence_advanced_test;
mod persistence_comprehensive_test;
mod persistence_debug_test;
mod persistence_index_debug_test;
mod persistence_open_error_test;
mod persistence_test;
mod persistence_trace_test;
mod persistence_wal_dump2_test;
mod persistence_wal_dump_test;
mod pragma_restore_test;
mod recovery_reseal_test;
mod retype_fixture_test;
mod rowid_fixture_test;
mod snapshot_recovery_test;
mod snapshot_system_test;
mod stored_length_test;
mod transaction_table_replay_test;
mod vacuum_test;
mod volume_checkpoint_test;
mod volume_format_test;
mod wal_acknowledged_commit_test;
mod wal_path_test;
mod wal_torn_tail_test;
mod wal_truncation_rotation_test;
mod wal_truncation_test;
mod wal_visibility_test;
