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

//! A call's deadline and the loops that charge their work against it

use std::cell::Cell;
use std::time::Duration;

use crate::common::time_compat::Instant;
use crate::core::Result;

/// Units of work a thread does between two reads of the clock
pub(crate) const CHARGE_BEAT: usize = 4096;

thread_local! {
    /// Units this thread charges against one deadline before it reads the
    /// clock again; nested loops of that deadline draw from it as one, and
    /// a charge against another deadline reads the clock first
    static BEAT: Cell<(usize, Option<Instant>)> = const { Cell::new((0, None)) };
}

/// A loop that charges its work against the call's deadline, by name, so
/// a test can stop a statement at each one
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeadlineSite {
    /// A prepared statement's scan of its rows
    Scan,
    /// A grouped statement's keys computed over each row its run returns
    LateKeys,
    /// A subquery block's run and its scan
    Block,
    /// A join's rows the general path judges as it streams them
    JoinedRows,
    /// A join read in parts
    JoinParts,
    /// A streamed result's rows as the caller reads them
    Result,
    /// An ordered walk of an index
    IndexWalk,
}

impl DeadlineSite {
    /// Every site, which a test stops a statement at in turn
    pub const ALL: [DeadlineSite; 7] = [
        DeadlineSite::Scan,
        DeadlineSite::LateKeys,
        DeadlineSite::Block,
        DeadlineSite::JoinedRows,
        DeadlineSite::JoinParts,
        DeadlineSite::Result,
        DeadlineSite::IndexWalk,
    ];
}

/// The instant past which a call's statements stop, copied into every
/// context and result of the call; None runs them to their end
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deadline(Option<Instant>);

impl Deadline {
    pub(crate) const NONE: Self = Self(None);

    /// Stop past `at`; None never stops
    pub(crate) fn at(at: Option<Instant>) -> Self {
        Self(at)
    }

    /// `timeout_ms` from now; 0 sets none
    pub(crate) fn after_ms(timeout_ms: u64) -> Self {
        Self((timeout_ms > 0).then(|| Instant::now() + Duration::from_millis(timeout_ms)))
    }

    /// The instant past which the statements stop, if any
    pub(crate) fn instant(self) -> Option<Instant> {
        self.0
    }

    /// Count `units` of work at `site`; the clock is read when the thread's
    /// beat runs out, and at every charge once the deadline has passed
    #[inline]
    pub(crate) fn charge(self, site: DeadlineSite, units: usize) -> Result<()> {
        match self.0 {
            None => Ok(()),
            Some(at) => charge_until(at, site, units),
        }
    }

    /// Fail when the deadline has passed, reading the clock now
    pub(crate) fn check(self) -> Result<()> {
        match self.0 {
            None => Ok(()),
            Some(at) => passed(at, None),
        }
    }
}

#[inline(never)]
fn charge_until(at: Instant, site: DeadlineSite, units: usize) -> Result<()> {
    let (left, owner) = BEAT.with(Cell::get);
    #[cfg(feature = "test-failpoints")]
    let left = match crate::test_failpoints::deadline_forced() {
        true => 0,
        false => left,
    };
    if left > units && owner == Some(at) {
        BEAT.with(|cell| cell.set((left - units, owner)));
        return Ok(());
    }
    BEAT.with(|cell| cell.set((0, None)));
    passed(at, Some(site))?;
    BEAT.with(|cell| cell.set((CHARGE_BEAT, Some(at))));
    Ok(())
}

/// QueryCancelled once `at` is past, read at `site` or at a statement's step
fn passed(at: Instant, site: Option<DeadlineSite>) -> Result<()> {
    #[cfg(feature = "test-failpoints")]
    if crate::test_failpoints::deadline_passed_at(site) {
        return Err(crate::core::Error::QueryCancelled);
    }
    #[cfg(not(feature = "test-failpoints"))]
    let _ = site;
    match Instant::now() >= at {
        true => Err(crate::core::Error::QueryCancelled),
        false => Ok(()),
    }
}
