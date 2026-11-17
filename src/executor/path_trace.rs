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

//! The registered paths a run took, noted while an EXPLAIN ANALYZE runs on
//! the thread; any other run reads one flag and notes nothing

use std::cell::RefCell;
use std::sync::atomic::{AtomicUsize, Ordering};

/// The traces open on any thread
static OPEN: AtomicUsize = AtomicUsize::new(0);

/// Paths noted, by family and name, each with how many times, in the order
/// first taken
type Noted = Vec<(&'static str, &'static str, u32)>;

thread_local! {
    /// The paths this thread's open trace noted
    static TRACE: RefCell<Option<Noted>> = const { RefCell::new(None) };
}

/// Notes that path `name` of `family` was taken, while a trace is open
#[inline]
pub(crate) fn note(family: &'static str, name: &'static str) {
    if OPEN.load(Ordering::Relaxed) != 0 {
        note_open(family, name);
    }
}

#[cold]
fn note_open(family: &'static str, name: &'static str) {
    TRACE.with(|trace| {
        let mut trace = trace.borrow_mut();
        let Some(paths) = trace.as_mut() else {
            return;
        };
        match (paths.iter_mut()).find(|(f, n, _)| *f == family && *n == name) {
            Some((.., count)) => *count += 1,
            None => paths.push((family, name, 1)),
        }
    });
}

/// A trace open on this thread until it drops, when the one it opened
/// over comes back
pub(crate) struct Trace {
    over: Option<Noted>,
}

impl Trace {
    pub(crate) fn open() -> Self {
        let over = TRACE.with(|trace| trace.borrow_mut().replace(Vec::new()));
        OPEN.fetch_add(1, Ordering::Relaxed);
        Trace { over }
    }

    /// The paths noted so far, as `family.name`, a count after one taken
    /// more than once
    pub(crate) fn paths(&self) -> Vec<String> {
        TRACE.with(|trace| {
            (trace.borrow().iter().flatten())
                .map(|(family, name, count)| match count {
                    1 => format!("{family}.{name}"),
                    _ => format!("{family}.{name} x{count}"),
                })
                .collect()
        })
    }
}

impl Drop for Trace {
    fn drop(&mut self) {
        let over = self.over.take();
        TRACE.with(|trace| *trace.borrow_mut() = over);
        OPEN.fetch_sub(1, Ordering::Relaxed);
    }
}
