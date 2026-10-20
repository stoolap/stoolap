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

//! A transaction's write gate: every statement its handles run is admitted
//! once, a commit, rollback or drop closes the gate and waits for those in
//! flight, and a savepoint holds new ones off while it waits

use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use crate::core::{Error, Result};

/// No statement is admitted any more
const CLOSED: u64 = 1 << 63;
/// New statements wait until a savepoint is done
const HOLD: u64 = 1 << 62;
/// The last statement out runs the transaction's deferred cleanup
const FINISH_PENDING: u64 = 1 << 61;
const IN_FLIGHT: u64 = FINISH_PENDING - 1;

type Finish = Box<dyn FnOnce() + Send>;

/// The statements of one transaction, counted in and out
pub struct TxnGate {
    state: AtomicU64,
    finish: Mutex<Option<Finish>>,
}

/// What a `GateRef` reads once its own gate went back to the pool, which
/// only its drop does
static RETURNED: TxnGate = TxnGate {
    state: AtomicU64::new(CLOSED),
    finish: Mutex::new(None),
};

/// Admissions kept inline per thread before the rest spill
const INLINE: usize = 8;

/// The gates this thread holds an outer admission of: plain cells, so a
/// statement reaches them without a lazy start or a destructor check
struct Admitted {
    len: Cell<usize>,
    inline: [Cell<usize>; INLINE],
}

thread_local! {
    static ADMITTED: Admitted = const {
        Admitted {
            len: Cell::new(0),
            inline: [const { Cell::new(0) }; INLINE],
        }
    };
    /// Admissions past the inline ones, for transactions nested deeper
    static SPILLED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// Gates of finished transactions no one holds any more
    static POOL: RefCell<Vec<Arc<TxnGate>>> = const { RefCell::new(Vec::new()) };
}

impl Admitted {
    fn contains(&self, address: usize) -> bool {
        let len = self.len.get();
        self.inline[..len.min(INLINE)]
            .iter()
            .any(|slot| slot.get() == address)
            || (len > INLINE && SPILLED.with(|spilled| spilled.borrow().contains(&address)))
    }

    fn push(&self, address: usize) {
        let len = self.len.get();
        match self.inline.get(len) {
            Some(slot) => slot.set(address),
            None => SPILLED.with(|spilled| spilled.borrow_mut().push(address)),
        }
        self.len.set(len + 1);
    }

    /// Takes `address` out, moving the last admission into its place
    fn remove(&self, address: usize) {
        let len = self.len.get();
        if len == 0 {
            return;
        }
        let last = if len > INLINE {
            SPILLED
                .with(|spilled| spilled.borrow_mut().pop())
                .unwrap_or(0)
        } else {
            self.inline[len - 1].get()
        };
        self.len.set(len - 1);
        if last == address {
            return;
        }
        if let Some(slot) = self.inline[..(len - 1).min(INLINE)]
            .iter()
            .find(|slot| slot.get() == address)
        {
            slot.set(last);
            return;
        }
        SPILLED.with(|spilled| {
            let mut spilled = spilled.borrow_mut();
            if let Some(slot) = spilled.iter_mut().find(|slot| **slot == address) {
                *slot = last;
            }
        });
    }
}

/// Gates kept for reuse per thread
const POOL_SIZE: usize = 16;

/// A shared hold on a transaction's gate; the last one let go keeps the
/// gate for the next transaction on its thread
#[derive(Clone)]
pub struct GateRef(Option<Arc<TxnGate>>);

impl std::ops::Deref for GateRef {
    type Target = TxnGate;

    fn deref(&self) -> &TxnGate {
        self.0.as_deref().unwrap_or(&RETURNED)
    }
}

impl Drop for GateRef {
    fn drop(&mut self) {
        let Some(mut gate) = self.0.take() else {
            return;
        };
        // One load spares an admission's exit the CAS of `get_mut`
        if Arc::strong_count(&gate) != 1 {
            return;
        }
        let Some(free) = Arc::get_mut(&mut gate) else {
            return;
        };
        let finish = free
            .finish
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner);
        if finish.is_some() {
            return;
        }
        *free.state.get_mut() = 0;
        let _ = POOL.try_with(|pool| {
            let mut pool = pool.borrow_mut();
            if pool.len() < POOL_SIZE {
                pool.push(gate);
            }
        });
    }
}

impl GateRef {
    /// Admits a statement, or refuses it once the gate is closed. A
    /// statement nested in one this thread already runs shares its
    /// admission, so a close that came after the outer one never cuts it
    #[inline(never)]
    pub(crate) fn admit(&self) -> Result<Admission> {
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::txn_gate_looked_up();
        let address = self.address();
        ADMITTED.with(|held| {
            if held.contains(address) {
                return Ok(Admission {
                    gate: None,
                    on_this_thread: PhantomData,
                });
            }
            let mut backoff = Backoff::default();
            loop {
                let before = self.state.fetch_add(1, Ordering::AcqRel);
                if before & (CLOSED | HOLD) == 0 {
                    break;
                }
                self.leave();
                if before & CLOSED != 0 {
                    return Err(Error::TransactionEnded);
                }
                #[cfg(feature = "test-failpoints")]
                crate::test_failpoints::txn_gate_held_off();
                backoff.snooze();
            }
            #[cfg(feature = "test-failpoints")]
            crate::test_failpoints::txn_gate_admitted();
            held.push(address);
            Ok(Admission {
                gate: Some(self.clone()),
                on_this_thread: PhantomData,
            })
        })
    }
}

impl TxnGate {
    /// An open gate, a finished transaction's when one is free
    pub(crate) fn open() -> GateRef {
        let gate = POOL
            .with(|pool| pool.borrow_mut().pop())
            .unwrap_or_else(|| {
                Arc::new(Self {
                    state: AtomicU64::new(0),
                    finish: Mutex::new(None),
                })
            });
        GateRef(Some(gate))
    }

    fn address(&self) -> usize {
        self as *const Self as usize
    }

    /// Whether this thread is inside one of the gate's statements
    pub(crate) fn held_here(&self) -> bool {
        let address = self.address();
        ADMITTED.with(|held| held.contains(address))
    }

    /// Admits a statement on a handle that has a gate; a handle owned by
    /// another, which admits for it, has none
    #[inline(always)]
    pub(crate) fn admit_for(gate: &Option<GateRef>) -> Result<Option<Admission>> {
        match gate {
            None => Ok(None),
            Some(gate) => gate.admit().map(Some),
        }
    }

    /// Admits no statement from now on
    pub(crate) fn close(&self) {
        self.state.fetch_or(CLOSED, Ordering::AcqRel);
    }

    /// Waits until no admitted statement is in flight
    pub(crate) fn wait_idle(&self) {
        if self.state.load(Ordering::Acquire) & IN_FLIGHT == 0 {
            return;
        }
        #[cfg(feature = "test-failpoints")]
        crate::test_failpoints::txn_gate_waiting();
        let mut backoff = Backoff::default();
        while self.state.load(Ordering::Acquire) & IN_FLIGHT != 0 {
            backoff.snooze();
        }
    }

    /// Holds new statements off and waits for those in flight, until the
    /// hold is let go
    pub(crate) fn hold(&self) -> Hold<'_> {
        self.state.fetch_or(HOLD, Ordering::AcqRel);
        self.wait_idle();
        Hold(self)
    }

    /// Closes the gate and leaves `finish` to the last of its statements
    /// out: the transaction is dropped from inside one of them
    pub(crate) fn finish_after_statements(&self, finish: Finish) {
        *self.finish.lock().unwrap_or_else(PoisonError::into_inner) = Some(finish);
        self.state
            .fetch_or(CLOSED | FINISH_PENDING, Ordering::AcqRel);
    }

    fn leave(&self) {
        let before = self.state.fetch_sub(1, Ordering::AcqRel);
        if before & FINISH_PENDING != 0 && before & IN_FLIGHT == 1 {
            let finish = self
                .finish
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(finish) = finish {
                finish();
            }
        }
    }
}

/// A statement admitted through a gate, counted out when dropped on the
/// thread whose list it entered
pub struct Admission {
    /// None for a statement nested in one this thread already runs
    gate: Option<GateRef>,
    on_this_thread: PhantomData<*const ()>,
}

impl Drop for Admission {
    fn drop(&mut self) {
        if let Some(gate) = self.gate.take() {
            ADMITTED.with(|held| held.remove(gate.address()));
            gate.leave();
        }
    }
}

/// A savepoint's hold on new statements
pub struct Hold<'a>(&'a TxnGate);

impl Drop for Hold<'_> {
    fn drop(&mut self) {
        self.0.state.fetch_and(!HOLD, Ordering::AcqRel);
    }
}

/// Spins, then yields, then sleeps up to a millisecond
#[derive(Default)]
struct Backoff(u32);

impl Backoff {
    fn snooze(&mut self) {
        self.0 = self.0.saturating_add(1);
        match self.0 {
            0..=64 => std::hint::spin_loop(),
            65..=128 => std::thread::yield_now(),
            #[cfg(target_arch = "wasm32")]
            _ => std::hint::spin_loop(),
            #[cfg(not(target_arch = "wasm32"))]
            n => std::thread::sleep(std::time::Duration::from_micros(u64::from(
                (n - 128).min(1000),
            ))),
        }
    }
}
