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

//! External signed-row-ID sorting of complete payload positions.
//!
//! Push each append receipt immediately in physical spool order. Initial runs
//! use only the caller's fixed key slice; finish binds the run stream to the
//! opaque completed spool. Two-way merge uses three bounded byte buffers and
//! distinct caller-owned streams. On error, the destination is invalid and must
//! be discarded; the source remains usable. No operation flushes, opens,
//! truncates, removes, or establishes durability for either stream.

use super::*;

pub const RECORD_BYTES: usize = 52;
pub const MAX_RUN_ROWS: usize = 65_536;

#[derive(Clone, Copy)]
pub struct RowRuns<'schema> {
    spool: FinishedSpool<'schema>,
    run_len: u64,
}
impl<'schema> RowRuns<'schema> {
    pub const fn row_count(self) -> u64 {
        self.spool.rows
    }
    pub const fn byte_len(self) -> u64 {
        self.spool.rows * RECORD_BYTES as u64
    }
    pub const fn is_sorted(self) -> bool {
        self.spool.rows <= self.run_len
    }
    pub const fn spool(self) -> FinishedSpool<'schema> {
        self.spool
    }
}

pub struct RowRunWriter<'a, 'schema, W: Write + ?Sized> {
    sink: &'a mut W,
    keys: &'a mut [RowPosition],
    output: &'a mut [u8],
    provisional: FinishedSpool<'schema>,
    len: usize,
    count: u64,
    bytes: u64,
    state: State,
}
impl<'a, 'schema, W: Write + Seek + ?Sized> RowRunWriter<'a, 'schema, W> {
    pub fn new(
        sink: &'a mut W,
        binding: SpoolBinding<'schema>,
        limits: SpoolLimits,
        keys: &'a mut [RowPosition],
        output: &'a mut [u8],
    ) -> Result<Self> {
        limits.validate()?;
        check_io(output)?;
        if keys.is_empty() || keys.len() > MAX_RUN_ROWS {
            return Err(SpoolError::Limits);
        }
        if sink.seek(SeekFrom::Start(0))? != 0 {
            return Err(SpoolError::InvalidIoCount);
        }
        Ok(Self {
            sink,
            keys,
            output,
            provisional: FinishedSpool {
                binding,
                fingerprint: binding.fingerprint(),
                limits,
                rows: limits.rows,
                bytes: limits.spool_bytes,
            },
            len: 0,
            count: 0,
            bytes: 0,
            state: State::Open,
        })
    }
    pub fn push(&mut self, row: RowPosition) -> Result<()> {
        self.state.check()?;
        self.provisional.validate_position(row)?;
        if row.offset != self.bytes || self.count == self.provisional.limits.rows {
            return Err(SpoolError::Identity);
        }
        self.state = State::Poisoned;
        self.keys[self.len] = row;
        self.len += 1;
        self.count += 1;
        self.bytes += row.len; // validate_position checked end against the spool cap.
        if self.len == self.keys.len() {
            self.flush_run()?;
        }
        self.state = State::Open;
        Ok(())
    }
    pub fn finish(&mut self, spool: FinishedSpool<'schema>) -> Result<RowRuns<'schema>> {
        self.state.check()?;
        if self.count != spool.rows
            || self.bytes != spool.bytes
            || !same_binding(self.provisional.binding, spool.binding)
        {
            return Err(SpoolError::Identity);
        }
        self.state = State::Poisoned;
        self.flush_run()?;
        self.state = State::Finished;
        Ok(RowRuns {
            spool,
            run_len: self.keys.len() as u64,
        })
    }
    fn flush_run(&mut self) -> Result<()> {
        let rows = &mut self.keys[..self.len];
        rows.sort_unstable_by_key(|row| row.row_id);
        if rows.windows(2).any(|pair| pair[0].row_id >= pair[1].row_id) {
            return Err(SpoolError::RowOrder);
        }
        let mut output = RecordOutput {
            sink: self.sink,
            buffer: self.output,
            len: 0,
        };
        for &row in rows.iter() {
            output.push(row)?;
        }
        output.finish()?;
        self.len = 0;
        Ok(())
    }
}
fn same_binding(a: SpoolBinding<'_>, b: SpoolBinding<'_>) -> bool {
    a.identity == b.identity
        && a.schema_version == b.schema_version
        && std::ptr::eq(a.columns, b.columns)
        && a.source.legacy_base() == b.source.legacy_base()
}

/// Rewrites all adjacent run pairs into the other exclusive stream. Source and
/// destination must not alias; scratch lifetime and actual file ownership are
/// the caller's responsibility. A failed pass returns no publishable summary.
pub fn merge_pass<'schema, R: ReadAt + ?Sized, W: Write + Seek + ?Sized>(
    source: &R,
    sink: &mut W,
    runs: RowRuns<'schema>,
    left: &mut [u8],
    right: &mut [u8],
    output: &mut [u8],
) -> Result<RowRuns<'schema>> {
    check_io(left)?;
    check_io(right)?;
    check_io(output)?;
    if runs.is_sorted() {
        return Err(SpoolError::AlreadySorted);
    }
    if sink.seek(SeekFrom::Start(0))? != 0 {
        return Err(SpoolError::InvalidIoCount);
    }
    let step = runs.run_len.checked_mul(2).ok_or(SpoolError::Overflow)?;
    let mut destination = RecordOutput {
        sink,
        buffer: output,
        len: 0,
    };
    let mut start = 0;
    while start < runs.spool.rows {
        let middle = start.saturating_add(runs.run_len).min(runs.spool.rows);
        let end = start.saturating_add(step).min(runs.spool.rows);
        let mut a = RunCursor::new(source, runs.spool, start, middle, left);
        let mut b = RunCursor::new(source, runs.spool, middle, end, right);
        let mut x = a.next()?;
        let mut y = b.next()?;
        let mut previous = None;
        while x.is_some() || y.is_some() {
            let row = match (x, y) {
                (Some(v), Some(w)) if v.row_id <= w.row_id => {
                    x = a.next()?;
                    v
                }
                (Some(_), Some(w)) => {
                    y = b.next()?;
                    w
                }
                (Some(v), None) => {
                    x = a.next()?;
                    v
                }
                (None, Some(w)) => {
                    y = b.next()?;
                    w
                }
                (None, None) => unreachable!(),
            };
            if previous.is_some_and(|id| id >= row.row_id) {
                return Err(SpoolError::RowOrder);
            }
            destination.push(row)?;
            previous = Some(row.row_id);
        }
        start = end;
    }
    destination.finish()?;
    Ok(RowRuns {
        spool: runs.spool,
        run_len: step,
    })
}

/// Buffered stream of sorted immutable receipts; gather can batch these into a
/// fixed RowPosition slice while retaining the exact original payload backing.
pub struct SortedRows<'a, 'schema, R: ReadAt + ?Sized> {
    cursor: RunCursor<'a, 'schema, R>,
    state: State,
}
impl<'a, 'schema, R: ReadAt + ?Sized> SortedRows<'a, 'schema, R> {
    pub fn new(source: &'a R, runs: RowRuns<'schema>, scratch: &'a mut [u8]) -> Result<Self> {
        check_io(scratch)?;
        if !runs.is_sorted() {
            return Err(SpoolError::NotSorted);
        }
        Ok(Self {
            cursor: RunCursor::new(source, runs.spool, 0, runs.spool.rows, scratch),
            state: State::Open,
        })
    }
    pub fn next_position(&mut self) -> Result<Option<RowPosition>> {
        self.state.check()?;
        self.state = State::Poisoned;
        let row = self.cursor.next()?;
        self.state = State::Open;
        Ok(row)
    }
}
struct RunCursor<'a, 'schema, R: ReadAt + ?Sized> {
    source: &'a R,
    spool: FinishedSpool<'schema>,
    index: u64,
    end: u64,
    buffer: &'a mut [u8],
    at: usize,
    len: usize,
    previous: Option<i64>,
}
impl<'a, 'schema, R: ReadAt + ?Sized> RunCursor<'a, 'schema, R> {
    fn new(
        source: &'a R,
        spool: FinishedSpool<'schema>,
        index: u64,
        end: u64,
        buffer: &'a mut [u8],
    ) -> Self {
        Self {
            source,
            spool,
            index,
            end,
            buffer,
            at: 0,
            len: 0,
            previous: None,
        }
    }
    fn next(&mut self) -> Result<Option<RowPosition>> {
        if self.index == self.end {
            return Ok(None);
        }
        if self.at == self.len {
            let count =
                (self.end - self.index).min((self.buffer.len() / RECORD_BYTES) as u64) as usize;
            self.len = 0;
            read_exact(
                self.source,
                self.index * RECORD_BYTES as u64,
                &mut self.buffer[..count * RECORD_BYTES],
            )?;
            self.at = 0;
            self.len = count * RECORD_BYTES;
        }
        let row = decode(&self.buffer[self.at..self.at + RECORD_BYTES])?;
        self.spool.validate_position(row)?;
        if self.previous.is_some_and(|id| id >= row.row_id) {
            return Err(SpoolError::RowOrder);
        }
        self.previous = Some(row.row_id);
        self.index += 1;
        self.at += RECORD_BYTES;
        Ok(Some(row))
    }
}
struct RecordOutput<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    buffer: &'a mut [u8],
    len: usize,
}
impl<W: Write + ?Sized> RecordOutput<'_, W> {
    fn push(&mut self, row: RowPosition) -> Result<()> {
        if self.buffer.len() - self.len < RECORD_BYTES {
            self.finish()?;
        }
        self.buffer[self.len..self.len + RECORD_BYTES].copy_from_slice(&encode(row));
        self.len += RECORD_BYTES;
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        write_all(self.sink, &self.buffer[..self.len])?;
        self.len = 0;
        Ok(())
    }
}
fn encode(row: RowPosition) -> [u8; RECORD_BYTES] {
    let mut bytes = [0; RECORD_BYTES];
    bytes[..8].copy_from_slice(&row.row_id.to_le_bytes());
    bytes[8..16].copy_from_slice(&row.creator.to_le_bytes());
    bytes[16..24].copy_from_slice(&row.source.to_le_bytes());
    bytes[24..32].copy_from_slice(&row.offset.to_le_bytes());
    bytes[32..40].copy_from_slice(&row.len.to_le_bytes());
    bytes[40..44].copy_from_slice(&row.header_crc.to_le_bytes());
    let crc = crc32fast::hash(&bytes[..48]);
    bytes[48..].copy_from_slice(&crc.to_le_bytes());
    bytes
}
fn decode(bytes: &[u8]) -> Result<RowPosition> {
    if crc32fast::hash(&bytes[..48]) != u32_at(bytes, 48) {
        return Err(SpoolError::Checksum);
    }
    if bytes[44..48] != [0; 4] {
        return Err(SpoolError::Tag);
    }
    Ok(RowPosition {
        row_id: i64_at(bytes, 0),
        creator: i64_at(bytes, 8),
        source: u64_at(bytes, 16),
        offset: u64_at(bytes, 24),
        len: u64_at(bytes, 32),
        header_crc: u32_at(bytes, 40),
    })
}
