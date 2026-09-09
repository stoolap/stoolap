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

//! Immutable V5 file ownership and bounded, lazy page access.
//!
//! Open reads only the header, footer and 128-byte decoded root. A handle keeps
//! scalar metadata and the existing VolumeFile identity owner, with no idle FD
//! and no compressed/decoded column cache. Each active lease opens and verifies
//! one positioned-read FD. Managed rename/retirement uses that same owner.
//! Before publishing a handle in an engine generation, register its backing
//! with the existing retirement queue, including independent opens/aliases.
//!
//! Root and directory checks do not certify unread payloads or whole-tree
//! coverage. Callers validate typed payloads and required keys as they read
//! them. Legacy source decoding still requires installed checkpoint evidence.
//! Successful writer/open completion never fsyncs or acknowledges a manifest.
//! The caller supplies reserved root, directory and payload buffers, and must
//! never acquire/read a lease while holding a publication fence. Production
//! activation remains gated by the durable catalog/bootstrap protocol.

use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::storage::volume::io::{VolumeFile, VolumeReadLease};

use super::directory::RootSummary;
use super::directory_reader::{DirectoryBufferRequirements, DirectoryLookup, DirectoryReadError};
use super::envelope::{FileIdentity, Header, PageDescriptor, ReadLimits};
use super::page_io::{FinishedVolume, OpenedEnvelope, PageIoError, PageReadPlan, ReadAt};

#[derive(Debug)]
pub enum FileVolumeError {
    Io(std::io::Error),
    Page(PageIoError),
    Directory(DirectoryReadError),
    CompletedFileMismatch,
}
impl fmt::Display for FileVolumeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 file-backed volume: {self:?}")
    }
}
impl std::error::Error for FileVolumeError {}
impl From<std::io::Error> for FileVolumeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<PageIoError> for FileVolumeError {
    fn from(error: PageIoError) -> Self {
        Self::Page(error)
    }
}
impl From<DirectoryReadError> for FileVolumeError {
    fn from(error: DirectoryReadError) -> Self {
        Self::Directory(error)
    }
}

/// A clone shares only the existing physical file owner and fixed root data.
#[derive(Clone)]
pub struct FileBackedVolume {
    backing: Arc<VolumeFile>,
    envelope: OpenedEnvelope,
    summary: RootSummary,
    limits: ReadLimits,
}

impl FileBackedVolume {
    /// The file is immutable, including through other writable aliases. The
    /// expected logical identity comes from the caller's catalog/build record.
    /// Scratch must already be reserved; no payload-sized allocation occurs.
    pub fn open(
        path: &Path,
        expected: FileIdentity,
        limits: ReadLimits,
        root_stored: &mut [u8],
        root_decoded: &mut [u8],
    ) -> Result<Self, FileVolumeError> {
        let lease = VolumeReadLease::capture(path, File::open(path)?)?;
        let envelope = OpenedEnvelope::read(&lease, lease.file_length(), &limits)?;
        envelope.require_identity(expected)?;
        let plan = PageReadPlan::for_root(&envelope.footer, &limits)?;
        let bytes = plan.read_into(&lease, root_stored, root_decoded)?;
        let summary =
            RootSummary::decode(bytes, &envelope.footer, &limits).map_err(PageIoError::from)?;
        summary
            .validate_header(&envelope.header)
            .map_err(PageIoError::from)?;
        // Validate resident root requirements without touching directory pages.
        DirectoryBufferRequirements::new(&envelope.footer, &summary, &limits)?;
        Ok(Self {
            backing: lease.backing().clone(),
            envelope,
            summary,
            limits,
        })
    }

    /// Attach a completed writer output after its sink has been flushed. Read
    /// back just its bounded envelopes/root and compare the exact completion;
    /// this never invokes the V4 whole-file/whole-column reader.
    pub fn from_finished(
        path: &Path,
        finished: &FinishedVolume,
        limits: ReadLimits,
        root_stored: &mut [u8],
        root_decoded: &mut [u8],
    ) -> Result<Self, FileVolumeError> {
        let volume = Self::open(path, finished.identity, limits, root_stored, root_decoded)?;
        if volume.envelope.footer != finished.footer || volume.summary != finished.summary {
            return Err(FileVolumeError::CompletedFileMismatch);
        }
        Ok(volume)
    }

    pub const fn header(&self) -> &Header {
        &self.envelope.header
    }

    pub const fn summary(&self) -> &RootSummary {
        &self.summary
    }

    pub fn directory_buffers(&self) -> Result<DirectoryBufferRequirements, DirectoryReadError> {
        DirectoryBufferRequirements::new(&self.envelope.footer, &self.summary, &self.limits)
    }

    pub(crate) fn backing(&self) -> &Arc<VolumeFile> {
        &self.backing
    }

    /// Open/verify outside transfer locks. Drop releases this FD; the volume
    /// handle and retirement catalog retain only the physical identity owner.
    pub fn lease(&self) -> Result<FileVolumeLease<'_>, std::io::Error> {
        Ok(FileVolumeLease {
            file: self.backing().open_reader()?,
            volume: self,
        })
    }
}

/// One active reader, reusable for bounded directory and payload operations.
pub struct FileVolumeLease<'volume> {
    file: VolumeReadLease,
    volume: &'volume FileBackedVolume,
}
impl<'volume> FileVolumeLease<'volume> {
    pub fn directory<'lease>(
        &'lease self,
        node: &'lease mut [u8],
        stored: &'lease mut [u8],
    ) -> Result<DirectoryLookup<'lease, Self>, DirectoryReadError> {
        DirectoryLookup::new(
            self,
            self.volume.envelope.footer,
            self.volume.summary,
            self.volume.limits,
            node,
            stored,
        )
    }

    /// Validate sizes before the caller reserves payload buffers. The returned
    /// plan borrows this exact lease, so it cannot read through another file.
    pub fn page(
        &self,
        descriptor: PageDescriptor,
    ) -> Result<LeasedPagePlan<'_, 'volume>, PageIoError> {
        Ok(LeasedPagePlan {
            lease: self,
            plan: PageReadPlan::for_page(
                &self.volume.envelope.footer,
                descriptor,
                &self.volume.limits,
            )?,
        })
    }
}
impl ReadAt for FileVolumeLease<'_> {
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> std::io::Result<usize> {
        self.file.read_at(offset, bytes)
    }
}
impl ReadAt for VolumeReadLease {
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> std::io::Result<usize> {
        VolumeReadLease::read_at(self, offset, bytes)
    }
}

pub struct LeasedPagePlan<'lease, 'volume> {
    lease: &'lease FileVolumeLease<'volume>,
    plan: PageReadPlan,
}
impl LeasedPagePlan<'_, '_> {
    pub const fn stored_buffer_len(&self) -> usize {
        self.plan.stored_buffer_len()
    }

    pub const fn decoded_buffer_len(&self) -> usize {
        self.plan.decoded_buffer_len()
    }

    pub fn read_into<'buffer>(
        &self,
        stored: &'buffer mut [u8],
        decoded: &'buffer mut [u8],
    ) -> Result<&'buffer [u8], PageIoError> {
        self.plan.read_into(self.lease, stored, decoded)
    }
}

#[cfg(test)]
mod tests;
