/* Copyright 2025 Stoolap Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

package mvcc

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stoolap/stoolap/internal/common"
	"github.com/stoolap/stoolap/internal/fastmap"
	"github.com/stoolap/stoolap/internal/storage"
	"github.com/stoolap/stoolap/internal/storage/binser"
)

// SyncMode represents the WAL sync strategy
type SyncMode int

const (
	// SyncNone doesn't force syncs - fastest but least durable
	SyncNone SyncMode = iota
	// SyncNormal syncs on transaction commits
	SyncNormal
	// SyncFull forces syncs on every WAL write - slowest but most durable
	SyncFull
)

// WALOperationType defines the type of operation in a WAL entry
type WALOperationType byte

const (
	WALInsert WALOperationType = iota + 1
	WALUpdate
	WALDelete
	WALCommit
	WALRollback
	WALCreateTable // DDL operation type for table creation
	WALDropTable   // DDL operation type for table deletion
	WALAlterTable  // DDL operation type for table alteration
	WALCreateIndex // DDL operation type for index creation
	WALDropIndex   // DDL operation type for index deletion
)

const (
	// WAL related defaults
	DefaultWALMaxSize      = 64 * 1024 * 1024 // 64MB
	DefaultWALFlushTrigger = 32 * 1024        // 32KB
	DefaultWALBufferSize   = 64 * 1024        // 64KB
)

// CheckpointMetadata stores information about a checkpoint
type CheckpointMetadata struct {
	WALFile            string    // Current WAL file name
	LSN                uint64    // Last sequence number included in this checkpoint
	Timestamp          time.Time // When this checkpoint was created
	IsConsistent       bool      // Whether the checkpoint represents a consistent state
	ActiveTransactions []int64   // List of transaction IDs active at checkpoint time (for validation)
}

// WALEntry represents a single operation in the WAL
type WALEntry struct {
	LSN       uint64           // Log Sequence Number
	TxnID     int64            // Transaction ID
	TableName string           // Table name (empty for commits/rollbacks)
	RowID     int64            // Row ID (0 for commits/rollbacks)
	Operation WALOperationType // Operation type
	Data      []byte           // Serialized row data (nil for commits/rollbacks)
	Timestamp int64            // Operation timestamp (nanoseconds since epoch)
}

// WALManager handles the write-ahead log
type WALManager struct {
	path           string
	walFile        *os.File
	currentWALFile string // Current WAL file name
	mu             sync.Mutex
	currentLSN     atomic.Uint64 // Log Sequence Number
	walBuffer      *common.ByteBuffer
	flushTrigger   uint64 // Bytes before auto-flush
	maxWALSize     uint64 // Maximum WAL file size before rotation
	lastCheckpoint uint64 // Last checkpoint LSN
	syncMode       SyncMode
	running        atomic.Bool
	config         *storage.PersistenceConfig // Reference to persistence config

	// Optimization for SyncNormal mode
	pendingCommits   atomic.Int32 // Number of commits since last sync
	lastSyncTimeNano atomic.Int64 // Last sync time in nanoseconds
	commitBatchSize  int32        // Number of commits to batch before syncing (for SyncNormal)
	syncIntervalNano int64        // Minimum time between syncs in nanoseconds (for SyncNormal)

	// Track active transactions at the time of the last checkpoint
	// Using SegmentInt64Map for thread-safe, lock-free access
	activeTransactions *fastmap.SegmentInt64Map[bool]
}

// NewWALManager creates a new WAL manager
func NewWALManager(path string, syncMode SyncMode, config *storage.PersistenceConfig) (*WALManager, error) {
	// Create WAL directory if it doesn't exist
	err := os.MkdirAll(path, 0755)
	if err != nil {
		return nil, fmt.Errorf("failed to create WAL directory: %w", err)
	}

	var walFile *os.File
	var initialLSN uint64 = 0
	var existingWAL bool = false
	var walFilename string
	var walPath string

	// Check if active WAL exists from checkpoint
	checkpointFile := filepath.Join(path, "checkpoint.meta")
	checkpointMeta, err := ReadCheckpointMeta(checkpointFile)

	// Use checkpoint info if available
	if err == nil && checkpointMeta.WALFile != "" {
		walFilename = checkpointMeta.WALFile
		walPath = filepath.Join(path, checkpointMeta.WALFile)
		initialLSN = checkpointMeta.LSN
		existingWAL = true
	} else {
		// If no checkpoint, look for existing WAL files instead of creating a new one
		files, err := os.ReadDir(path)
		if err == nil {
			// Find all WAL files
			var walFiles []string
			for _, file := range files {
				if !file.IsDir() && strings.HasPrefix(file.Name(), "wal-") && strings.HasSuffix(file.Name(), ".log") {
					walFiles = append(walFiles, file.Name())
				}
			}

			// Sort them by timestamp and LSN
			sort.Strings(walFiles)

			// Use the newest one if available
			if len(walFiles) > 0 {
				walFilename = walFiles[len(walFiles)-1]
				walPath = filepath.Join(path, walFilename)
				existingWAL = true

				// Try to extract LSN from filename if it follows our naming convention
				lsnStrMatch := regexp.MustCompile(`lsn-(\d+)\.log$`).FindStringSubmatch(walFilename)
				if len(lsnStrMatch) == 2 {
					if parsedLSN, parseErr := strconv.ParseUint(lsnStrMatch[1], 10, 64); parseErr == nil {
						initialLSN = parsedLSN
					}
				}
			} else {
				// No WAL files found, create a new one with LSN 0
				timestamp := time.Now().Format("20060102-150405")
				walFilename = fmt.Sprintf("wal-%s-lsn-0.log", timestamp)
				walPath = filepath.Join(path, walFilename)
			}
		} else {
			// Error reading directory, create a new WAL file
			timestamp := time.Now().Format("20060102-150405")
			walFilename = fmt.Sprintf("wal-%s-lsn-0.log", timestamp)
			walPath = filepath.Join(path, walFilename)
		}
	}

	// Open existing WAL file or create a new one
	if existingWAL {
		// Open existing WAL file
		walFile, err = os.OpenFile(walPath, os.O_RDWR|os.O_APPEND, 0644)
		if err != nil {
			// Fallback to creating a new file if we can't open the existing one
			timestamp := time.Now().Format("20060102-150405")
			walFilename = fmt.Sprintf("wal-%s-lsn-0.log", timestamp)
			walPath = filepath.Join(path, walFilename)
			walFile, err = os.OpenFile(walPath, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
			if err != nil {
				return nil, fmt.Errorf("failed to open WAL file: %w", err)
			}
		}
	} else {
		// Create new WAL file
		walFile, err = os.OpenFile(walPath, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0644)
		if err != nil {
			return nil, fmt.Errorf("failed to open WAL file: %w", err)
		}
	}

	// Get file size to determine last LSN
	info, err := walFile.Stat()
	if err != nil {
		walFile.Close()
		return nil, fmt.Errorf("failed to stat WAL file: %w", err)
	}

	// Start with last LSN + 1 if file exists
	if info.Size() > 0 && initialLSN == 0 {
		// Scan through file to find last LSN
		lastLSN, err := findLastLSN(walFile)
		if err != nil {
			walFile.Close()
			return nil, fmt.Errorf("failed to find last LSN: %w", err)
		}
		initialLSN = lastLSN
	}

	// Default values for sync optimization
	commitBatchSize := int32(100)                    // Default batch size of 100 commits
	syncIntervalNano := int64(10 * time.Millisecond) // 10ms default sync interval

	// If config has custom values, use those instead
	if config != nil {
		if config.CommitBatchSize > 0 {
			commitBatchSize = int32(config.CommitBatchSize)
		}
		if config.SyncIntervalMs > 0 {
			syncIntervalNano = int64(config.SyncIntervalMs * int(time.Millisecond))
		}
	}

	wm := &WALManager{
		path:               path,
		walFile:            walFile,
		currentWALFile:     walFilename,
		walBuffer:          common.GetBufferPool(), // Get a buffer from the pool
		flushTrigger:       DefaultWALFlushTrigger, // Flush trigger
		maxWALSize:         DefaultWALMaxSize,      // Default max WAL size
		lastCheckpoint:     initialLSN,             // Start with current LSN as checkpoint
		syncMode:           syncMode,
		config:             config,                                     // Reference to persistence config
		activeTransactions: fastmap.NewSegmentInt64Map[bool](8, 10000), // 2^16 = 65536 buckets for concurrent access

		// Optimization for SyncNormal mode
		commitBatchSize:  commitBatchSize,
		syncIntervalNano: syncIntervalNano,
	}

	// Initialize time to current time
	wm.lastSyncTimeNano.Store(time.Now().UnixNano())

	// Set initial LSN
	wm.currentLSN.Store(initialLSN)
	wm.running.Store(true)

	return wm, nil
}

// ReadCheckpointMeta reads the checkpoint metadata file
// Returns the checkpoint metadata
func ReadCheckpointMeta(path string) (CheckpointMetadata, error) {
	meta := CheckpointMetadata{}

	// Check if file exists
	_, err := os.Stat(path)
	if err != nil {
		return meta, err
	}

	// Read checkpoint file
	data, err := os.ReadFile(path)
	if err != nil {
		return meta, err
	}

	// Parse checkpoint info
	if len(data) < 16 {
		return meta, fmt.Errorf("invalid checkpoint file format")
	}

	reader := binser.NewReader(data)

	// Verify magic number
	magic, err := reader.ReadUint32()
	if err != nil || magic != 0x43504F49 { // "CHKP" in hex
		return meta, fmt.Errorf("invalid checkpoint file format")
	}

	// Read WAL filename
	meta.WALFile, err = reader.ReadString()
	if err != nil {
		return meta, err
	}

	// Read last LSN
	meta.LSN, err = reader.ReadUint64()
	if err != nil {
		return meta, err
	}

	// Read timestamp
	timestamp, err := reader.ReadInt64()
	if err != nil {
		return meta, err
	}
	meta.Timestamp = time.Unix(0, timestamp)

	// Read consistency flag (added for improved checkpointing)
	meta.IsConsistent, err = reader.ReadBool()
	if err != nil {
		// For backward compatibility with old checkpoint format
		meta.IsConsistent = false
		return meta, nil
	}

	// Read active transactions count
	activeCount, err := reader.ReadUint32()
	if err != nil {
		// For backward compatibility
		return meta, nil
	}

	// Read active transaction IDs
	meta.ActiveTransactions = make([]int64, activeCount)
	for i := uint32(0); i < activeCount; i++ {
		txnID, err := reader.ReadInt64()
		if err != nil {
			return meta, err
		}
		meta.ActiveTransactions[i] = txnID
	}

	return meta, nil
}

// findLastLSN scans the WAL file to find the last LSN
func findLastLSN(file *os.File) (uint64, error) {
	// Reset file position
	_, err := file.Seek(0, io.SeekStart)
	if err != nil {
		return 0, err
	}

	var lastLSN uint64 = 0

	for {
		// Try to read entry header (LSN + size)
		headerBuf := make([]byte, 16) // 8 bytes for LSN, 8 bytes for size
		n, err := file.Read(headerBuf)
		if err != nil {
			if err == io.EOF {
				break
			}
			return 0, err
		}
		if n < 16 {
			// Incomplete header, stop
			break
		}

		// Extract LSN and entry size
		lsn := binary.LittleEndian.Uint64(headerBuf[:8])
		size := binary.LittleEndian.Uint64(headerBuf[8:16])

		// Update last LSN
		if lsn > lastLSN {
			lastLSN = lsn
		}

		// Skip to next entry
		_, err = file.Seek(int64(size), io.SeekCurrent)
		if err != nil {
			return 0, err
		}
	}

	return lastLSN, nil
}

// BatchCommit performs a batch commit of multiple WAL entries
func (wm *WALManager) BatchCommit(entries []WALEntry) error {
	// Skip if not running or empty batch
	if !wm.running.Load() || len(entries) == 0 {
		return fmt.Errorf("WAL manager is not running or empty batch")
	}

	// Acquire the lock for the batch operation
	wm.mu.Lock()
	defer wm.mu.Unlock()

	// Check again after acquiring lock
	if !wm.running.Load() || wm.walFile == nil {
		return fmt.Errorf("WAL manager is not running or file is closed")
	}

	// Track highest LSN for updating metadata
	var highestLSN uint64

	// Prepare buffer for all entries
	// This is more efficient than calling AppendEntryLocked for each entry
	batchBuffer := common.GetBufferPool() // Get a ByteBuffer from the pool
	defer common.PutBufferPool(batchBuffer)

	// Serialize all entries into the buffer
	for _, entry := range entries {
		// Encode the entry directly into the batch buffer
		if err := encodeEntry(batchBuffer, entry); err != nil {
			return fmt.Errorf("failed to serialize WAL entry: %w", err)
		}

		// Track highest LSN
		if entry.LSN > highestLSN {
			highestLSN = entry.LSN
		}
	}

	// Write the entire batch at once
	_, err := wm.walFile.Write(batchBuffer.Bytes())
	if err != nil {
		return fmt.Errorf("failed to write batch to WAL: %w", err)
	}

	// Force sync for batch
	err = OptimizedSync(wm.walFile)
	if err != nil {
		return fmt.Errorf("failed to sync WAL after batch: %w", err)
	}

	// Update LSN
	wm.currentLSN.Store(highestLSN)

	// Update metadata
	if highestLSN > 0 {
		// We might not have access to the persistence manager here
		// so we don't update lastWALLSN directly
	}

	return nil
}

// Close safely closes the WAL manager with proper resource cleanup
func (wm *WALManager) Close() error {
	// Flag as not running to prevent new operations immediately
	if !wm.running.CompareAndSwap(true, false) {
		// Already closed
		return nil
	}

	var closeErr error

	// Flush any pending data in buffer
	if wm.walBuffer != nil {
		bufferData := wm.walBuffer.Bytes()
		if len(bufferData) > 0 && wm.walFile != nil {
			// Write directly to file
			if _, err := wm.walFile.Write(bufferData); err != nil {
				fmt.Printf("Error: during final WAL flush: %v\n", err)
			} else {
				OptimizedSync(wm.walFile)
			}
		}

		// Release buffer resources
		common.PutBufferPool(wm.walBuffer)
		wm.walBuffer = nil
	}

	// Acquire lock to safely close file
	wm.mu.Lock()
	defer wm.mu.Unlock()

	// Close file if open
	if wm.walFile != nil {
		// Check if WAL file is empty before closing
		var filePath string
		var fileIsEmpty bool

		// Get file info for potential cleanup
		if fileInfo, err := wm.walFile.Stat(); err == nil {
			fileIsEmpty = fileInfo.Size() == 0
			filePath = filepath.Join(wm.path, wm.currentWALFile)
		}

		// Close file
		if err := wm.walFile.Close(); err != nil {
			closeErr = fmt.Errorf("error closing WAL file: %w", err)
			fmt.Printf("Error: closing WAL file: %v\n", err)
		}

		// Clear reference regardless of outcome
		wm.walFile = nil

		// Delete empty WAL file if it exists
		if fileIsEmpty && filePath != "" {
			if err := os.Remove(filePath); err != nil {
				fmt.Printf("Warning: Failed to remove empty WAL file: %v\n", err)
			}
		}
	}

	return closeErr
}

// SyncLocked forces a sync to disk - assumes caller has already acquired the lock
func (wm *WALManager) SyncLocked() error {
	// Check if running
	if !wm.running.Load() {
		return fmt.Errorf("WAL manager is not running")
	}

	// Check if file is valid
	if wm.walFile == nil {
		return nil
	}

	// Track time for diagnostics
	startTime := time.Now()

	// Direct sync call
	err := OptimizedSync(wm.walFile)

	// Log slow syncs
	duration := time.Since(startTime)
	if duration > 100*time.Millisecond {
		fmt.Printf("Warning: Slow disk sync: %v\n", duration)
	}

	return err
}

// Sync forces a sync to disk - public version that acquires lock
func (wm *WALManager) Sync() error {
	// Check if running before taking lock
	if !wm.running.Load() {
		return fmt.Errorf("WAL manager is not running")
	}

	// Use a non-blocking attempt first to avoid potential deadlocks
	if !wm.mu.TryLock() {
		// If we can't get the lock immediately, log and use regular Lock
		fmt.Println("Warning: Sync lock contention, waiting for lock")
		wm.mu.Lock()
	}

	// We got the lock either way
	defer wm.mu.Unlock()

	return wm.SyncLocked()
}

// Flush flushes the WAL buffer to disk without syncing
// Callers must call Sync separately if needed based on their operation type
func (wm *WALManager) Flush() error {
	// Check if running to avoid unnecessary lock acquisition
	if !wm.running.Load() {
		return fmt.Errorf("WAL manager is not running")
	}

	// Acquire lock for buffer access
	wm.mu.Lock()
	defer wm.mu.Unlock()

	// Check if buffer is available after acquiring lock
	if wm.walBuffer == nil {
		return fmt.Errorf("WAL buffer has been released")
	}

	// Get a copy of the buffer data
	data := wm.walBuffer.Bytes()

	// Reset buffer only if we have data (avoid unnecessary operations)
	if len(data) > 0 {
		wm.walBuffer.Reset()
	} else {
		// Nothing to flush
		return nil
	}

	// Write to disk (we already have the lock)
	// writeToFile should only write bytes to disk, not sync
	err := wm.writeToFile(data)
	if err != nil {
		return fmt.Errorf("failed to flush WAL buffer: %w", err)
	}

	return nil
}

// AppendEntry adds a new operation to the WAL - acquires its own lock
func (wm *WALManager) AppendEntry(entry WALEntry) (uint64, error) {
	// Check if running before taking the lock
	if !wm.running.Load() {
		return 0, fmt.Errorf("WAL manager is not running")
	}

	// Get the lock
	wm.mu.Lock()
	defer wm.mu.Unlock()

	// Call the locked version with lock already held
	return wm.AppendEntryLocked(entry)
}

// AppendEntryLocked adds a new operation to the WAL - assumes caller has lock
func (wm *WALManager) AppendEntryLocked(entry WALEntry) (uint64, error) {
	// Skip if not running - caller should already have checked
	if !wm.running.Load() {
		return 0, fmt.Errorf("WAL manager is not running")
	}

	// Check for nil file in case we're shutting down
	if wm.walFile == nil {
		return 0, fmt.Errorf("WAL file has been closed")
	}

	// Assign LSN atomically
	entry.LSN = wm.currentLSN.Add(1)

	buf := common.GetBufferPool()
	defer common.PutBufferPool(buf)
	// Serialize entry directly without goroutine
	if err := encodeEntry(buf, entry); err != nil {
		return 0, fmt.Errorf("failed to serialize WAL entry: %w", err)
	}

	// For SyncNormal mode with commit/rollback operations, track for batched syncing
	if wm.syncMode == SyncNormal && (entry.Operation == WALCommit || entry.Operation == WALRollback) {
		wm.pendingCommits.Add(1)
	}

	// For small entries, we batch them
	if buf.Len() < int(wm.flushTrigger) {
		// Check for buffer
		if wm.walBuffer == nil {
			return 0, fmt.Errorf("WAL buffer has been released")
		}

		// Write to the buffer with explicit byte counting for better diagnostic
		beforeLen := wm.walBuffer.Len()
		wm.walBuffer.Write(buf.Bytes())
		afterLen := wm.walBuffer.Len()

		// Ensure all bytes were written
		if afterLen-beforeLen != buf.Len() {
			return 0, fmt.Errorf("failed to write all bytes to WAL buffer (%d of %d written)",
				afterLen-beforeLen, buf.Len())
		}

		needsFlush := wm.walBuffer.Len() >= int(wm.flushTrigger)

		// For commit/rollback in SyncNormal or SyncFull, or any operation in SyncFull,
		// we need to force a flush regardless of buffer size
		forceFlush := (wm.syncMode == SyncFull) ||
			(wm.syncMode == SyncNormal && (entry.Operation == WALCommit ||
				entry.Operation == WALRollback ||
				entry.Operation == WALCreateTable ||
				entry.Operation == WALDropTable ||
				entry.Operation == WALAlterTable ||
				entry.Operation == WALCreateIndex ||
				entry.Operation == WALDropIndex))

		// Flush if needed or forced
		if needsFlush || forceFlush {
			// Get the buffer data
			bufferData := wm.walBuffer.Bytes()

			// Only proceed if we actually have data to flush
			if len(bufferData) > 0 {
				// Write directly to file
				err := wm.writeToFile(bufferData)
				if err != nil {
					return 0, fmt.Errorf("failed to write WAL buffer to file: %w", err)
				}

				// Reset buffer
				wm.walBuffer.Reset()

				// Sync if needed based on operation type
				if wm.shouldSync(entry.Operation) {
					// Direct call to SyncLocked with timing diagnostic
					syncStart := time.Now()
					err := wm.SyncLocked()

					// Log slow syncs
					syncTime := time.Since(syncStart)
					if syncTime > 100*time.Millisecond {
						fmt.Printf("Warning: WAL sync took %v during buffer flush\n", syncTime)
					}

					if err != nil {
						return 0, fmt.Errorf("failed to sync WAL after write: %w", err)
					}
				}
			}
		}

		return entry.LSN, nil
	}

	// Large entries written directly
	if err := wm.writeToFile(buf.Bytes()); err != nil {
		return 0, fmt.Errorf("failed to write large WAL entry to file: %w", err)
	}

	// Sync if needed based on operation type
	if wm.shouldSync(entry.Operation) {
		// Direct call to SyncLocked with timing diagnostic
		syncStart := time.Now()
		err := wm.SyncLocked()

		// Log slow syncs
		syncTime := time.Since(syncStart)
		if syncTime > 100*time.Millisecond {
			fmt.Printf("Warning: WAL sync took %v for large entry\n", syncTime)
		}

		if err != nil {
			return 0, fmt.Errorf("failed to sync WAL file: %w", err)
		}
	}

	return entry.LSN, nil
}

// serializeEntry serializes a WAL entry to binary format with improved performance
func (wm *WALManager) serializeEntry(entry WALEntry) ([]byte, error) {
	// Get a buffer from the pool
	buf := common.GetBufferPool()
	defer common.PutBufferPool(buf)

	// Use the optimized encoding function
	if err := encodeEntry(buf, entry); err != nil {
		return nil, err
	}

	// Return a copy of the bytes to avoid buffer reuse issues
	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())
	return result, nil
}

// encodeEntry serializes a WAL entry to binary format with improved performance
func encodeEntry(w *common.ByteBuffer, entry WALEntry) error {
	var (
		buf8 [8]byte
		buf4 [4]byte
		buf2 [2]byte
	)

	// 1) LSN
	binary.LittleEndian.PutUint64(buf8[:], entry.LSN)
	if _, err := w.Write(buf8[:]); err != nil {
		return err
	}

	// 2) totalSize (all remaining bytes after LSN+size)
	//    fixed fields = 8(txnID)+2(nameLen)+len(name)+8(rowID)+1(op)+8(ts)+4(dataLen)+len(data)
	totalSize := uint64(8 + 2 + len(entry.TableName) + 8 + 1 + 8 + 4 + len(entry.Data))
	binary.LittleEndian.PutUint64(buf8[:], totalSize)
	if _, err := w.Write(buf8[:]); err != nil {
		return err
	}

	// 3) TxnID
	binary.LittleEndian.PutUint64(buf8[:], uint64(entry.TxnID))
	if _, err := w.Write(buf8[:]); err != nil {
		return err
	}

	// 4) TableName length + bytes
	binary.LittleEndian.PutUint16(buf2[:], uint16(len(entry.TableName)))
	if _, err := w.Write(buf2[:]); err != nil {
		return err
	}

	if _, err := w.WriteString(entry.TableName); err != nil {
		return err
	}

	// 5) RowID
	binary.LittleEndian.PutUint64(buf8[:], uint64(entry.RowID))
	if _, err := w.Write(buf8[:]); err != nil {
		return err
	}

	// 6) Operation
	if _, err := w.Write([]byte{byte(entry.Operation)}); err != nil {
		return err
	}

	// 7) Timestamp
	binary.LittleEndian.PutUint64(buf8[:], uint64(entry.Timestamp))
	if _, err := w.Write(buf8[:]); err != nil {
		return err
	}

	// 8) Data length + bytes
	binary.LittleEndian.PutUint32(buf4[:], uint32(len(entry.Data)))
	if _, err := w.Write(buf4[:]); err != nil {
		return err
	}

	if len(entry.Data) > 0 {
		if _, err := w.Write(entry.Data); err != nil {
			return err
		}
	}

	return nil
}

// writeToFile writes data to the WAL file - assumes caller has lock
func (wm *WALManager) writeToFile(data []byte) error {
	// Check if file is closed
	if wm.walFile == nil {
		return fmt.Errorf("WAL file is closed")
	}

	// Check if we have data to write
	if len(data) == 0 {
		return nil
	}

	// WAL truncation is now handled directly after checkpoints in the Snapshotter

	// Track time for diagnostic purposes
	startTime := time.Now()

	// Write data to file with a direct call instead of using a goroutine
	n, err := wm.walFile.Write(data)
	if err != nil {
		return fmt.Errorf("failed to write to WAL file: %w", err)
	}

	// Check partial writes
	if n < len(data) {
		return fmt.Errorf("partial write: %d of %d bytes written", n, len(data))
	}

	// Log slow writes for diagnostic purposes
	writeTime := time.Since(startTime)
	if writeTime > 100*time.Millisecond {
		fmt.Printf("Warning: Slow WAL write: %v for %d bytes\n", writeTime, len(data))
	}

	return nil
}

// Replay replays WAL entries from a given LSN
func (wm *WALManager) Replay(fromLSN uint64, callback func(WALEntry) error) (uint64, error) {
	// Flush buffer to make sure all entries are on disk
	err := wm.Flush()
	if err != nil {
		return 0, err
	}

	// Load checkpoint if available
	checkpointFile := filepath.Join(wm.path, "checkpoint.meta")
	checkpoint, checkpointErr := ReadCheckpointMeta(checkpointFile)

	// We'll keep track of all WAL files that need to be replayed
	var walFilesToReplay []string

	// First, check if we need to replay from checkpoint
	if checkpointErr == nil && checkpoint.LSN > fromLSN {
		// We have a valid checkpoint that is newer than fromLSN
		// Replay from checkpoint instead, but keep track of the active WAL file
		fromLSN = checkpoint.LSN

		// Add the active WAL file first
		checkpointWALPath := filepath.Join(wm.path, checkpoint.WALFile)
		if _, err := os.Stat(checkpointWALPath); err == nil {
			walFilesToReplay = append(walFilesToReplay, checkpointWALPath)
		}
	} else {
		// No valid checkpoint, or it's older than fromLSN
		// Collect all WAL files in the directory
		if dirEntries, err := os.ReadDir(wm.path); err == nil {
			for _, entry := range dirEntries {
				if !entry.IsDir() && strings.HasPrefix(entry.Name(), "wal-") && strings.HasSuffix(entry.Name(), ".log") {
					walFilesToReplay = append(walFilesToReplay, filepath.Join(wm.path, entry.Name()))
				}
			}
		}

		// Sort WAL files by name (which includes timestamp)
		sort.Strings(walFilesToReplay)

		// Add current WAL file only if it's not already in the list
		currentWALPath := filepath.Join(wm.path, wm.currentWALFile)

		// Check if current WAL file is already in the list
		currentWALInList := false
		for _, path := range walFilesToReplay {
			if path == currentWALPath {
				currentWALInList = true
				break
			}
		}

		// Only append if not already in the list
		if !currentWALInList {
			walFilesToReplay = append(walFilesToReplay, currentWALPath)
		}
	}

	// If we have no WAL files to replay, we're done
	if len(walFilesToReplay) == 0 {
		return fromLSN, nil
	}

	// Replay each WAL file in sequence
	var lastLSN uint64 = fromLSN

	for _, walPath := range walFilesToReplay {
		// Open the WAL file
		walFile, err := os.Open(walPath)
		if err != nil {
			// Skip missing or inaccessible files
			continue
		}

		// Reuse header buffer to reduce allocations
		headerBuf := make([]byte, 16) // 8 bytes for LSN, 8 bytes for size

		// Get a data buffer from the pool to reduce allocations
		entryBuf := common.GetBufferPool()

		// Reset file position
		_, err = walFile.Seek(0, io.SeekStart)
		if err != nil {
			walFile.Close()
			common.PutBufferPool(entryBuf)
			return lastLSN, err
		}

		for {
			// Try to read entry header (LSN + size)
			n, err := walFile.Read(headerBuf)
			if err != nil {
				if err == io.EOF {
					break
				}
				walFile.Close()
				common.PutBufferPool(entryBuf)
				return lastLSN, err
			}
			if n < 16 {
				// Incomplete header, stop
				break
			}

			// Extract LSN and entry size
			lsn := binary.LittleEndian.Uint64(headerBuf[:8])
			size := binary.LittleEndian.Uint64(headerBuf[8:16])

			// Skip entries before fromLSN
			if lsn < fromLSN {
				_, err = walFile.Seek(int64(size), io.SeekCurrent)
				if err != nil {
					walFile.Close()
					common.PutBufferPool(entryBuf)
					return lastLSN, err
				}
				continue
			}

			// Prepare a temporary buffer for reading the entry data
			// This is more efficient than resizing the ByteBuffer directly
			tempBuf := make([]byte, size)

			// Read entry data
			n, err = walFile.Read(tempBuf)
			if err != nil {
				walFile.Close()
				common.PutBufferPool(entryBuf)
				return lastLSN, err
			}
			if uint64(n) < size {
				// Incomplete entry, stop
				walFile.Close()
				common.PutBufferPool(entryBuf)
				return lastLSN, fmt.Errorf("incomplete WAL entry")
			}

			// Deserialize entry
			entry, err := wm.deserializeEntry(lsn, tempBuf)
			if err != nil {
				walFile.Close()
				common.PutBufferPool(entryBuf)
				return lastLSN, err
			}

			// Call callback
			err = callback(entry)
			if err != nil {
				walFile.Close()
				common.PutBufferPool(entryBuf)
				return lastLSN, err
			}

			// Update last LSN
			lastLSN = lsn

			// Update fromLSN to avoid re-processing in the next file
			fromLSN = lsn
		}

		// Close file and return buffer to pool
		walFile.Close()
		common.PutBufferPool(entryBuf)
	}

	// Update the current LSN if the last replayed LSN is greater
	if lastLSN > wm.currentLSN.Load() {
		wm.currentLSN.Store(lastLSN)
	}

	return lastLSN, nil
}

// Helper method to determine if a sync is needed
func (wm *WALManager) shouldSync(opType WALOperationType) bool {
	// Batched sync logic for commits/rollbacks
	now := time.Now().UnixNano()
	last := wm.lastSyncTimeNano.Load()
	switch wm.syncMode {
	case SyncNone:
		return false
	case SyncNormal:
		// Immediate sync for all DDL operations
		switch opType {
		case WALCreateTable, WALDropTable, WALAlterTable, WALCreateIndex, WALDropIndex:
			return true
		case WALCommit, WALRollback:
			// Determine if we should sync based on:
			// 1. Reached batch size threshold
			// 2. Time interval since last sync exceeded
			if wm.pendingCommits.Load() >= wm.commitBatchSize ||
				now-last >= wm.syncIntervalNano {
				wm.pendingCommits.Store(0)
				wm.lastSyncTimeNano.Store(now)
				return true
			}
		}
		// Everything else in SyncNormal is not synced immediately
		return false
	case SyncFull:
		// In SyncFull mode, we sync ALL operations
		return true
	default:
		return false
	}
}

// deserializeEntry deserializes a WAL entry from binary data
func (wm *WALManager) deserializeEntry(lsn uint64, data []byte) (WALEntry, error) {
	// Minimum data length check
	if len(data) < 16 {
		return WALEntry{}, fmt.Errorf("data too short for WAL entry: %d bytes", len(data))
	}

	entry := WALEntry{
		LSN: lsn,
	}

	// Use a cursor-based approach for more controlled parsing
	pos := 0

	// Read Transaction ID (int64/uint64 - 8 bytes)
	if pos+8 > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading transaction ID at position %d", pos)
	}
	entry.TxnID = int64(binary.LittleEndian.Uint64(data[pos:]))
	pos += 8

	// Read table name with length prefix (uint16 + variable)
	if pos+2 > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading table name length at position %d", pos)
	}
	tableNameLen := binary.LittleEndian.Uint16(data[pos:])
	pos += 2

	if pos+int(tableNameLen) > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading table name of length %d at position %d", tableNameLen, pos)
	}
	entry.TableName = string(data[pos : pos+int(tableNameLen)])
	pos += int(tableNameLen)

	// Read Row ID (int64/uint64 - 8 bytes)
	if pos+8 > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading row ID at position %d", pos)
	}
	entry.RowID = int64(binary.LittleEndian.Uint64(data[pos:]))
	pos += 8

	// Read operation type (byte/uint8 - 1 byte)
	if pos+1 > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading operation type at position %d", pos)
	}
	entry.Operation = WALOperationType(data[pos])
	pos += 1

	// Read timestamp (int64/uint64 - 8 bytes)
	if pos+8 > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading timestamp at position %d", pos)
	}
	entry.Timestamp = int64(binary.LittleEndian.Uint64(data[pos:]))
	pos += 8

	// Read data with length prefix (uint32 + variable)
	if pos+4 > len(data) {
		return entry, fmt.Errorf("unexpected end of data reading data length at position %d", pos)
	}
	dataLen := binary.LittleEndian.Uint32(data[pos:])
	pos += 4

	// Verify data length is reasonable (avoid potential OOM)
	if dataLen > 100*1024*1024 { // 100MB limit for safety
		return entry, fmt.Errorf("data length too large: %d bytes", dataLen)
	}

	if dataLen > 0 {
		if pos+int(dataLen) > len(data) {
			return entry, fmt.Errorf("unexpected end of data reading entry data of length %d at position %d", dataLen, pos)
		}
		entry.Data = make([]byte, dataLen)
		copy(entry.Data, data[pos:pos+int(dataLen)])
	} else {
		entry.Data = nil
	}

	return entry, nil
}

// UpdateActiveTransactions tracks active transactions for checkpoint validation
func (wm *WALManager) UpdateActiveTransactions(txnID int64, isActive bool) {
	if isActive {
		wm.activeTransactions.Set(txnID, true)
	} else {
		wm.activeTransactions.Del(txnID)
	}
}

// GetActiveTransactionIDs returns a list of active transaction IDs
func (wm *WALManager) GetActiveTransactionIDs() []int64 {
	count := wm.activeTransactions.Len()
	txnIDs := make([]int64, 0, count)

	wm.activeTransactions.ForEach(func(txnID int64, _ bool) bool {
		txnIDs = append(txnIDs, txnID)
		return true
	})

	return txnIDs
}

// createConsistentCheckpoint creates a checkpoint that represents a transactionally consistent state
func (wm *WALManager) createConsistentCheckpoint(lsn uint64, isConsistent bool) {
	// Skip if not running
	if !wm.running.Load() {
		return
	}

	// First flush any pending writes
	err := wm.Flush()
	if err != nil {
		fmt.Printf("Error: flushing WAL before checkpoint: %v\n", err)
		return
	}

	wm.mu.Lock()
	// Check if we're still running after acquiring lock
	if !wm.running.Load() || wm.walFile == nil {
		wm.mu.Unlock()
		return
	}

	// Sync current WAL file
	err = OptimizedSync(wm.walFile)
	if err != nil {
		fmt.Printf("Error: syncing WAL before checkpoint: %v\n", err)
		wm.mu.Unlock()
		return
	}

	// Get current state to write to checkpoint
	currentWAL := wm.currentWALFile
	currentLSN := lsn
	wm.lastCheckpoint = currentLSN

	// Get active transactions for validation during recovery
	activeTransactions := wm.GetActiveTransactionIDs()

	// Only mark as consistent if explicitly requested and no active transactions
	isReallyConsistent := isConsistent && len(activeTransactions) == 0

	wm.mu.Unlock()

	// Write checkpoint metadata
	err = wm.writeEnhancedCheckpointMeta(currentWAL, currentLSN, isReallyConsistent, activeTransactions)
	if err != nil {
		fmt.Printf("Error: writing checkpoint metadata: %v\n", err)
		return
	}
}

// writeEnhancedCheckpointMeta writes improved checkpoint metadata to disk
func (wm *WALManager) writeEnhancedCheckpointMeta(walFile string, lsn uint64, isConsistent bool, activeTransactions []int64) error {
	// If we're going to truncate the WAL file after this checkpoint,
	// the WAL filename will change to include the LSN
	// So we need to anticipate what the new name will be
	timestamp := time.Now().Format("20060102-150405")
	newWalFilename := fmt.Sprintf("wal-%s-lsn-%d.log", timestamp, lsn)

	// Create checkpoint file
	checkpointPath := filepath.Join(wm.path, "checkpoint.meta")
	file, err := os.Create(checkpointPath)
	if err != nil {
		return fmt.Errorf("failed to create checkpoint file: %w", err)
	}
	defer file.Close()

	// Create writer
	writer := binser.NewWriter()
	defer writer.Release()

	// Write magic number
	writer.WriteUint32(0x43504F49) // "CHKP" in hex

	// Write WAL file - use the anticipated new name if this is a consistent checkpoint
	// that will lead to truncation
	if isConsistent {
		writer.WriteString(newWalFilename)
	} else {
		writer.WriteString(walFile)
	}

	// Write LSN
	writer.WriteUint64(lsn)

	// Write timestamp
	writer.WriteInt64(time.Now().UnixNano())

	// Write consistency flag
	writer.WriteBool(isConsistent)

	// Write active transactions count
	writer.WriteUint32(uint32(len(activeTransactions)))

	// Write active transaction IDs
	for _, txnID := range activeTransactions {
		writer.WriteInt64(txnID)
	}

	// Write to file
	_, err = file.Write(writer.Bytes())
	if err != nil {
		return fmt.Errorf("failed to write checkpoint data: %w", err)
	}

	// Ensure data is on disk
	return OptimizedSync(file)
}

// TruncateWAL truncates the WAL file to remove entries up to the given LSN
// This is used after a successful checkpoint to reclaim disk space
func (wm *WALManager) TruncateWAL(upToLSN uint64) error {
	// Skip if not running or if upToLSN is zero (no valid checkpoint)
	if !wm.running.Load() {
		return fmt.Errorf("WAL manager is not running")
	}

	if upToLSN == 0 {
		return fmt.Errorf("invalid LSN for WAL truncation: %d", upToLSN)
	}

	// Get a lock to ensure no concurrent operations during truncation
	wm.mu.Lock()
	defer wm.mu.Unlock()

	// Verify we're still running
	if !wm.running.Load() || wm.walFile == nil {
		return fmt.Errorf("WAL manager is not running or file is closed")
	}

	// Extract LSN from current WAL filename
	lsnStrMatch := regexp.MustCompile(`lsn-(\d+)\.log$`).FindStringSubmatch(wm.currentWALFile)
	if len(lsnStrMatch) == 2 {
		if currentFileLSN, parseErr := strconv.ParseUint(lsnStrMatch[1], 10, 64); parseErr == nil {
			// If upToLSN <= currentFileLSN, there's nothing to truncate as all entries
			// in this file are already newer than upToLSN
			if upToLSN <= currentFileLSN {
				return nil
			}
		}
	}

	// First, flush any pending data to make sure everything is on disk
	bufferData := wm.walBuffer.Bytes()
	if len(bufferData) > 0 {
		// Write directly to file
		if _, err := wm.walFile.Write(bufferData); err != nil {
			return fmt.Errorf("failed to flush buffer during truncation: %w", err)
		}
		wm.walBuffer.Reset()
	}

	// Sync file to ensure all data is persisted
	if err := OptimizedSync(wm.walFile); err != nil {
		return fmt.Errorf("failed to sync WAL during truncation: %w", err)
	}

	// Create a new file for the truncated WAL with LSN-based naming
	timestamp := time.Now().Format("20060102-150405")
	newWalFilename := fmt.Sprintf("wal-%s-lsn-%d.log", timestamp, upToLSN)
	tempWalPath := filepath.Join(wm.path, fmt.Sprintf("wal-temp-%d.log", time.Now().UnixNano()))
	tempWalFile, err := os.Create(tempWalPath)
	if err != nil {
		return fmt.Errorf("failed to create temporary WAL file: %w", err)
	}

	// Make sure the temp file is closed if there's an error
	tempFileClosed := false
	defer func() {
		if !tempFileClosed {
			tempWalFile.Close()
			os.Remove(tempWalPath) // Clean up the temp file on error
		}
	}()

	// Reset the current WAL file position to beginning
	_, err = wm.walFile.Seek(0, io.SeekStart)
	if err != nil {
		return fmt.Errorf("failed to seek WAL file: %w", err)
	}

	// Copy entries that are newer than upToLSN to the temp file
	// Reuse header buffer to reduce allocations
	headerBuf := make([]byte, 16) // 8 bytes for LSN, 8 bytes for size
	entriesCopied := 0
	bytesSkipped := int64(0)
	bytesCopied := int64(0)

	for {
		// Try to read entry header (LSN + size)
		n, err := wm.walFile.Read(headerBuf)
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("error reading WAL header: %w", err)
		}
		if n < 16 {
			// Incomplete header, stop
			break
		}

		// Extract LSN and entry size
		lsn := binary.LittleEndian.Uint64(headerBuf[:8])
		size := binary.LittleEndian.Uint64(headerBuf[8:16])

		// Calculate total entry size (header + data)
		totalEntrySize := int64(16 + size)

		// If the entry's LSN is older than or equal to upToLSN, skip it
		if lsn <= upToLSN {
			// Skip to next entry
			_, err = wm.walFile.Seek(int64(size), io.SeekCurrent)
			if err != nil {
				return fmt.Errorf("failed to seek past old entry: %w", err)
			}
			bytesSkipped += totalEntrySize
		} else {
			// Write the header to the temp file
			_, err = tempWalFile.Write(headerBuf)
			if err != nil {
				return fmt.Errorf("failed to write header to temp file: %w", err)
			}

			// Copy the entry data
			_, err = io.CopyN(tempWalFile, wm.walFile, int64(size))
			if err != nil {
				return fmt.Errorf("failed to copy entry data: %w", err)
			}

			entriesCopied++
			bytesCopied += totalEntrySize
		}
	}

	// If we didn't copy any entries (all entries were old), create a marker entry
	// so the WAL file isn't empty
	if entriesCopied == 0 {
		markerEntry := WALEntry{
			LSN:       upToLSN + 1,
			TxnID:     -1000, // Special marker transaction
			Operation: WALCommit,
			Timestamp: time.Now().UnixNano(),
		}

		batchBuffer := common.GetBufferPool()
		defer common.PutBufferPool(batchBuffer)
		if err := encodeEntry(batchBuffer, markerEntry); err != nil {
			return fmt.Errorf("failed to create marker entry: %w", err)
		}

		if _, err = tempWalFile.Write(batchBuffer.Bytes()); err != nil {
			return fmt.Errorf("failed to write marker entry: %w", err)
		}
	}

	// Sync the temp file to ensure data is flushed to disk
	if err := OptimizedSync(tempWalFile); err != nil {
		return fmt.Errorf("failed to sync temp WAL file: %w", err)
	}

	// Close the current WAL file
	walFilePath := filepath.Join(wm.path, wm.currentWALFile)
	if err := wm.walFile.Close(); err != nil {
		// Try to recover by reopening the original file
		wm.walFile, _ = os.OpenFile(walFilePath, os.O_RDWR|os.O_APPEND, 0644)
		return fmt.Errorf("failed to close WAL file: %w", err)
	}

	// Close the temp file
	if err := tempWalFile.Close(); err != nil {
		// Try to recover by reopening the original file
		wm.walFile, _ = os.OpenFile(walFilePath, os.O_RDWR|os.O_APPEND, 0644)
		return fmt.Errorf("failed to close temp WAL file: %w", err)
	}
	tempFileClosed = true

	// Create the final path with the new LSN-based name
	newWalPath := filepath.Join(wm.path, newWalFilename)

	// First rename the temp file to the new LSN-based file
	if err := os.Rename(tempWalPath, newWalPath); err != nil {
		// Try to recover by reopening the original file
		wm.walFile, _ = os.OpenFile(walFilePath, os.O_RDWR|os.O_APPEND, 0644)
		return fmt.Errorf("failed to rename temp file to new WAL file: %w", err)
	}

	// Remove the old WAL file if it's different from the new one
	if walFilePath != newWalPath {
		// Try to remove the old file (this might fail on Windows if file is in use)
		if err := os.Remove(walFilePath); err != nil {
			fmt.Printf("Warning: Could not remove old WAL file %s: %v\n", walFilePath, err)
		}
	}

	// Update the current WAL file information
	wm.currentWALFile = newWalFilename

	// Open the new WAL file
	wm.walFile, err = os.OpenFile(newWalPath, os.O_RDWR|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to reopen WAL file after truncation: %w", err)
	}

	return nil
}

// ValidateCheckpoint verifies the consistency of a checkpoint against the current database state
func (wm *WALManager) ValidateCheckpoint(checkpoint CheckpointMetadata) (bool, error) {
	// Check if any active transactions from checkpoint are still active
	for _, txnID := range checkpoint.ActiveTransactions {
		if wm.IsTransactionActive(txnID) {
			return false, fmt.Errorf("transaction %d from checkpoint is still active", txnID)
		}
	}

	// Validate WAL file exists
	walPath := filepath.Join(wm.path, checkpoint.WALFile)
	if _, err := os.Stat(walPath); err != nil {
		if os.IsNotExist(err) {
			return false, fmt.Errorf("WAL file %s referenced in checkpoint does not exist", checkpoint.WALFile)
		}
		return false, fmt.Errorf("error checking WAL file: %w", err)
	}

	return true, nil
}

// IsTransactionActive checks if a transaction is still active
func (wm *WALManager) IsTransactionActive(txnID int64) bool {
	_, exists := wm.activeTransactions.Get(txnID)
	return exists
}
