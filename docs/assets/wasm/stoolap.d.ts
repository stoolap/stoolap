/* tslint:disable */
/* eslint-disable */

/**
 * A Stoolap database instance for use from JavaScript.
 *
 * Create with `new StoolapDB()`, then call `execute(sql)` to run queries.
 * Returns JSON strings for all results. Supports transactions via
 * `begin()`, `commit()`, and `rollback()`.
 */
export class StoolapDB {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Execute a SQL statement and return the result as a JSON string.
     *
     * Returns JSON with one of these shapes:
     * - `{ "type": "rows", "columns": [...], "rows": [[...], ...], "count": N }`
     * - `{ "type": "affected", "affected": N }`
     * - `{ "type": "error", "message": "..." }`
     *
     * Transaction commands (BEGIN, COMMIT, ROLLBACK) are handled automatically.
     */
    execute(sql: string): string;
    /**
     * Execute multiple semicolon-separated SQL statements.
     * Handles quotes and comments correctly (like the CLI).
     * Returns the result of the last statement.
     */
    execute_batch(sql: string): string;
    /**
     * Create a new in-memory Stoolap database.
     */
    constructor();
    /**
     * Return the Stoolap version string (e.g. "0.3.0-abc1234").
     */
    version(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_stoolapdb_free: (a: number, b: number) => void;
    readonly stoolapdb_execute: (a: number, b: number, c: number) => [number, number];
    readonly stoolapdb_execute_batch: (a: number, b: number, c: number) => [number, number];
    readonly stoolapdb_new: () => [number, number, number];
    readonly stoolapdb_version: (a: number) => [number, number];
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
