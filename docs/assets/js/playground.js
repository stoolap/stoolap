/**
 * Stoolap Playground
 *
 * Terminal-style interface that loads Stoolap via WebAssembly.
 * Mirrors the CLI experience: command history, multi-line input,
 * ASCII table output, transaction support.
 */

const SAMPLE_SCHEMA = `
CREATE TABLE users (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE,
  age INTEGER
);

INSERT INTO users VALUES
  (1, 'Alice', 'alice@example.com', 32),
  (2, 'Bob', 'bob@example.com', 28),
  (3, 'Charlie', 'charlie@example.com', 45),
  (4, 'Diana', 'diana@example.com', 36),
  (5, 'Eve', 'eve@example.com', 24);

CREATE TABLE orders (
  id INTEGER PRIMARY KEY,
  user_id INTEGER,
  product TEXT NOT NULL,
  amount FLOAT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO orders VALUES
  (1, 1, 'Laptop', 999.99, '2024-01-15 10:30:00'),
  (2, 1, 'Mouse', 29.99, '2024-01-16 14:20:00'),
  (3, 2, 'Keyboard', 79.99, '2024-02-01 09:15:00'),
  (4, 3, 'Monitor', 449.99, '2024-02-10 16:45:00'),
  (5, 3, 'Webcam', 59.99, '2024-02-10 16:50:00'),
  (6, 4, 'Headphones', 149.99, '2024-03-05 11:00:00'),
  (7, 2, 'USB Hub', 39.99, '2024-03-12 08:30:00'),
  (8, 5, 'Tablet', 599.99, '2024-03-20 13:15:00'),
  (9, 1, 'Charger', 24.99, '2024-04-01 10:00:00'),
  (10, 4, 'Stand', 89.99, '2024-04-15 15:30:00');

CREATE TABLE products (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  category TEXT,
  price FLOAT
);

INSERT INTO products VALUES
  (1, 'Laptop', 'Electronics', 999.99),
  (2, 'Mouse', 'Accessories', 29.99),
  (3, 'Keyboard', 'Accessories', 79.99),
  (4, 'Monitor', 'Electronics', 449.99),
  (5, 'Webcam', 'Accessories', 59.99),
  (6, 'Headphones', 'Audio', 149.99),
  (7, 'USB Hub', 'Accessories', 39.99),
  (8, 'Tablet', 'Electronics', 599.99),
  (9, 'Charger', 'Accessories', 24.99),
  (10, 'Stand', 'Accessories', 89.99),
  (11, 'Speakers', 'Audio', 199.99),
  (12, 'Microphone', 'Audio', 129.99);

CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_products_category ON products(category);
`;

const MAX_HISTORY = 200;

class Playground {
  constructor() {
    this.db = null;
    this.history = [];
    this.historyIndex = -1;
    this.currentInput = '';
    this.multiLineBuffer = '';
    this.executing = false;

    this.output = document.getElementById('pg-output');
    this.input = document.getElementById('pg-input');
    this.prompt = document.getElementById('pg-prompt');
    this.loading = document.getElementById('pg-loading');
    this.btnClear = document.getElementById('btn-clear');
    this.btnReset = document.getElementById('btn-reset');

    if (!this.output || !this.input || !this.prompt || !this.loading) return;

    this.bindEvents();
  }

  bindEvents() {
    this.input.addEventListener('keydown', (e) => this.handleKeydown(e));
    if (this.btnClear) this.btnClear.addEventListener('click', () => this.clearOutput());
    if (this.btnReset) this.btnReset.addEventListener('click', () => this.resetDatabase());

    // Hint chips
    document.querySelectorAll('.hint-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const sql = chip.dataset.sql;
        this.input.value = sql;
        this.input.focus();
      });
    });

    // Click on terminal body focuses input
    this.output.addEventListener('click', () => this.input.focus());
  }

  async init() {
    try {
      const wasmUrl = new URL('/assets/wasm/stoolap.js', window.location.origin).href;
      const wasmModule = await import(wasmUrl);
      await wasmModule.default();
      this.db = new wasmModule.StoolapDB();
      this.loadSampleData();
      this.loading.style.display = 'none';
      this.input.disabled = false;
      this.input.focus();

      const version = this.db.version();
      this.appendOutput('welcome', [
        `Stoolap v${version} (WebAssembly)`,
        'Type SQL commands or click a query chip below.',
        'Use Up/Down arrows for history, Ctrl+L to clear.',
        '',
        'Sample tables loaded: users, orders, products',
        'Type SHOW TABLES; to see all tables.',
        ''
      ].join('\n'));
    } catch (err) {
      if (this.loading) {
        this.loading.textContent = '';
        const errorSpan = document.createElement('span');
        errorSpan.className = 'pg-error';
        errorSpan.textContent = 'Failed to load WASM: ' + err.message;
        const helpP = document.createElement('p');
        helpP.style.cssText = 'color:var(--color-text-muted);font-size:0.85rem;margin-top:0.5rem;';
        helpP.textContent = 'Make sure the WASM build exists at /assets/wasm/stoolap_bg.wasm. Build with: wasm-pack build --target web --out-dir docs/assets/wasm';
        this.loading.appendChild(errorSpan);
        this.loading.appendChild(helpP);
      }
    }
  }

  loadSampleData() {
    if (!this.db) return;
    const result = this.db.execute_batch(SAMPLE_SCHEMA);
    let parsed;
    try {
      parsed = JSON.parse(result);
    } catch {
      this.appendOutput('error', `Error loading sample data: ${result}`);
      return;
    }
    if (parsed.type === 'error') {
      this.appendOutput('error', `Error loading sample data: ${parsed.message}`);
    }
  }

  handleKeydown(e) {
    // Ctrl+L: clear
    if (e.ctrlKey && e.key === 'l') {
      e.preventDefault();
      this.clearOutput();
      return;
    }

    // Up arrow: history back
    if (e.key === 'ArrowUp' && !e.shiftKey) {
      e.preventDefault();
      if (this.history.length > 0) {
        if (this.historyIndex === -1) {
          this.currentInput = this.input.value;
          this.historyIndex = this.history.length - 1;
        } else if (this.historyIndex > 0) {
          this.historyIndex--;
        }
        this.input.value = this.history[this.historyIndex];
      }
      return;
    }

    // Down arrow: history forward
    if (e.key === 'ArrowDown' && !e.shiftKey) {
      e.preventDefault();
      if (this.historyIndex >= 0) {
        this.historyIndex++;
        if (this.historyIndex >= this.history.length) {
          this.historyIndex = -1;
          this.input.value = this.currentInput;
        } else {
          this.input.value = this.history[this.historyIndex];
        }
      }
      return;
    }

    // Enter: execute or multi-line
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const line = this.input.value;

      // Accumulate multi-line
      if (this.multiLineBuffer) {
        this.multiLineBuffer += '\n' + line;
      } else {
        this.multiLineBuffer = line;
      }

      const full = this.multiLineBuffer.trim();

      // Transaction commands don't need semicolons
      const upper = full.toUpperCase();
      const isTxnCmd = upper === 'BEGIN' || upper === 'COMMIT' || upper === 'ROLLBACK' || upper.startsWith('BEGIN ');

      if (full.endsWith(';') || isTxnCmd || full === '') {
        this.input.value = '';
        this.multiLineBuffer = '';
        this.prompt.textContent = 'stoolap>';

        if (full) {
          this.executeCommand(full);
        }
      } else {
        // Multi-line continuation
        this.input.value = '';
        this.prompt.textContent = '     ->';
      }
      return;
    }
  }

  executeCommand(sql) {
    if (this.executing) return;
    this.executing = true;

    // Add to history (avoid duplicates), cap at MAX_HISTORY
    if (this.history.length === 0 || this.history[this.history.length - 1] !== sql) {
      this.history.push(sql);
      if (this.history.length > MAX_HISTORY) {
        this.history.shift();
      }
    }
    this.historyIndex = -1;
    this.currentInput = '';

    // Show the command in output
    this.appendOutput('command', sql);

    if (!this.db) {
      this.appendOutput('error', 'Database not loaded');
      this.executing = false;
      return;
    }

    const startTime = performance.now();
    const result = this.db.execute(sql);
    const elapsed = performance.now() - startTime;

    let parsed;
    try {
      parsed = JSON.parse(result);
    } catch {
      this.appendOutput('error', `Invalid response: ${result}`);
      this.executing = false;
      return;
    }

    if (parsed.type === 'error') {
      this.appendOutput('error', `Error: ${parsed.message}`);
    } else if (parsed.type === 'rows') {
      this.appendOutput('table', this.formatTable(parsed.columns, parsed.rows, parsed.count));
      this.appendOutput('info', `${parsed.count} row${parsed.count === 1 ? '' : 's'} in set (${elapsed.toFixed(1)}ms)`);
    } else if (parsed.type === 'affected') {
      const rowText = parsed.affected === 1 ? 'row' : 'rows';
      this.appendOutput('info', `${parsed.affected} ${rowText} affected (${elapsed.toFixed(1)}ms)`);
    }

    this.scrollToBottom();
    this.executing = false;
  }

  formatTable(columns, rows, count) {
    if (!columns || columns.length === 0) return '(empty result)';
    if (!rows || rows.length === 0) return '(0 rows)';

    // Calculate column widths
    const widths = columns.map((col, i) => {
      let max = col.length;
      for (const row of rows) {
        const val = this.formatValue(row[i]);
        if (val.length > max) max = val.length;
      }
      return Math.min(max, 40); // cap width
    });

    // Build table
    const lines = [];
    const separator = '+' + widths.map(w => '-'.repeat(w + 2)).join('+') + '+';

    lines.push(separator);
    lines.push('|' + columns.map((col, i) => ' ' + col.padEnd(widths[i]) + ' ').join('|') + '|');
    lines.push(separator);

    // Smart truncation for large results (mirrors CLI)
    const limit = 40;
    if (count > limit) {
      const topRows = Math.floor(limit / 2);
      const bottomRows = limit - topRows;

      for (let r = 0; r < topRows && r < rows.length; r++) {
        lines.push(this.formatRow(rows[r], widths));
      }
      const hidden = count - limit;
      const mid = '|' + widths.map((w, i) => {
        if (i === Math.floor(widths.length / 2)) {
          const msg = `... (${hidden} more rows) ...`;
          return ' ' + msg.padEnd(w) + ' ';
        }
        return ' '.repeat(w + 2);
      }).join('|') + '|';
      lines.push(mid);

      const startIdx = Math.max(rows.length - bottomRows, topRows);
      for (let r = startIdx; r < rows.length; r++) {
        lines.push(this.formatRow(rows[r], widths));
      }
    } else {
      for (const row of rows) {
        lines.push(this.formatRow(row, widths));
      }
    }

    lines.push(separator);
    return lines.join('\n');
  }

  formatRow(row, widths) {
    return '|' + row.map((val, i) => {
      const formatted = this.formatValue(val);
      const truncated = formatted.length > widths[i]
        ? formatted.substring(0, widths[i] - 1) + '\u2026'
        : formatted;

      // Right-align numbers
      if (typeof val === 'number') {
        return ' ' + truncated.padStart(widths[i]) + ' ';
      }
      return ' ' + truncated.padEnd(widths[i]) + ' ';
    }).join('|') + '|';
  }

  formatValue(val) {
    if (val === null || val === undefined) return 'NULL';
    if (typeof val === 'boolean') return val ? 'true' : 'false';
    if (typeof val === 'number') {
      if (Number.isInteger(val)) return val.toString();
      // Float formatting (mirrors CLI's format_float)
      if (val === Math.trunc(val)) return val.toFixed(1);
      return parseFloat(val.toFixed(6)).toString();
    }
    return String(val);
  }

  appendOutput(type, text) {
    const div = document.createElement('div');
    div.className = `pg-line pg-${type}`;

    if (type === 'command') {
      const promptSpan = document.createElement('span');
      promptSpan.className = 'pg-cmd-prompt';
      promptSpan.textContent = 'stoolap>';
      div.appendChild(promptSpan);
      div.appendChild(document.createTextNode(' ' + text));
    } else if (type === 'table') {
      const pre = document.createElement('pre');
      pre.className = 'pg-table-pre';
      pre.textContent = text;
      div.appendChild(pre);
    } else {
      div.textContent = text;
    }

    this.output.appendChild(div);
    this.scrollToBottom();
  }

  scrollToBottom() {
    this.output.scrollTop = this.output.scrollHeight;
  }

  clearOutput() {
    // Keep only the loading div (hidden), remove everything else
    const children = Array.from(this.output.children);
    for (const child of children) {
      if (child.id !== 'pg-loading') {
        child.remove();
      }
    }
  }

  async resetDatabase() {
    if (!this.db || this.executing) return;
    this.executing = true;
    this.clearOutput();

    try {
      const wasmUrl = new URL('/assets/wasm/stoolap.js', window.location.origin).href;
      const wasmModule = await import(wasmUrl);
      this.db = new wasmModule.StoolapDB();
      this.loadSampleData();
      this.appendOutput('info', 'Database reset. Sample tables reloaded.');
    } catch (err) {
      this.appendOutput('error', `Reset failed: ${err.message}`);
    } finally {
      this.executing = false;
    }
  }
}

// Boot after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', function () {
    const playground = new Playground();
    playground.init();
  });
} else {
  const playground = new Playground();
  playground.init();
}
