// BenchFlow's leak check: run as the agent user inside a sandbox while the
// provider forwarder holds a key, it scans everything this user can read for
// that key, knowing only the key's SHA-256, its length and its public format
// prefix (so the key itself never enters the sandbox for the check).
//
// Usage: node leak_scan.js <sha256 hex> <length> <prefix>
// Prints one JSON object: what was scanned, and where the key was found.

'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const [target, lengthArg, prefixArg] = process.argv.slice(2);
const length = Number(lengthArg);
const prefix = Buffer.from(prefixArg);
const CHUNK = 16 * 1024 * 1024;

const result = {
  uid: process.getuid(),
  files: 0,
  bytes: 0,
  unreadable_files: 0,
  unreadable_dirs: 0,
  proc_cmdline_read: 0,
  proc_environ_read: 0,
  proc_environ_denied: 0,
  candidates: 0,
  hits: [],
  forwarder_dir: null,
};

// Every window that starts with the prefix is hashed and compared.
function check(buf, where) {
  let at = buf.indexOf(prefix);
  while (at !== -1) {
    if (at + length <= buf.length) {
      result.candidates += 1;
      const hash = crypto.createHash('sha256').update(buf.subarray(at, at + length)).digest('hex');
      if (hash === target) {
        result.hits.push(where);
        return;
      }
    }
    at = buf.indexOf(prefix, at + 1);
  }
}

function scanFile(file) {
  let fd;
  try {
    fd = fs.openSync(file, 'r');
  } catch {
    result.unreadable_files += 1;
    return;
  }
  try {
    // Chunks overlap by length - 1 bytes, so no window is missed.
    const buf = Buffer.alloc(CHUNK + length);
    let carry = 0;
    for (;;) {
      const read = fs.readSync(fd, buf, carry, CHUNK, null);
      if (read === 0) break;
      const filled = carry + read;
      result.bytes += read;
      check(buf.subarray(0, filled), file);
      carry = Math.min(length - 1, filled);
      buf.copy(buf, 0, filled - carry, filled);
    }
    result.files += 1;
  } catch {
    result.unreadable_files += 1;
  } finally {
    fs.closeSync(fd);
  }
}

// 1. This process's own environment.
check(Buffer.from(JSON.stringify(process.env)), 'own environment');

// 2. Every process's command line and, where readable, environment.
for (const pid of fs.readdirSync('/proc').filter((name) => /^\d+$/.test(name))) {
  try {
    check(fs.readFileSync(`/proc/${pid}/cmdline`), `/proc/${pid}/cmdline`);
    result.proc_cmdline_read += 1;
  } catch {
    // The process ended.
  }
  try {
    check(fs.readFileSync(`/proc/${pid}/environ`), `/proc/${pid}/environ`);
    result.proc_environ_read += 1;
  } catch {
    result.proc_environ_denied += 1;
  }
}

// 3. Every regular file this user can read, outside /proc, /sys and /dev.
const SKIP = new Set(['/proc', '/sys', '/dev']);
function walk(dir) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    result.unreadable_dirs += 1;
    return;
  }
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (SKIP.has(full)) continue;
    if (entry.isDirectory()) walk(full);
    else if (entry.isFile()) scanFile(full);
  }
}
walk('/');

// 4. The forwarder's folder must be closed to this user.
try {
  fs.readdirSync('/opt/benchflow/forwarder');
  result.forwarder_dir = 'readable';
} catch (e) {
  result.forwarder_dir = e.code || 'error';
}

process.stdout.write(`${JSON.stringify(result)}\n`);
