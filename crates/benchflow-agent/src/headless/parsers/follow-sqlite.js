// BenchFlow: run an agent (OpenClaw) and stream the transcript it keeps in SQLite.
//
// usage: node follow-sqlite.js DATABASE SESSION_KEY -- COMMAND...
//
// OpenClaw 2026.9 keeps a session's transcript in its agent database
// (`~/.openclaw/agents/<agent>/agent/openclaw-agent.sqlite`): the session key
// (`agent:main:main`) names the current session in `session_nodes`, and its
// events are rows of `transcript_events` in `seq` order, each the JSON line
// the older JSON Lines transcript held, stored as text (`event_json`) or,
// when large, zstd-compressed (`event_zstd`). Runs COMMAND, printing its
// stdout and stderr line by line as they come. Every two seconds, and once
// more after COMMAND exits, each new event is printed as
// {"@bf":"session","item":<event>}, the line follow.sh prints. Exits with
// COMMAND's exit status. Needs Node.js 23.4 or later (node:sqlite, zstd),
// which OpenClaw itself needs.

'use strict';

const { spawn } = require('node:child_process');
const fs = require('node:fs');
const zlib = require('node:zlib');

const [database, sessionKey, dashes, ...command] = process.argv.slice(2);
if (!database || !sessionKey || dashes !== '--' || command.length === 0) {
  process.stderr.write('usage: node follow-sqlite.js DATABASE SESSION_KEY -- COMMAND...\n');
  process.exit(2);
}

let sqlite = null;
try {
  sqlite = require('node:sqlite');
} catch {
  process.stderr.write('BenchFlow: this Node.js has no node:sqlite, so the session is not followed\n');
}

// The last event printed, per session id.
const printed = new Map();
let warned = false;

function poll() {
  if (sqlite === null || !fs.existsSync(database)) return;
  let db;
  try {
    db = new sqlite.DatabaseSync(database, { readOnly: true, timeout: 1000 });
    const node = db
      .prepare('SELECT current_session_id AS id FROM session_nodes WHERE session_key = ?')
      .get(sessionKey);
    if (!node) return;
    const after = printed.has(node.id) ? printed.get(node.id) : -1;
    const rows = db
      .prepare(
        'SELECT seq, event_json, event_zstd FROM transcript_events WHERE session_id = ? AND seq > ? ORDER BY seq',
      )
      .all(node.id, after);
    for (const row of rows) {
      let text = row.event_json;
      if (text === null && row.event_zstd !== null) {
        text = zlib.zstdDecompressSync(Buffer.from(row.event_zstd)).toString('utf8');
      }
      printed.set(node.id, row.seq);
      if (text === null) continue;
      let event;
      try {
        event = JSON.parse(text);
      } catch {
        continue;
      }
      process.stdout.write(`${JSON.stringify({ '@bf': 'session', item: event })}\n`);
    }
  } catch (e) {
    // The database is being created or migrated: try again next time.
    if (!warned && !/locked|busy|no such table/i.test(String(e && e.message))) {
      warned = true;
      process.stderr.write(`BenchFlow: cannot read OpenClaw's session yet: ${e && e.message}\n`);
    }
  } finally {
    if (db) db.close();
  }
}

// The command's output, a line at a time, so an event never lands in the
// middle of one of its lines.
function relay(stream) {
  let held = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk) => {
    held += chunk;
    const end = held.lastIndexOf('\n');
    if (end >= 0) {
      process.stdout.write(held.slice(0, end + 1));
      held = held.slice(end + 1);
    }
  });
  return () => {
    if (held) process.stdout.write(`${held}\n`);
    held = '';
  };
}

const child = spawn(command[0], command.slice(1), { stdio: ['ignore', 'pipe', 'pipe'] });
const flushOut = relay(child.stdout);
const flushErr = relay(child.stderr);
for (const signal of ['SIGTERM', 'SIGINT', 'SIGHUP']) {
  process.on(signal, () => child.kill(signal));
}
const timer = setInterval(poll, 2000);
child.on('error', (e) => {
  process.stderr.write(`BenchFlow: cannot start ${command[0]}: ${e.message}\n`);
  process.exit(127);
});
child.on('close', (code, signal) => {
  clearInterval(timer);
  flushOut();
  flushErr();
  poll();
  process.exitCode = code === null ? 128 + (require('node:os').constants.signals[signal] || 0) : code;
});
