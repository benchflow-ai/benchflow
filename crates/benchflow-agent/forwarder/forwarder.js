// BenchFlow's provider forwarder.
//
// Holds a model provider's key for an agent that must not see it. BenchFlow
// starts it as root inside the sandbox before the agent. It listens on
// loopback, adds the key to every request in the header the provider wants,
// and streams the answers back as they arrive (server-sent events included).
// It serves only the routes in its config, each a path prefix mapped onto one
// provider endpoint, so it cannot be used to reach anything else. Requests
// are read whole (they are prompts, not uploads), so one whose connection
// failed before the provider could have seen it is sent again, up to twice.
//
// Usage: node forwarder.js <config.json>
//
// The config holds no secret:
//   {"listen": "127.0.0.1", "port": 0, "key_file": "/opt/benchflow/forwarder/key",
//    "routes": [{"prefix": "/anthropic", "upstream": "https://...", "auth": "bearer",
//                "models": ["deepseek/deepseek-v4-flash-0731"]}]}
// A prefix is `/name`, or empty for a gateway's own API served from the root.
// A request that names a model not in `models` (a `[...]` variant suffix
// aside), in its JSON body or in the headers a gateway's own protocol uses, is
// refused with 403; an empty list allows any model. On a router that picks
// among upstream providers, the route also has one of:
//   `only`: the upstreams BenchFlow allows (the Vercel AI Gateway). Every JSON
//     object body gets `providerOptions.gateway.only` set to that list, merged
//     into any `providerOptions` the agent sent and replacing any `only` it set;
//   `ignore`: the upstreams BenchFlow denies (OpenRouter). Every JSON object
//     body gets them added to `provider.ignore`, keeping the agent's own
//     provider preferences and entries.
// Either way, never Baseten.
// On a route that checks the model or pins upstreams, a request body the
// forwarder cannot read as a JSON object (a compressed one, or one that is
// not JSON) is refused with 415, so nothing it forwards there went unchecked.
// An empty body passes: it names no model and has nothing to pin.
//
// The key file must be readable by its owner only. The forwarder reads it once
// and deletes it, so the key is never in a file, in this process's command line
// or environment, or in a log. An answer that ever echoes the key gets it
// masked (same length, `*`), which is why upstream answers are asked for
// without compression.
//
// stdout: one line once listening: {"benchflow_forwarder": "ready", "port": N}
// stderr: one JSON line per request (method, path, status, bytes, ms) and errors.
// A model call's line also has the token counts its answer reported (`usage`:
// input not read from a cache, cache reads, cache writes, output) and the cost
// the provider charged for it (`cost`, US dollars), when the answer said so
// (the Vercel AI Gateway and OpenRouter do), so a trial's cost is what the
// provider billed, not an agent's estimate at another model's prices.

'use strict';

const crypto = require('node:crypto');
const fs = require('node:fs');
const http = require('node:http');
const https = require('node:https');
const { Transform } = require('node:stream');

// Whatever key the agent sends is replaced.
const AUTH_HEADERS = new Set([
  'authorization',
  'x-api-key',
  'api-key',
  'x-goog-api-key',
  'proxy-authorization',
]);
// Headers about one connection, never passed on (RFC 9110 7.6.1), plus
// `expect`, which this forwarder answers itself.
const HOP_BY_HOP = new Set([
  'connection',
  'keep-alive',
  'proxy-connection',
  'proxy-authenticate',
  'te',
  'trailer',
  'transfer-encoding',
  'upgrade',
  'host',
  'expect',
]);

// Requests that may be sent twice; connection errors after which a request
// can be retried; errors that mean the request never left this machine.
const IDEMPOTENT = new Set(['GET', 'HEAD', 'OPTIONS']);
const RETRYABLE = new Set(['ECONNRESET', 'EPIPE', 'ETIMEDOUT', 'ECONNREFUSED', 'EAI_AGAIN']);
const CONNECT_ERRORS = new Set(['ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'ENETUNREACH', 'EHOSTUNREACH']);
const MAX_BODY = 64 * 1024 * 1024;
// Where a gateway's own protocol (the AI SDK's, which fx speaks) names the
// model: in headers, not in the body.
const MODEL_HEADERS = ['ai-language-model-id', 'x-vercel-ai-gateway-model', 'x-vercel-ai-gateway-model-id'];

function log(fields) {
  process.stderr.write(JSON.stringify({ time: new Date().toISOString(), ...fields }) + '\n');
}

function fail(message) {
  log({ error: message });
  process.exit(2);
}

function readKey(path) {
  let stat;
  try {
    stat = fs.statSync(path);
  } catch (e) {
    fail(`cannot read the key file: ${e.code || 'error'}`);
  }
  if ((stat.mode & 0o077) !== 0) {
    fail('the key file must be readable by its owner only (mode 600)');
  }
  const key = fs.readFileSync(path, 'utf8').trim();
  try {
    fs.unlinkSync(path);
  } catch (e) {
    fail(`cannot delete the key file after reading it: ${e.code || 'error'}`);
  }
  if (!key) fail('the key file is empty');
  if (/[\s]/.test(key)) fail('the key has whitespace in it');
  return key;
}

function compileRoute(route) {
  if (typeof route.prefix !== 'string' || !/^(\/[a-z0-9_-]+)?$/.test(route.prefix)) {
    fail('a route prefix must look like /name, or be empty for the root');
  }
  let url;
  try {
    url = new URL(route.upstream);
  } catch {
    fail(`the upstream of ${route.prefix} is not a URL`);
  }
  if (url.protocol !== 'https:' && url.protocol !== 'http:') {
    fail(`the upstream of ${route.prefix} must be http or https`);
  }
  if (url.search || url.hash || url.username || url.password) {
    fail(`the upstream of ${route.prefix} must have no query, fragment, or user`);
  }
  if (route.auth !== 'bearer' && route.auth !== 'x-api-key') {
    fail(`the auth of ${route.prefix} must be bearer or x-api-key`);
  }
  const models = route.models === undefined ? [] : route.models;
  if (!Array.isArray(models) || models.some((m) => typeof m !== 'string')) {
    fail(`the models of ${route.prefix} must be a list of strings`);
  }
  const names = (field) => {
    const list = route[field] === undefined || route[field] === null ? null : route[field];
    if (list !== null && (!Array.isArray(list) || list.length === 0 || list.some((m) => typeof m !== 'string' || !m))) {
      fail(`the upstreams (${field}) of ${route.prefix || '/'} must be a non-empty list of names`);
    }
    return list;
  };
  const only = names('only');
  const ignore = names('ignore');
  if (only !== null && ignore !== null) {
    fail(`the route ${route.prefix || '/'} has both only and ignore upstreams`);
  }
  const client = url.protocol === 'https:' ? https : http;
  return {
    prefix: route.prefix,
    url,
    basePath: url.pathname.replace(/\/+$/, ''),
    auth: route.auth,
    models,
    only,
    ignore,
    client,
    agent: new client.Agent({ keepAlive: true }),
  };
}

// No dot segments (raw or percent-encoded), no backslashes, no NUL: the path
// under a route's prefix can never climb out of the provider's base path.
function safePath(path) {
  if (path.includes('\\') || path.includes('\0')) return false;
  for (const segment of path.split('/')) {
    let decoded;
    try {
      decoded = decodeURIComponent(segment);
    } catch {
      return false;
    }
    if (decoded === '.' || decoded === '..' || /[\\/\0]/.test(decoded)) return false;
  }
  return true;
}

function connectionTokens(headers) {
  return String(headers.connection || '')
    .split(',')
    .map((token) => token.trim().toLowerCase())
    .filter(Boolean);
}

// Replaces every occurrence of the secret with `*`s of the same length, so a
// Content-Length stays right. Holds back only a tail that could be the start
// of the secret, which a server-sent event (ending in a blank line) never is.
class Masker extends Transform {
  constructor(secret) {
    super();
    this.secret = Buffer.from(secret, 'utf8');
    this.mask = Buffer.alloc(this.secret.length, 0x2a);
    this.held = Buffer.alloc(0);
    this.bytes = 0;
  }

  _transform(chunk, _encoding, callback) {
    this.bytes += chunk.length;
    let buf = this.held.length ? Buffer.concat([this.held, chunk]) : chunk;
    let at = buf.indexOf(this.secret);
    if (at !== -1) {
      buf = Buffer.from(buf);
      while (at !== -1) {
        this.mask.copy(buf, at);
        at = buf.indexOf(this.secret, at + this.secret.length);
      }
    }
    const hold = partialTail(buf, this.secret);
    this.held = Buffer.from(buf.subarray(buf.length - hold));
    if (buf.length > hold) this.push(buf.subarray(0, buf.length - hold));
    callback();
  }

  _flush(callback) {
    if (this.held.length) this.push(this.held);
    callback();
  }
}

// The length of the longest end of `buf` that begins the secret.
function partialTail(buf, secret) {
  const longest = Math.min(secret.length - 1, buf.length);
  for (let n = longest; n > 0; n -= 1) {
    if (buf[buf.length - n] === secret[0] && buf.subarray(buf.length - n).equals(secret.subarray(0, n))) {
      return n;
    }
  }
  return 0;
}

// The request body as a JSON object, or null when it is not one.
function jsonObject(body) {
  const text = body.toString('utf8').trimStart();
  if (!text.startsWith('{')) return null;
  try {
    const value = JSON.parse(text);
    return isPlainObject(value) ? value : null;
  } catch {
    return null;
  }
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

// Pin a gateway request to the allowed upstreams.
function pinUpstreams(value, only) {
  const options = isPlainObject(value.providerOptions) ? value.providerOptions : {};
  const gateway = isPlainObject(options.gateway) ? options.gateway : {};
  value.providerOptions = { ...options, gateway: { ...gateway, only: [...only] } };
}

// Keep a router's request off the denied upstreams (OpenRouter's
// `provider.ignore`), adding to whatever the agent asked to ignore.
function ignoreUpstreams(value, denied) {
  const provider = isPlainObject(value.provider) ? value.provider : {};
  const ignore = Array.isArray(provider.ignore) ? [...provider.ignore] : [];
  for (const name of denied) {
    if (!ignore.includes(name)) ignore.push(name);
  }
  value.provider = { ...provider, ignore };
}

// The models a request names: its JSON body's `model`, the fallback lists
// a router takes (`models`, `providerOptions.gateway.models`), each request
// of a message batch (`requests[].params.model`), and a gateway protocol's
// model headers. A name that is not a string is returned as itself, so it
// is never one of the run's models.
function askedModels(parsed, headers) {
  const asked = [];
  const add = (value) => asked.push(typeof value === 'string' ? value : `(${typeof value})`);
  const addAll = (list) => {
    if (list === undefined) return;
    if (Array.isArray(list)) list.forEach(add);
    else add(list);
  };
  if (parsed !== null) {
    if (parsed.model !== undefined) add(parsed.model);
    addAll(parsed.models);
    const options = isPlainObject(parsed.providerOptions) ? parsed.providerOptions : {};
    if (isPlainObject(options.gateway)) addAll(options.gateway.models);
    if (Array.isArray(parsed.requests)) {
      for (const r of parsed.requests) {
        add(isPlainObject(r) && isPlainObject(r.params) ? r.params.model : undefined);
      }
    } else if (parsed.requests !== undefined) {
      add(parsed.requests);
    }
  }
  for (const name of MODEL_HEADERS) {
    const value = headers[name];
    if (typeof value === 'string' && value) asked.push(value);
  }
  return asked;
}

// A batch names its models in a file uploaded earlier, which the forwarder
// never sees; on a route kept to the run's model it is not served.
function isBatch(path) {
  return /\/batches(\/|$)/.test(path);
}

// How much of an answer is kept to read its usage from (answers are model
// replies, a few MiB at most).
const USAGE_KEEP = 16 * 1024 * 1024;

// A count, or null.
function count(value) {
  const n = typeof value === 'string' ? Number(value) : value;
  return typeof n === 'number' && Number.isFinite(n) && n >= 0 ? n : null;
}

// One usage object's counts, as {input, cache_read, cache_write, output},
// input being the prompt tokens not read from or written to a cache. Reads
// OpenAI's Chat Completions (`prompt_tokens`, cached ones among them),
// OpenAI's Responses API (`input_tokens` with `input_tokens_details`, cached
// ones among them), Anthropic's Messages (`input_tokens` without the cache's)
// and the AI SDK's own protocol (`inputTokens`, a number or {total, noCache,
// cacheRead, cacheWrite}).
function readUsage(u) {
  if (!isPlainObject(u)) return null;
  const n = (v) => count(v) ?? 0;
  if (u.prompt_tokens !== undefined || u.completion_tokens !== undefined) {
    const read = n(u.prompt_tokens_details && u.prompt_tokens_details.cached_tokens);
    const write = n(u.cache_creation_input_tokens);
    return { input: Math.max(0, n(u.prompt_tokens) - read - write), cache_read: read, cache_write: write, output: n(u.completion_tokens) };
  }
  if (u.inputTokens !== undefined || u.outputTokens !== undefined) {
    const out = isPlainObject(u.outputTokens) ? n(u.outputTokens.total) : n(u.outputTokens);
    if (isPlainObject(u.inputTokens)) {
      const t = u.inputTokens;
      const read = n(t.cacheRead);
      const write = n(t.cacheWrite);
      const input = count(t.noCache) ?? Math.max(0, n(t.total) - read - write);
      return { input, cache_read: read, cache_write: write, output: out };
    }
    const read = n(u.cachedInputTokens);
    return { input: Math.max(0, n(u.inputTokens) - read), cache_read: read, cache_write: 0, output: out };
  }
  if (u.input_tokens !== undefined || u.output_tokens !== undefined) {
    if (isPlainObject(u.input_tokens_details)) {
      const read = n(u.input_tokens_details.cached_tokens);
      return { input: Math.max(0, n(u.input_tokens) - read), cache_read: read, cache_write: 0, output: n(u.output_tokens) };
    }
    return {
      input: n(u.input_tokens),
      cache_read: n(u.cache_read_input_tokens),
      cache_write: n(u.cache_creation_input_tokens),
      output: n(u.output_tokens),
    };
  }
  return null;
}

// The cost a router put in an answer event: `usage.cost` (OpenRouter, the
// Vercel AI Gateway), or the gateway's provider metadata
// (`provider_metadata.gateway.cost`, `providerMetadata.gateway.cost`), at
// the top, in a message, a response, or a choice's message or delta.
function readCost(event) {
  const places = [event, event.message, event.response, event.item];
  if (Array.isArray(event.choices)) {
    for (const c of event.choices) {
      if (isPlainObject(c)) places.push(c.message, c.delta);
    }
  }
  let cost = null;
  for (const p of places) {
    if (!isPlainObject(p)) continue;
    for (const meta of [p.provider_metadata, p.providerMetadata]) {
      if (isPlainObject(meta) && isPlainObject(meta.gateway)) {
        const c = count(meta.gateway.cost);
        if (c !== null) cost = Math.max(cost ?? 0, c);
      }
    }
    if (isPlainObject(p.usage)) {
      const c = count(p.usage.cost);
      if (c !== null) cost = Math.max(cost ?? 0, c);
    }
  }
  return cost;
}

// What one answer (a JSON body, or server-sent events, or JSON lines) says
// it used and cost. A streamed answer repeats or grows its counts from event
// to event (Anthropic's input in message_start, output in message_delta), so
// each count is the largest seen. Null when it says neither.
function tally(text) {
  const events = [];
  const whole = text.trimStart();
  if (whole.startsWith('{')) {
    try {
      events.push(JSON.parse(whole));
    } catch {
      // JSON lines, below.
    }
  }
  if (events.length === 0) {
    for (const raw of text.split('\n')) {
      let line = raw.trim();
      if (line.startsWith('data:')) line = line.slice(5).trim();
      if (!line.startsWith('{')) continue;
      try {
        events.push(JSON.parse(line));
      } catch {
        // Not an event.
      }
    }
  }
  let usage = null;
  let cost = null;
  for (const e of events) {
    if (!isPlainObject(e)) continue;
    for (const u of [e.usage, e.message && e.message.usage, e.response && e.response.usage]) {
      const counts = readUsage(u);
      if (counts === null) continue;
      usage = usage || { input: 0, cache_read: 0, cache_write: 0, output: 0 };
      for (const k of Object.keys(usage)) usage[k] = Math.max(usage[k], counts[k]);
    }
    const c = readCost(e);
    if (c !== null) cost = Math.max(cost ?? 0, c);
  }
  if (usage === null && cost === null) return null;
  return { ...(usage ? { usage } : {}), ...(cost !== null ? { cost } : {}) };
}

function deny(res, status, message, started, req, path) {
  const body = JSON.stringify({ error: { type: 'benchflow_forwarder', message } });
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
  log({ method: req.method, path, status, bytes: body.length, ms: Date.now() - started, note: message });
}

function handle(req, res, routes, key) {
  const started = Date.now();
  const target = req.url || '';
  if (!target.startsWith('/')) {
    deny(res, 400, 'only paths are served, not absolute URLs', started, req, '(absolute)');
    return;
  }
  const q = target.indexOf('?');
  const path = q < 0 ? target : target.slice(0, q);
  const query = q < 0 ? '' : target.slice(q);
  if (!safePath(path)) {
    deny(res, 400, 'that path is not served', started, req, '(unsafe)');
    return;
  }
  const route = routes.find((r) => path === r.prefix || path.startsWith(`${r.prefix}/`));
  if (!route) {
    const served = routes.map((r) => `${r.prefix}/`).join(' and ');
    deny(res, 404, `only ${served} is served`, started, req, path);
    return;
  }
  const upstreamPath = `${route.basePath}${path.slice(route.prefix.length)}${query}` || '/';

  const dropped = new Set(connectionTokens(req.headers));
  const headers = {};
  for (const [name, value] of Object.entries(req.headers)) {
    if (HOP_BY_HOP.has(name) || AUTH_HEADERS.has(name) || dropped.has(name)) continue;
    headers[name] = value;
  }
  headers.host = route.url.host;
  headers['accept-encoding'] = 'identity';
  if (route.auth === 'bearer') {
    headers.authorization = `Bearer ${key}`;
  } else {
    headers['x-api-key'] = key;
  }

  // The body is read whole (prompts, not uploads), so a request whose
  // connection failed before the provider can have seen it is sent again.
  const chunks = [];
  let size = 0;
  let current = null;
  // How the body was pinned to allowed upstreams, for the log.
  let pinned;
  // The body's hash, for the log: a request sent again after a refusal
  // has the same one, so BenchFlow can tell a client's backoff (waiting for
  // that answer) from the agent going on without it.
  let sent;
  req.on('data', (chunk) => {
    size += chunk.length;
    if (size > MAX_BODY) {
      // Read the rest and drop it, so the client gets the 413 below rather
      // than a reset connection.
      chunks.length = 0;
      return;
    }
    chunks.push(chunk);
  });
  req.on('error', () => {
    if (current) current.destroy();
  });
  res.on('close', () => {
    if (!res.writableFinished && current) current.destroy();
  });
  req.on('end', () => {
    if (size > MAX_BODY) {
      deny(res, 413, 'the request body is larger than 64 MiB', started, req, path);
      return;
    }
    let body = Buffer.concat(chunks);
    // A body it cannot read would pass the model check and the pin unchecked.
    const enforced = route.models.length > 0 || route.only !== null || route.ignore !== null;
    const encoding = String(req.headers['content-encoding'] || 'identity')
      .trim()
      .toLowerCase()
      .slice(0, 40);
    if (enforced && body.length > 0 && encoding !== 'identity') {
      deny(
        res,
        415,
        `BenchFlow's forwarder cannot read a request body sent with content-encoding ${encoding}, so it cannot check the model or keep the request off denied upstreams; send the body uncompressed`,
        started,
        req,
        path,
      );
      return;
    }
    const parsed = body.length > 0 ? jsonObject(body) : null;
    if (enforced && body.length > 0 && parsed === null) {
      deny(
        res,
        415,
        "BenchFlow's forwarder takes only JSON object bodies here, which it checks for the run's model and keeps off denied upstreams; this body is not one",
        started,
        req,
        path,
      );
      return;
    }
    // Only the run's model: an agent's helper asking for another one (a
    // title generator's small model, say) is refused here, before it can be
    // billed to the key.
    if (route.models.length > 0 && isBatch(path) && req.method !== 'GET') {
      deny(res, 403, `this run may use only ${route.models.join(', ')}; batches are not served, since the forwarder cannot check their models`, started, req, path);
      return;
    }
    const refused =
      route.models.length > 0
        ? askedModels(parsed, req.headers).find((m) => !route.models.includes(m.replace(/\[[^\]]*\]$/, '')))
        : undefined;
    if (refused !== undefined) {
      deny(res, 403, `this run may use only ${route.models.join(', ')}; the request asked for ${refused}`, started, req, path);
      return;
    }
    // Never a denied upstream, whatever the agent asked for.
    if (parsed !== null && route.only !== null) {
      pinUpstreams(parsed, route.only);
      pinned = 'only';
    } else if (parsed !== null && route.ignore !== null) {
      ignoreUpstreams(parsed, route.ignore);
      pinned = 'ignore';
    }
    // What was checked is what is sent: a body with a key twice, or bytes
    // that are not UTF-8, could read one way here and another upstream.
    if (parsed !== null && (pinned !== undefined || route.models.length > 0)) {
      body = Buffer.from(JSON.stringify(parsed));
      headers['content-type'] = 'application/json';
    }
    if (body.length > 0 || req.headers['content-length'] !== undefined) {
      headers['content-length'] = String(body.length);
    }
    if (body.length > 0) sent = crypto.createHash('sha256').update(body).digest('hex').slice(0, 16);
    send(1);

    function send(attempt) {
      const upstream = route.client.request(
        {
          protocol: route.url.protocol,
          hostname: route.url.hostname,
          port: route.url.port || undefined,
          method: req.method,
          path: upstreamPath,
          headers,
          agent: route.agent,
        },
        (answer) => respond(answer),
      );
      current = upstream;
      upstream.setTimeout(0);
      upstream.on('error', (e) => {
        if (res.headersSent || res.destroyed) {
          res.destroy();
          return;
        }
        // Safe to send again: an idempotent request, a connection that never
        // opened, or a pooled one the provider had already closed.
        const unsent =
          CONNECT_ERRORS.has(e.code) || (e.code === 'ECONNRESET' && upstream.reusedSocket);
        if (attempt < 3 && (unsent || (IDEMPOTENT.has(req.method) && RETRYABLE.has(e.code)))) {
          log({ method: req.method, path, note: `retrying after ${e.code}`, attempt });
          setTimeout(() => send(attempt + 1), 200 * attempt);
          return;
        }
        deny(res, 502, `the provider could not be reached (${e.code || 'error'})`, started, req, path);
      });
      upstream.end(body);
    }
  });

  function respond(answer) {
    const skip = new Set(connectionTokens(answer.headers));
    const out = {};
    // A provider that echoes the request's headers would show the key.
    const hide = (v) => (typeof v === 'string' ? v.split(key).join('*'.repeat(key.length)) : v);
    for (const [name, value] of Object.entries(answer.headers)) {
      if (HOP_BY_HOP.has(name) || skip.has(name)) continue;
      out[name] = Array.isArray(value) ? value.map(hide) : hide(value);
    }
    try {
      res.writeHead(answer.statusCode || 502, answer.statusMessage, out);
      res.flushHeaders();
    } catch {
      answer.destroy();
      if (!res.headersSent) deny(res, 502, 'the provider sent headers that cannot be passed on', started, req, path);
      return;
    }
    const masker = new Masker(key);
    // A model call's answer is kept (up to USAGE_KEEP) to read its usage
    // and cost from; a token count request reports no use.
    const kept = [];
    let keptBytes = 0;
    const counted = req.method === 'POST' && !/count_tokens/.test(path);
    if (counted) {
      answer.on('data', (chunk) => {
        if (keptBytes + chunk.length <= USAGE_KEEP) kept.push(chunk);
        keptBytes += chunk.length;
      });
    }
    answer.on('error', () => res.destroy());
    answer.pipe(masker).pipe(res);
    res.on('finish', () => {
      let spent = null;
      if (counted && keptBytes > 0 && keptBytes <= USAGE_KEEP) {
        try {
          spent = tally(Buffer.concat(kept).toString('utf8'));
        } catch {
          spent = null;
        }
      }
      log({
        method: req.method,
        path,
        status: answer.statusCode,
        bytes: masker.bytes,
        ms: Date.now() - started,
        ...(sent ? { body_hash: sent } : {}),
        ...(pinned ? { pinned } : {}),
        ...(spent || {}),
      });
    });
  }
}

function main() {
  const configPath = process.argv[2];
  if (!configPath) fail('usage: forwarder.js <config.json>');
  let config;
  try {
    config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  } catch (e) {
    fail(`cannot read the config: ${e.code || e.message}`);
  }
  if (!Array.isArray(config.routes) || config.routes.length === 0) fail('the config has no routes');
  const routes = config.routes.map(compileRoute);
  const key = readKey(config.key_file);

  const server = http.createServer((req, res) => {
    req.socket.setNoDelay(true);
    handle(req, res, routes, key);
  });
  server.requestTimeout = 0;
  server.timeout = 0;
  server.on('upgrade', (_req, socket) => {
    socket.end('HTTP/1.1 501 Not Implemented\r\nconnection: close\r\ncontent-length: 0\r\n\r\n');
  });
  server.on('clientError', (_e, socket) => {
    if (socket.writable) socket.end('HTTP/1.1 400 Bad Request\r\nconnection: close\r\ncontent-length: 0\r\n\r\n');
  });
  server.on('error', (e) => fail(`cannot listen: ${e.code || 'error'}`));
  server.listen(Number(config.port) || 0, config.listen || '127.0.0.1', () => {
    process.stdout.write(`${JSON.stringify({ benchflow_forwarder: 'ready', port: server.address().port })}\n`);
  });
}

// A failure in one request must not take every other request down with it.
process.on('uncaughtException', (e) => log({ error: `internal: ${e && e.code ? e.code : 'exception'}` }));

main();
