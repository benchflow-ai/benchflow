#!/usr/bin/env node
const fs = require("fs");
const http = require("http");
const https = require("https");
const { URL } = require("url");

function getArg(name) {
  const prefix = `--${name}=`;
  const arg = process.argv.find((value) => value.startsWith(prefix));
  return arg ? arg.slice(prefix.length) : "";
}

function getConfig(name, argName) {
  return process.env[`BENCHFLOW_USAGE_PROXY_${name}`] || getArg(argName);
}

const target = new URL(getConfig("TARGET", "target").replace(/\/+$/, ""));
// Google AI Studio (Gemini) needs upstream path normalization. With a custom api_base
// (this proxy), litellm builds gemini URLs as `{api_base}/models/{model}:{action}` — it
// omits the required `/v1beta` version prefix, and for context caching it emits the bogus
// `:cachedContents` model-action instead of Google's real top-level `/v1beta/cachedContents`
// collection (litellm vertex_llm_base._check_custom_proxy, still present in 1.88.x). Without
// this rewrite every gemini call — especially the prompt-cache probe — 404s through the proxy.
// Gated on the gemini host, so all other providers (openai/anthropic/bedrock) are untouched.
const isGeminiTarget = /(^|\.)generativelanguage\.googleapis\.com$/i.test(target.hostname);
const statePath = getConfig("STATE_PATH", "state");
const logPath = getConfig("LOG_PATH", "log");
const pidPath = getConfig("PID_PATH", "pid");
const sessionId = getConfig("SESSION_ID", "session-id");
const agentName = getConfig("AGENT_NAME", "agent-name");
const promptCacheRetention = getConfig("PROMPT_CACHE_RETENTION", "prompt-cache-retention");

function sanitizeHeaders(headers) {
  const result = { ...headers };
  for (const key of Object.keys(result)) {
    const lower = key.toLowerCase();
    if (["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "upgrade"].includes(lower)) {
      delete result[key];
    }
  }
  return result;
}

function responseHeaders(headers) {
  const result = sanitizeHeaders(headers);
  delete result["content-length"];
  delete result["Content-Length"];
  delete result["transfer-encoding"];
  delete result["Transfer-Encoding"];
  result["connection"] = "close";
  return result;
}

const sensitiveHeaderNames = new Set([
  "authorization",
  "proxy-authorization",
  "x-api-key",
  "api-key",
  "openai-api-key",
  "anthropic-api-key",
  "x-goog-api-key",
  "cookie",
  "set-cookie",
]);
const sensitiveQueryNames = new Set([
  "key",
  "api_key",
  "apikey",
  "access_token",
]);

function captureHeaders(headers) {
  const result = { ...headers };
  for (const key of Object.keys(result)) {
    if (sensitiveHeaderNames.has(key.toLowerCase())) {
      result[key] = "__BENCHFLOW_REDACTED__";
    }
  }
  return result;
}

function capturePath(requestUrl) {
  const parsed = new URL(requestUrl, "http://benchflow.local");
  for (const key of Array.from(parsed.searchParams.keys())) {
    if (sensitiveQueryNames.has(key.toLowerCase())) {
      parsed.searchParams.set(key, "__BENCHFLOW_REDACTED__");
    }
  }
  return `${parsed.pathname}${parsed.search}`;
}

function appendCapture(record) {
  fs.appendFileSync(logPath, JSON.stringify(record) + "\n", { encoding: "utf8" });
}

function bodyB64(chunks) {
  return Buffer.concat(chunks).toString("base64");
}

// Keep in sync with the Python source of truth:
// benchflow/trajectories/gemini_paths.py:normalize_gemini_upstream_path
// (tests/test_gemini_path_normalization.py pins parity across both runtimes).
function normalizeGeminiUpstreamPath(pathWithQuery) {
  const qIdx = pathWithQuery.indexOf("?");
  let path = qIdx === -1 ? pathWithQuery : pathWithQuery.slice(0, qIdx);
  const query = qIdx === -1 ? "" : pathWithQuery.slice(qIdx);
  // already version-prefixed -> leave as-is
  if (path.startsWith("/v1beta/") || path.startsWith("/v1/")) return pathWithQuery;
  // litellm's bogus per-model cache action -> Google's real top-level collection
  path = path.replace(/^\/models\/[^/]+:cachedContents\b/, "/cachedContents");
  // all Google AI Studio resources live under /v1beta
  if (!path.startsWith("/v1beta/")) path = `/v1beta${path}`;
  return `${path}${query}`;
}

function upstreamPath(requestUrl) {
  const requestPath = isGeminiTarget
    ? normalizeGeminiUpstreamPath(requestUrl)
    : requestUrl;
  const basePath = target.pathname.replace(/\/+$/, "");
  if (!basePath) return requestPath;
  return `${basePath}${requestPath.startsWith("/") ? requestPath : `/${requestPath}`}`;
}

function maybeApplyPromptCacheRetention(requestUrl, headers, body) {
  if (!promptCacheRetention) return { headers, body };
  const requestPath = new URL(requestUrl, "http://127.0.0.1").pathname.replace(/\/+$/, "");
  if (!requestPath.endsWith("/responses") && !requestPath.endsWith("/chat/completions")) {
    return { headers, body };
  }
  const contentEncoding = String(headers["content-encoding"] || headers["Content-Encoding"] || "identity").toLowerCase();
  if (contentEncoding !== "identity") return { headers, body };
  try {
    const parsed = body.length > 0 ? JSON.parse(body.toString("utf8")) : {};
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      return { headers, body };
    }
    if (Object.prototype.hasOwnProperty.call(parsed, "prompt_cache_retention")) {
      return { headers, body };
    }
    parsed.prompt_cache_retention = promptCacheRetention;
    const updatedBody = Buffer.from(JSON.stringify(parsed));
    const updatedHeaders = { ...headers };
    delete updatedHeaders["content-encoding"];
    delete updatedHeaders["Content-Encoding"];
    updatedHeaders["content-type"] = updatedHeaders["content-type"] || updatedHeaders["Content-Type"] || "application/json";
    updatedHeaders["content-length"] = updatedBody.length;
    return { headers: updatedHeaders, body: updatedBody };
  } catch (_error) {
    return { headers, body };
  }
}

const server = http.createServer((clientReq, clientRes) => {
  const healthPath = new URL(clientReq.url, "http://127.0.0.1").pathname;
  if ((clientReq.method === "GET" || clientReq.method === "HEAD") && ["/health", "/__benchflow_health"].includes(healthPath.replace(/\/+$/, "") || "/")) {
    const payload = Buffer.from(JSON.stringify({ status: "ok" }));
    clientRes.writeHead(200, {
      "content-type": "application/json",
      "content-length": clientReq.method === "HEAD" ? 0 : payload.length,
    });
    if (clientReq.method !== "HEAD") clientRes.end(payload);
    else clientRes.end();
    return;
  }

  const requestChunks = [];
  clientReq.on("data", (chunk) => requestChunks.push(chunk));
  clientReq.on("end", () => {
    const originalRequestBody = Buffer.concat(requestChunks);
    const startedAt = Date.now();
    const prepared = maybeApplyPromptCacheRetention(
      clientReq.url,
      sanitizeHeaders(clientReq.headers),
      originalRequestBody,
    );
    const requestBody = prepared.body;
    const upstreamHeaders = prepared.headers;
    upstreamHeaders.host = target.host;
    if (requestBody.length > 0) upstreamHeaders["content-length"] = requestBody.length;

    const options = {
      protocol: target.protocol,
      hostname: target.hostname,
      port: target.port || (target.protocol === "https:" ? 443 : 80),
      method: clientReq.method,
      path: upstreamPath(clientReq.url),
      headers: upstreamHeaders,
    };
    const transport = target.protocol === "https:" ? https : http;
    const upstreamReq = transport.request(options, (upstreamRes) => {
      const responseChunks = [];
      clientRes.writeHead(
        upstreamRes.statusCode || 502,
        upstreamRes.statusMessage || "OK",
        responseHeaders(upstreamRes.headers),
      );
      upstreamRes.on("data", (chunk) => {
        responseChunks.push(chunk);
        clientRes.write(chunk);
      });
      upstreamRes.on("end", () => {
        clientRes.end();
        appendCapture({
          session_id: sessionId,
          agent_name: agentName,
          duration_ms: Date.now() - startedAt,
          request: {
            method: clientReq.method,
            path: capturePath(clientReq.url),
            headers: captureHeaders(clientReq.headers),
            body_b64: requestBody.toString("base64"),
          },
          response: {
            status_code: upstreamRes.statusCode || 0,
            headers: captureHeaders(upstreamRes.headers),
            body_b64: bodyB64(responseChunks),
          },
        });
      });
    });
    upstreamReq.on("error", (error) => {
      const payload = Buffer.from(JSON.stringify({ error: String(error.message || error) }));
      clientRes.writeHead(502, {
        "content-type": "application/json",
        "content-length": payload.length,
        "connection": "close",
      });
      clientRes.end(payload);
      appendCapture({
        session_id: sessionId,
        agent_name: agentName,
        duration_ms: Date.now() - startedAt,
        request: {
          method: clientReq.method,
          path: capturePath(clientReq.url),
          headers: captureHeaders(clientReq.headers),
          body_b64: requestBody.toString("base64"),
        },
        response: {
          status_code: 502,
          headers: { "content-type": "application/json" },
          body_b64: payload.toString("base64"),
        },
      });
    });
    if (requestBody.length > 0) upstreamReq.write(requestBody);
    upstreamReq.end();
  });
});

server.listen(0, "127.0.0.1", () => {
  const address = server.address();
  fs.writeFileSync(pidPath, String(process.pid));
  fs.writeFileSync(statePath, JSON.stringify({ port: address.port, pid: process.pid }));
});

process.on("SIGTERM", () => server.close(() => process.exit(0)));
