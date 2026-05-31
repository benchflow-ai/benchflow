"""Sandbox-local provider usage proxy runtime.

The host-side :class:`benchflow.trajectories.proxy.TrajectoryProxy` works when
the agent can route back to the host. Remote sandboxes such as Daytona cannot,
so this module starts a tiny byte-forwarding proxy inside the same sandbox
network namespace as the agent and imports its raw captures back into the host
trajectory model during cleanup.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from benchflow.agents.registry import _NODE_INSTALL
from benchflow.trajectories.proxy import exchange_from_raw_capture
from benchflow.trajectories.types import Trajectory

logger = logging.getLogger(__name__)

_RUNTIME_ROOT = "/tmp/benchflow-usage-proxy"
_NODE_PROXY_SOURCE = r"""#!/usr/bin/env node
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

function upstreamPath(requestUrl) {
  const basePath = target.pathname.replace(/\/+$/, "");
  if (!basePath) return requestUrl;
  return `${basePath}${requestUrl.startsWith("/") ? requestUrl : `/${requestUrl}`}`;
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
"""

_NODE_LAUNCHER_SOURCE = r"""
const fs = require("fs");
const { spawn } = require("child_process");

const config = JSON.parse(process.env.BENCHFLOW_USAGE_PROXY_CONFIG || "{}");
const stdout = fs.openSync(config.stdout, "a");
const stderr = fs.openSync(config.stderr, "a");
const child = spawn(config.node, [config.script], {
  detached: true,
  stdio: ["ignore", stdout, stderr],
  env: { ...process.env, ...config.env },
});
child.unref();
console.log(child.pid);
"""


class SandboxUsageProxy:
    """Long-lived proxy process running in the agent sandbox."""

    def __init__(
        self,
        *,
        sandbox: Any,
        target: str,
        session_id: str,
        agent_name: str,
        prompt_cache_retention: str | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.target = target.rstrip("/")
        self.session_id = session_id
        self.agent_name = agent_name
        self.prompt_cache_retention = prompt_cache_retention
        self.trajectory = Trajectory(session_id=session_id, agent_name=agent_name)
        self._token = uuid4().hex[:16]
        self._runtime_dir = f"{_RUNTIME_ROOT}/{self._token}"
        self._script_path = f"{self._runtime_dir}/proxy.js"
        self._state_path = f"{self._runtime_dir}/state.json"
        self._log_path = f"{self._runtime_dir}/captures.jsonl"
        self._pid_path = f"{self._runtime_dir}/proxy.pid"
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        if self._base_url is None:
            raise RuntimeError("sandbox usage proxy has not started")
        return self._base_url

    async def start(self) -> None:
        await self._upload_proxy_script()
        node = await self._ensure_node()
        stdout_path = f"{_RUNTIME_ROOT}/{self._token}/stdout.log"
        stderr_path = f"{_RUNTIME_ROOT}/{self._token}/stderr.log"
        launcher_config = {
            "node": node,
            "script": self._script_path,
            "stdout": stdout_path,
            "stderr": stderr_path,
            "env": {
                "BENCHFLOW_USAGE_PROXY_TARGET": self.target,
                "BENCHFLOW_USAGE_PROXY_STATE_PATH": self._state_path,
                "BENCHFLOW_USAGE_PROXY_LOG_PATH": self._log_path,
                "BENCHFLOW_USAGE_PROXY_PID_PATH": self._pid_path,
                "BENCHFLOW_USAGE_PROXY_SESSION_ID": self.session_id,
                "BENCHFLOW_USAGE_PROXY_AGENT_NAME": self.agent_name,
                "BENCHFLOW_USAGE_PROXY_PROMPT_CACHE_RETENTION": (
                    self.prompt_cache_retention or ""
                ),
            },
        }
        command = " ".join(
            [
                "mkdir",
                "-p",
                shlex.quote(str(Path(self._script_path).parent)),
                "&&",
                "rm",
                "-f",
                shlex.quote(self._state_path),
                shlex.quote(self._log_path),
                shlex.quote(self._pid_path),
                "&&",
                f"BENCHFLOW_USAGE_PROXY_CONFIG={shlex.quote(json.dumps(launcher_config))}",
                shlex.quote(node),
                "-e",
                shlex.quote(_NODE_LAUNCHER_SOURCE),
            ]
        )
        result = await self.sandbox.exec(command, timeout_sec=15)
        if result.return_code != 0:
            raise RuntimeError(_exec_details("start sandbox usage proxy", result))
        state = await self._wait_for_state()
        self._base_url = f"http://127.0.0.1:{state['port']}"
        logger.info("Sandbox usage telemetry proxy listening on %s", self._base_url)

    async def is_running(self) -> bool:
        result = await self.sandbox.exec(
            (
                f"if [ -s {shlex.quote(self._pid_path)} ] && "
                f"kill -0 $(cat {shlex.quote(self._pid_path)}) 2>/dev/null; "
                "then echo yes; else echo no; fi"
            ),
            timeout_sec=5,
        )
        return result.return_code == 0 and (result.stdout or "").strip() == "yes"

    async def stop(self) -> None:
        try:
            await self._load_captures()
        except Exception as exc:
            logger.warning("Could not import sandbox usage captures: %s", exc)
        finally:
            await self._terminate()
            await self._cleanup_runtime_dir()

    async def _terminate(self) -> None:
        kill_cmd = (
            f"if [ -s {shlex.quote(self._pid_path)} ]; then "
            f"kill -TERM $(cat {shlex.quote(self._pid_path)}) 2>/dev/null || true; "
            "fi"
        )
        with contextlib.suppress(Exception):
            await self.sandbox.exec(kill_cmd, timeout_sec=10)

    async def _cleanup_runtime_dir(self) -> None:
        with contextlib.suppress(Exception):
            await self.sandbox.exec(
                f"rm -rf {shlex.quote(self._runtime_dir)}",
                timeout_sec=10,
            )

    async def _upload_proxy_script(self) -> None:
        parent = shlex.quote(str(Path(self._script_path).parent))
        result = await self.sandbox.exec(f"mkdir -p {parent}", timeout_sec=15)
        if result.return_code != 0:
            raise RuntimeError(_exec_details("prepare sandbox usage proxy dir", result))

        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as tmp:
            tmp.write(_NODE_PROXY_SOURCE)
            tmp_path = Path(tmp.name)
        try:
            await self.sandbox.upload_file(tmp_path, self._script_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _ensure_node(self) -> str:
        node_probe = (
            "if [ -x /opt/benchflow/node/bin/node ]; then "
            "echo /opt/benchflow/node/bin/node; "
            "elif command -v node >/dev/null 2>&1; then command -v node; "
            "else echo ''; fi"
        )
        result = await self.sandbox.exec(node_probe, timeout_sec=10)
        node = (result.stdout or "").strip().splitlines()[-1:] or [""]
        if node[0]:
            return node[0]

        install = await self.sandbox.exec(_NODE_INSTALL, timeout_sec=300)
        if install.return_code != 0:
            raise RuntimeError(_exec_details("install Node for usage proxy", install))
        result = await self.sandbox.exec(node_probe, timeout_sec=10)
        node = (result.stdout or "").strip().splitlines()[-1:] or [""]
        if not node[0]:
            raise RuntimeError("Node.js was not available after usage proxy bootstrap")
        return node[0]

    async def _wait_for_state(self) -> dict[str, Any]:
        last_output = ""
        for _ in range(50):
            result = await self.sandbox.exec(
                f"cat {shlex.quote(self._state_path)} 2>/dev/null || true",
                timeout_sec=5,
            )
            last_output = (result.stdout or "").strip()
            if last_output:
                try:
                    state = json.loads(last_output)
                except (json.JSONDecodeError, ValueError):
                    await asyncio.sleep(0.2)
                    continue
                if int(state.get("port") or 0) > 0:
                    return state
            await asyncio.sleep(0.2)
        stderr = await self.sandbox.exec(
            f"cat {shlex.quote(f'{_RUNTIME_ROOT}/{self._token}/stderr.log')} "
            "2>/dev/null || true",
            timeout_sec=5,
        )
        raise RuntimeError(
            "sandbox usage proxy did not publish its state"
            f": {last_output or (stderr.stdout or '').strip()}"
        )

    async def _load_captures(self) -> None:
        capture_text = await self._read_capture_log()
        trajectory = Trajectory(session_id=self.session_id, agent_name=self.agent_name)
        for line in capture_text.splitlines():
            if not line.strip():
                continue
            try:
                trajectory.exchanges.append(exchange_from_raw_capture(json.loads(line)))
            except Exception as exc:
                logger.warning("Skipping malformed sandbox usage capture: %s", exc)
        self.trajectory = trajectory

    async def _read_capture_log(self) -> str:
        download_file = getattr(self.sandbox, "download_file", None)
        if download_file is not None:
            with tempfile.NamedTemporaryFile("r", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                await download_file(self._log_path, tmp_path)
                return tmp_path.read_text()
            except Exception as exc:
                logger.debug("Sandbox usage capture download failed: %s", exc)
            finally:
                tmp_path.unlink(missing_ok=True)

        result = await self.sandbox.exec(
            f"cat {shlex.quote(self._log_path)} 2>/dev/null || true",
            timeout_sec=15,
        )
        if result.return_code != 0:
            logger.warning("Could not read sandbox usage captures: %s", result.stderr)
            return ""
        return result.stdout or ""


def _exec_details(label: str, result: Any) -> str:
    stdout = (getattr(result, "stdout", "") or "").strip()
    stderr = (getattr(result, "stderr", "") or "").strip()
    details = [f"{label} failed with exit code {getattr(result, 'return_code', '?')}"]
    if stdout:
        details.append(f"stdout: {stdout[:1000]}")
    if stderr:
        details.append(f"stderr: {stderr[:1000]}")
    return "; ".join(details)
