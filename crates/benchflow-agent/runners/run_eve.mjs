// Runs a user's Eve app once on the task, for BenchFlow.
//
// Written for BenchFlow; the steps follow Harbor's `eve` agent (Apache-2.0,
// harbor/agents/installed/eve.py and eve_runner.mjs): install the app's
// dependencies with the package manager its lockfile names, `eve build`,
// start `eve start` on loopback, and send the instruction through the app's
// own `eve/client`. Unlike Harbor, which reads the events after the turn,
// this runner reports them as they stream.
//
// The app comes from the task: `EVE_PROJECT` (`bench run --env
// EVE_PROJECT=<folder>`, default: the task's working folder) must hold a
// package.json depending on `eve` and an `agent/` folder (or a flat
// `agent.ts` with `instructions.md`). The app chooses its model and its
// sandbox (rooted at /workspace); BenchFlow passes the run's model as
// $BENCHFLOW_MODEL and $HARBOR_MODEL for apps that read it.
//
// Skills BenchFlow delivered ($BENCHFLOW_SKILLS_DIR) are copied into the
// app's skills folder (`agent/skills/`, or `skills/` for a flat app) as
// `benchflow_<name>`, each folder holding a SKILL.md, before the build, as
// Harbor copies them with its `harbor_` prefix.
//
// Other options: `EVE_INSTALL_COMMAND` (replaces the inferred install),
// `EVE_SKIP_BUILD=1`, `EVE_PORT` (default 3024), `EVE_STARTUP_TIMEOUT_MS`
// (default 120000).
//
// Events are JSON lines on stdout in the `benchflow-runner` format;
// everything else (install, build and server logs) goes to stderr.

import { spawn, spawnSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

const EVENTS_FD = 1;
// Anything else that writes to stdout (dependencies, our own logs) goes to stderr.
process.stdout.write = process.stderr.write.bind(process.stderr);

let streamedMessage = false;
function emit(type, fields = {}) {
  const event = { type };
  for (const [key, value] of Object.entries(fields)) {
    if (value !== undefined && value !== null) event[key] = value;
  }
  writeSync(EVENTS_FD, `${JSON.stringify(event)}\n`);
}

class SetupError extends Error {
  constructor(message, fix) {
    super(message);
    this.fix = fix;
  }
}

const FIX =
  "add an Eve app to the task (a package.json depending on `eve` and an agent/ folder) and pass " +
  "--env EVE_PROJECT=<its folder in the sandbox> when it is not the working folder";

function option(name, fallback) {
  const value = (process.env[name] ?? "").trim();
  return value === "" ? fallback : value;
}

function kindOf(text) {
  const words = String(text).toLowerCase();
  if (/quota|credit balance|insufficient_quota|usage limit/.test(words)) return "usage_limit";
  if (/\b401\b|\b403\b|unauthori[sz]ed|rejected.{0,40}api key|invalid api key|api key not valid|authentication|invalid token|incorrect api key/.test(words))
    return "login";
  if (/\b429\b|rate.?limit|\b50[0234]\b|overloaded|unavailable|timed? ?out|econnrefused|enotfound/.test(words)) return "provider";
  if (/model.{0,40}(not found|does not exist|unknown|not supported)/.test(words)) return "model";
  return "agent";
}

function checkProject(dir) {
  const pkgPath = join(dir, "package.json");
  if (!existsSync(pkgPath)) throw new SetupError(`eve runs your Eve app, and ${dir} has no package.json`, FIX);
  let pkg;
  try {
    pkg = JSON.parse(readFileSync(pkgPath, "utf8"));
  } catch (error) {
    throw new SetupError(`${pkgPath} is not JSON: ${error.message}`, "fix the app's package.json");
  }
  const deps = { ...pkg.dependencies, ...pkg.devDependencies };
  if (!deps.eve) throw new SetupError(`${pkgPath} does not depend on eve`, FIX);
  const flat = existsSync(join(dir, "agent.ts")) && (existsSync(join(dir, "instructions.md")) || existsSync(join(dir, "instructions.ts")));
  if (!existsSync(join(dir, "agent")) && !flat) {
    throw new SetupError(`${dir} has no agent/ folder (nor a flat agent.ts with instructions.md)`, FIX);
  }
  return pkg;
}

// Copy each delivered skill into the app's skills folder; the names placed.
function placeSkills(dir) {
  const from = option("BENCHFLOW_SKILLS_DIR");
  if (!from || !existsSync(from)) return [];
  const into = join(existsSync(join(dir, "agent")) ? join(dir, "agent") : dir, "skills");
  const placed = [];
  for (const entry of readdirSync(from, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith(".") || !existsSync(join(from, entry.name, "SKILL.md"))) continue;
    const to = join(into, `benchflow_${entry.name}`);
    try {
      mkdirSync(into, { recursive: true });
      rmSync(to, { recursive: true, force: true });
      cpSync(join(from, entry.name), to, { recursive: true });
    } catch (error) {
      // The app's folder, not the agent: a prebuilt app the agent user
      // cannot write cannot take the run's skills.
      throw new SetupError(
        `could not copy skill ${entry.name} into ${into}: ${error.message}`,
        "make the Eve app's folder writable by the agent user, or run without skills",
      );
    }
    placed.push(to);
  }
  if (placed.length) process.stderr.write(`eve: skills: ${placed.join(", ")}\n`);
  return placed;
}

function installCommand(dir, pkg) {
  const custom = option("EVE_INSTALL_COMMAND");
  if (custom) return custom;
  const manager = typeof pkg.packageManager === "string" ? pkg.packageManager.split("@")[0] : "";
  const has = (name) => existsSync(join(dir, name));
  if (manager === "pnpm" || has("pnpm-lock.yaml")) return "corepack enable --install-directory \"$HOME/.benchflow/bin\" && pnpm install --frozen-lockfile";
  if (manager === "yarn" || has("yarn.lock")) return "corepack enable --install-directory \"$HOME/.benchflow/bin\" && yarn install --frozen-lockfile";
  if (manager === "bun" || has("bun.lock") || has("bun.lockb")) return "npm install -g --prefix \"$HOME/.benchflow\" bun && bun install --frozen-lockfile";
  return has("package-lock.json") ? "npm ci" : "npm install";
}

function sh(command, cwd, what) {
  process.stderr.write(`eve: ${what}: ${command}\n`);
  const env = { ...process.env, PATH: `${process.env.HOME}/.benchflow/bin:${process.env.PATH}` };
  const done = spawnSync("sh", ["-c", command], { cwd, env, stdio: ["ignore", 2, 2] });
  if (done.status !== 0) throw new Error(`${what} failed (exit ${done.status ?? done.signal}): ${command}`);
}

async function waitForHealth(server, url, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let last = "no answer yet";
  while (Date.now() < deadline) {
    if (server.exitCode !== null) throw new Error(`eve start exited (code ${server.exitCode}) before it was ready`);
    try {
      const response = await fetch(`${url}/eve/v1/health`);
      if (response.ok) return;
      last = `health returned ${response.status}`;
    } catch (error) {
      last = error.message;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error(`eve start was not ready after ${timeoutMs} ms: ${last}`);
}

function textOf(value) {
  if (value === undefined || value === null) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

async function run() {
  const instructionFile = process.argv[2] || process.env.BENCHFLOW_INSTRUCTION_FILE;
  if (!instructionFile) throw new SetupError("BENCHFLOW_INSTRUCTION_FILE is not set", "run the agent with `bench run`");
  const instruction = readFileSync(instructionFile, "utf8");
  const dir = resolve(option("EVE_PROJECT", process.cwd()));
  process.stderr.write(`eve: Node.js ${process.version}, PATH=${process.env.PATH}\n`);
  const pkg = checkProject(dir);
  if (process.env.BENCHFLOW_MODEL) process.env.HARBOR_MODEL ??= process.env.BENCHFLOW_MODEL;
  placeSkills(dir);

  const eveBin = join(dir, "node_modules", ".bin", "eve");
  if (!existsSync(eveBin)) sh(installCommand(dir, pkg), dir, "installing the app's dependencies");
  if (!existsSync(eveBin)) throw new SetupError(`installing ${dir} left no node_modules/.bin/eve`, "make the app depend on eve");
  if (option("EVE_SKIP_BUILD") !== "1") sh(`"${eveBin}" build`, dir, "building the app");

  const require = createRequire(join(dir, "package.json"));
  const version = (() => {
    try {
      return JSON.parse(readFileSync(require.resolve("eve/package.json"), "utf8")).version;
    } catch {
      return undefined;
    }
  })();
  const port = Number(option("EVE_PORT", "3024"));
  const url = `http://127.0.0.1:${port}`;
  // The server listens on loopback only. Eve's default route auth admits
  // loopback callers only on a development server (`localDev()`, keyed on
  // EVE_DEV=1), so it runs as one; an app's own channel auth still applies.
  const server = spawn(eveBin, ["start", "--host", "127.0.0.1", "--port", String(port)], {
    cwd: dir,
    env: { ...process.env, EVE_DEV: process.env.EVE_DEV ?? "1" },
    stdio: ["ignore", 2, 2],
  });
  try {
    await waitForHealth(server, url, Number(option("EVE_STARTUP_TIMEOUT_MS", "120000")));
    const { Client } = await import(pathToFileURL(require.resolve("eve/client")).href);
    emit("start", { agent: "eve", framework: "eve", version, model: process.env.BENCHFLOW_MODEL || undefined });
    const client = new Client({ host: url });
    const { response } = await client.sessions.create({ message: instruction });
    let answer = "";
    const usage = { input_tokens: 0, output_tokens: 0, cache_read_tokens: 0, cache_write_tokens: 0, cost_usd: 0 };
    let failure;
    for await (const event of response) {
      const data = event.data ?? {};
      switch (event.type) {
        case "message.appended":
          if (data.messageDelta) {
            streamedMessage = true;
            emit("message", { text: data.messageDelta });
          }
          break;
        case "message.completed":
          if (data.message) answer = data.message;
          break;
        case "reasoning.appended":
          if (data.reasoningDelta) emit("thought", { text: data.reasoningDelta });
          break;
        case "actions.requested":
          for (const action of data.actions ?? []) {
            emit("tool_call", { id: action.callId, name: action.toolName ?? action.name ?? action.kind ?? "action", input: action.input });
          }
          break;
        case "action.result": {
          const result = data.result ?? {};
          const failed = Boolean(result.isError) || data.status === "error" || data.status === "failed" || Boolean(data.error);
          emit("tool_result", { id: result.callId, output: textOf(data.error?.message ?? result.output ?? result), failed });
          break;
        }
        case "step.completed":
          if (data.usage) {
            usage.input_tokens += data.usage.inputTokens ?? 0;
            usage.output_tokens += data.usage.outputTokens ?? 0;
            usage.cache_read_tokens += data.usage.cacheReadTokens ?? 0;
            usage.cache_write_tokens += data.usage.cacheWriteTokens ?? 0;
            usage.cost_usd += data.usage.costUsd ?? 0;
            const totals = Object.fromEntries(Object.entries(usage).filter(([, v]) => v));
            emit("usage", totals);
          }
          break;
        case "step.failed":
          emit("error", { message: `${data.code ?? "step.failed"}: ${data.message ?? ""}`, kind: kindOf(`${data.code} ${data.message}`), fatal: false });
          break;
        case "turn.failed":
        case "session.failed":
          failure ??= { message: `${data.code ?? event.type}: ${data.message ?? textOf(data.details)}`, kind: kindOf(`${data.code} ${data.message} ${textOf(data.details)}`) };
          break;
        case "turn.waiting":
          if (data.on === "input") failure ??= { message: "the Eve app asked for human input, which a benchmark run cannot give", kind: "agent" };
          break;
        default:
          break;
      }
    }
    if (failure) {
      emit("error", failure);
      return 1;
    }
    emit("done", { answer: streamedMessage ? undefined : answer || undefined });
    return 0;
  } finally {
    if (server.exitCode === null) {
      server.kill("SIGTERM");
      await Promise.race([new Promise((r) => server.once("exit", r)), new Promise((r) => setTimeout(r, 5000))]);
      if (server.exitCode === null) server.kill("SIGKILL");
    }
  }
}

try {
  process.exitCode = await run();
} catch (error) {
  if (error instanceof SetupError) {
    emit("error", { message: error.message, kind: "setup", fix: error.fix });
    process.exitCode = 2;
  } else {
    process.stderr.write(`${error?.stack ?? error}\n`);
    const message = `${error?.name ?? "Error"}: ${error?.message ?? String(error)}`;
    emit("error", { message, kind: kindOf(message) });
    process.exitCode = 1;
  }
}
