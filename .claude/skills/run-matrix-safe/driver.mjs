#!/usr/bin/env node
// Agent-facing driver for the matrix-safe stack (FastAPI :9000 + Vite :9001).
// No dependencies — Node 18+ global fetch only. Run from anywhere.
//
//   node .claude/skills/run-matrix-safe/driver.mjs smoke
//   node .claude/skills/run-matrix-safe/driver.mjs generate local-mlx.json --max-tokens 32
//
// Exit code is 0 only if every requested step succeeded.

const BACKEND = process.env.MATRIX_BACKEND_URL ?? "http://127.0.0.1:9000";
const FRONTEND = process.env.MATRIX_FRONTEND_URL ?? "http://localhost:9001";

// llama_cpp_binary spawns llama-server and loads a multi-GB GGUF on first
// acquire; 30s is not enough. mlx warms in ~5s.
const DEFAULT_TIMEOUT_MS = Number(process.env.MATRIX_TIMEOUT_MS ?? 300_000);

const log = (...a) => console.log(...a);
const fail = (msg, err) => {
  console.error(`✗ ${msg}${err ? `: ${err.message ?? err}` : ""}`);
  if (err?.stack && process.env.MATRIX_DEBUG) console.error(err.stack);
  return false;
};

async function getJSON(path, timeoutMs = 10_000) {
  const res = await fetch(`${BACKEND}${path}`, {
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!res.ok) throw new Error(`GET ${path} → HTTP ${res.status}`);
  return res.json();
}

// ---------------------------------------------------------------- commands

async function health() {
  let ok = true;
  try {
    const body = await getJSON("/api/health", 5_000);
    if (body.status !== "ok") throw new Error(`unexpected body ${JSON.stringify(body)}`);
    log(`✓ backend  ${BACKEND}/api/health → ok`);
  } catch (err) {
    ok = fail(`backend ${BACKEND} unreachable`, err) && ok;
    ok = false;
  }
  try {
    // Probe `localhost`, not 127.0.0.1 — Vite binds IPv6 [::1] by default and
    // an IPv4-only probe never connects even though the dev server is serving.
    const res = await fetch(`${FRONTEND}/`, { signal: AbortSignal.timeout(5_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    log(`✓ frontend ${FRONTEND}/ → ${res.status}`);
  } catch (err) {
    fail(`frontend ${FRONTEND} unreachable`, err);
    ok = false;
  }
  return ok;
}

async function agents() {
  const { agents } = await getJSON("/api/agents");
  for (const a of agents) log(`  ${a.file.padEnd(24)} ${String(a.backend_target).padEnd(18)} ${a.model_path}`);
  log(`✓ ${agents.length} agent config(s)`);
  return agents;
}

async function backends() {
  const { backends } = await getJSON("/api/backends");
  log(`✓ backends: ${backends.join(", ")}`);
  return backends;
}

/**
 * Stream POST /api/generate and print tokens as they arrive.
 * Returns {backend, text, finish} or throws.
 */
async function generate(agent, opts = {}) {
  const body = {
    agent,
    prompt: opts.prompt ?? "// fast inverse square root\nfloat Q_rsqrt(float number) {\n",
  };
  if (opts.maxTokens != null) body.max_tokens = opts.maxTokens;
  if (opts.temperature != null) body.temperature = opts.temperature;
  if (opts.backend) body.backend_override = opts.backend;

  const t0 = Date.now();
  const res = await fetch(`${BACKEND}/api/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS),
  });
  // A bad agent name / unloadable config is a 404 or 400 with a JSON detail —
  // it never reaches the SSE stream, so surface it here.
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`POST /api/generate → HTTP ${res.status} ${detail}`);
  }

  let usedBackend = "?";
  let text = "";
  let finish = null;
  let streamError = null;

  for await (const evt of sseEvents(res.body)) {
    if (evt.event === "ready") {
      usedBackend = evt.data.backend;
      log(`  ready  backend=${usedBackend} (+${Date.now() - t0}ms)`);
    } else if (evt.event === "token") {
      text += evt.data.content;
      if (!opts.quiet) process.stdout.write(evt.data.content);
    } else if (evt.event === "done") {
      finish = evt.data.finish_reason;
    } else if (evt.event === "error") {
      streamError = evt.data.error;
    }
  }
  if (!opts.quiet) process.stdout.write("\n");
  // The server reports generation failures as an SSE `error` event on a 200
  // response — treat it as a hard failure, not a quiet empty result.
  if (streamError) throw new Error(`stream error: ${streamError}`);
  if (!text) throw new Error("stream produced zero tokens");

  log(`✓ generate ${agent} → ${text.length} chars, finish=${finish}, ${Date.now() - t0}ms`);
  return { backend: usedBackend, text, finish };
}

/** OpenAI-compatible non-streaming call. `model` is an agent_id, not a filename. */
async function chat(model, opts = {}) {
  const res = await fetch(`${BACKEND}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: opts.prompt ?? "Say OK and nothing else." }],
      max_tokens: opts.maxTokens ?? 16,
      stream: false,
    }),
    signal: AbortSignal.timeout(opts.timeoutMs ?? DEFAULT_TIMEOUT_MS),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`POST /v1/chat/completions → HTTP ${res.status} ${detail}`);
  }
  const body = await res.json();
  const content = body.choices?.[0]?.message?.content;
  if (typeof content !== "string" || content === "") {
    throw new Error(`no content in response: ${JSON.stringify(body).slice(0, 300)}`);
  }
  if (!opts.quiet) log(content);
  log(`✓ chat ${model} → ${content.length} chars, finish=${body.choices[0].finish_reason}`);
  return content;
}

// ------------------------------------------------------------------- SSE

/** Parse a `text/event-stream` body into {event, data} objects. */
async function* sseEvents(stream) {
  const decoder = new TextDecoder();
  let buf = "";
  for await (const chunk of stream) {
    buf += decoder.decode(chunk, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const parsed = parseSSEBlock(raw);
      if (parsed) yield parsed;
    }
  }
}

function parseSSEBlock(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (dataLines.length === 0) return null;
  const payload = dataLines.join("\n");
  if (payload === "[DONE]") return { event: "done", data: {} };
  try {
    return { event, data: JSON.parse(payload) };
  } catch (err) {
    console.error(`✗ unparseable SSE data: ${payload.slice(0, 200)} (${err.message})`);
    return null;
  }
}

// ------------------------------------------------------------------ smoke

/**
 * Full end-to-end pass. Defaults to the mlx agent because it warms in seconds;
 * override with SMOKE_AGENT / SMOKE_MODEL for a GGUF path (much slower).
 */
async function smoke() {
  const agent = process.env.SMOKE_AGENT ?? "local-mlx.json";
  const model = process.env.SMOKE_MODEL ?? "local_mlx_agent";
  if (!(await health())) throw new Error("health check failed — is the stack up?");
  await backends();
  const list = await agents();
  if (!list.some((a) => a.file === agent)) {
    throw new Error(`smoke agent ${agent} not in /api/agents — set SMOKE_AGENT`);
  }
  await generate(agent, { maxTokens: 32, quiet: true });
  await chat(model, { maxTokens: 16, quiet: true });
  log("\n✓ smoke passed");
}

// -------------------------------------------------------------------- CLI

function flag(argv, name, fallback) {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
}

const [cmd, ...rest] = process.argv.slice(2);
const opts = {
  prompt: flag(rest, "prompt"),
  maxTokens: flag(rest, "max-tokens") && Number(flag(rest, "max-tokens")),
  temperature: flag(rest, "temperature") && Number(flag(rest, "temperature")),
  backend: flag(rest, "backend"),
};
const positional = rest.filter((a, i) => !a.startsWith("--") && !rest[i - 1]?.startsWith("--"));

const commands = {
  health,
  agents,
  backends,
  smoke,
  generate: () => generate(positional[0] ?? "local-mlx.json", opts),
  chat: () => chat(positional[0] ?? "local_mlx_agent", opts),
};

if (!cmd || !(cmd in commands)) {
  console.error(`usage: driver.mjs <${Object.keys(commands).join("|")}> [args]

  health                      probe :9000 /api/health and :9001
  backends                    list registered backend keys
  agents                      list agent configs
  generate [file.json]        stream SSE from POST /api/generate
  chat [agent_id]             non-streaming POST /v1/chat/completions
  smoke                       all of the above; exit 1 on any failure

  flags: --prompt TEXT  --max-tokens N  --temperature F  --backend KEY
  env:   MATRIX_BACKEND_URL MATRIX_FRONTEND_URL MATRIX_TIMEOUT_MS
         SMOKE_AGENT SMOKE_MODEL MATRIX_DEBUG`);
  process.exit(2);
}

try {
  const result = await commands[cmd]();
  process.exit(result === false ? 1 : 0);
} catch (err) {
  fail(`${cmd} failed`, err);
  process.exit(1);
}
