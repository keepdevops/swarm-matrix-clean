// API client for the FastAPI server. SSE parsing done manually because the
// browser's EventSource forces GET — we need POST.

export type Agent = {
  file: string;
  agent_id?: string;
  name?: string;
  backend_target?: string;
  model_path?: string;
};

export type GenerateBody = {
  agent: string;
  prompt: string;
  temperature?: number;
  max_tokens?: number;
  backend_override?: string;
};

export type StreamEvent =
  | { kind: "ready"; backend: string }
  | { kind: "token"; content: string }
  | { kind: "done"; finish_reason: string }
  | { kind: "error"; error: string };

export async function listBackends(): Promise<string[]> {
  const r = await fetch("/api/backends");
  if (!r.ok) throw new Error(`backends fetch ${r.status}`);
  const j = (await r.json()) as { backends: string[] };
  return j.backends;
}

export async function listAgents(): Promise<Agent[]> {
  const r = await fetch("/api/agents");
  if (!r.ok) throw new Error(`agents fetch ${r.status}`);
  const j = (await r.json()) as { agents: Agent[] };
  return j.agents;
}

export async function* streamGenerate(
  body: GenerateBody,
  signal: AbortSignal,
): AsyncGenerator<StreamEvent, void, void> {
  const r = await fetch("/api/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    const text = await r.text().catch(() => "");
    throw new Error(`generate ${r.status}: ${text.slice(0, 200)}`);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) return;
    buf += decoder.decode(value, { stream: true });

    // SSE frames are separated by a blank line.
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const evt = parseSseFrame(frame);
      if (evt) yield evt;
    }
  }
}

function parseSseFrame(frame: string): StreamEvent | null {
  let event = "message";
  let data = "";
  for (const raw of frame.split("\n")) {
    if (raw.startsWith("event:")) event = raw.slice(6).trim();
    else if (raw.startsWith("data:")) data += raw.slice(5).trim();
  }
  if (!data) return null;
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(data);
  } catch {
    console.error("malformed SSE data:", data);
    return null;
  }
  switch (event) {
    case "ready":
      return { kind: "ready", backend: String(payload.backend ?? "?") };
    case "token":
      return { kind: "token", content: String(payload.content ?? "") };
    case "done":
      return { kind: "done", finish_reason: String(payload.finish_reason ?? "stop") };
    case "error":
      return { kind: "error", error: String(payload.error ?? "unknown error") };
    default:
      return null;
  }
}
