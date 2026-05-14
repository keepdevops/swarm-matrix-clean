import { useEffect, useRef, useState } from "react";
import { Agent, listAgents, listBackends, streamGenerate } from "./api";
import { Editor, Language } from "./Editor";

type Phase = "idle" | "loading" | "streaming" | "success" | "error";

export default function App() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [backends, setBackends] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<string>("");
  const [backendOverride, setBackendOverride] = useState<string>("");
  const [language, setLanguage] = useState<Language>("cpp");
  const [temperature, setTemperature] = useState<number>(0.2);
  const [maxTokens, setMaxTokens] = useState<number>(512);

  const [prompt, setPrompt] = useState<string>(
    "// fast inverse square root\nfloat Q_rsqrt(float number) {\n",
  );
  const [output, setOutput] = useState<string>("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [message, setMessage] = useState<string>("ready");
  const [activeBackend, setActiveBackend] = useState<string>("");

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [a, b] = await Promise.all([listAgents(), listBackends()]);
        if (cancelled) return;
        setAgents(a);
        setBackends(b);
        if (a.length > 0) setSelectedAgent(a[0].file);
      } catch (err) {
        setPhase("error");
        setMessage(`load failed: ${(err as Error).message}`);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  async function run() {
    if (!selectedAgent) {
      setPhase("error");
      setMessage("no agent selected");
      return;
    }
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    setOutput("");
    setActiveBackend("");
    setPhase("loading");
    setMessage("acquiring backend…");

    try {
      const stream = streamGenerate(
        {
          agent: selectedAgent,
          prompt,
          temperature,
          max_tokens: maxTokens,
          backend_override: backendOverride || undefined,
        },
        ctrl.signal,
      );
      for await (const evt of stream) {
        if (evt.kind === "ready") {
          setActiveBackend(evt.backend);
          setPhase("streaming");
          setMessage(`streaming from ${evt.backend}`);
        } else if (evt.kind === "token") {
          setOutput((s) => s + evt.content);
        } else if (evt.kind === "done") {
          setPhase("success");
          setMessage(`done (${evt.finish_reason})`);
          return;
        } else if (evt.kind === "error") {
          setPhase("error");
          setMessage(evt.error);
          return;
        }
      }
      setPhase("success");
      setMessage("stream ended");
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        setPhase("idle");
        setMessage("cancelled");
        return;
      }
      setPhase("error");
      setMessage((err as Error).message);
    }
  }

  function cancel() {
    abortRef.current?.abort();
  }

  const isRunning = phase === "loading" || phase === "streaming";

  return (
    <div className="app">
      <header className="toolbar">
        <label htmlFor="agent">Agent</label>
        <select
          id="agent"
          value={selectedAgent}
          onChange={(e) => setSelectedAgent(e.target.value)}
        >
          {agents.map((a) => (
            <option key={a.file} value={a.file}>
              {a.name ?? a.file} ({a.backend_target})
            </option>
          ))}
        </select>

        <label htmlFor="backend">Override</label>
        <select
          id="backend"
          value={backendOverride}
          onChange={(e) => setBackendOverride(e.target.value)}
        >
          <option value="">(from agent)</option>
          {backends.map((b) => (<option key={b} value={b}>{b}</option>))}
        </select>

        <label htmlFor="lang">Lang</label>
        <select
          id="lang"
          value={language}
          onChange={(e) => setLanguage(e.target.value as Language)}
        >
          <option value="cpp">C/C++</option>
          <option value="python">Python</option>
          <option value="javascript">JS/TS</option>
          <option value="plaintext">Plain</option>
        </select>

        <label htmlFor="temp">Temp</label>
        <input
          id="temp" type="number" step="0.05" min={0} max={2}
          value={temperature}
          onChange={(e) => setTemperature(Number(e.target.value))}
          style={{ width: 64 }}
        />

        <label htmlFor="max">Max tok</label>
        <input
          id="max" type="number" step={16} min={1} max={8192}
          value={maxTokens}
          onChange={(e) => setMaxTokens(Number(e.target.value))}
          style={{ width: 80 }}
        />

        <span className="spacer" />

        {isRunning ? (
          <button className="btn secondary" onClick={cancel}>Cancel</button>
        ) : (
          <button className="btn" onClick={run} disabled={!selectedAgent}>
            Run ⌘↩
          </button>
        )}
      </header>

      <main className="panes">
        <section className="pane">
          <div className="pane-header">prompt · {language}</div>
          <Editor value={prompt} onChange={setPrompt} language={language} />
        </section>
        <section className="pane">
          <div className="pane-header">
            output {activeBackend && <span>· {activeBackend}</span>}
          </div>
          <pre className={`output${output ? "" : " empty"}`}>
            {output || "(no output yet — click Run)"}
          </pre>
        </section>
      </main>

      <footer className={`status ${phase}`}>
        <span className={`dot${isRunning ? " pulse" : ""}`} />
        <span>{message}</span>
      </footer>
    </div>
  );
}
