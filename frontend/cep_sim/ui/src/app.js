import { h, Component, htm, useState, createRoot } from "./deps.js";
import { SetupPanel }   from "./components/setup-panel.js";
import { ConfigPanel }  from "./components/config-panel.js";
import { ResultsPanel } from "./components/results-panel.js";
import { api } from "./api.js";
const html = htm.bind(h);

// ── Error boundary — shows crash details instead of blank screen ──────────
class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(err) {
    return { error: err };
  }
  componentDidCatch(err, info) {
    console.error("[CEP Sim] Render error:", err, info);
  }
  render() {
    if (this.state.error) {
      return h("div", { style: { padding: "2rem", color: "#f87171", fontFamily: "monospace", background: "#0f172a", minHeight: "100vh" } },
        h("h2", null, "Render error — check browser console"),
        h("pre", { style: { whiteSpace: "pre-wrap", fontSize: "12px", marginTop: "1rem", color: "#fca5a5" } },
          this.state.error.message + "\n\n" + (this.state.error.stack || "")
        ),
        h("button", {
          onClick: () => this.setState({ error: null }),
          style: { marginTop: "1rem", padding: "0.5rem 1rem", background: "#1e293b", border: "1px solid #334155", color: "#94a3b8", cursor: "pointer", borderRadius: "0.5rem" }
        }, "Dismiss")
      );
    }
    return this.props.children;
  }
}

// ── Header ────────────────────────────────────────────────────────────────
function Header({ session, onReset }) {
  if (!session) return null;
  const count = (session.respondent_count || 0).toLocaleString();
  return html`
    <header class="flex items-center justify-between px-6 py-3 border-b border-slate-800 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
      <div class="flex items-center gap-3">
        <div class="w-7 h-7 rounded-lg bg-brand-500 flex items-center justify-center text-slate-900 font-bold text-xs">C</div>
        <span class="font-semibold tracking-tight">CEP Simulator</span>
        <span class="px-2 py-0.5 rounded-full bg-slate-800 border border-slate-700 text-xs text-slate-400">
          ${session.country || "—"} · ${count} respondents · ${session.brand_count || 0} brands · ${session.cep_count || 0} CEPs
        </span>
      </div>
      <button onClick=${onReset} class="text-xs text-slate-500 hover:text-slate-300 transition-colors">
        ← Change market
      </button>
    </header>
  `;
}

// ── App ───────────────────────────────────────────────────────────────────
function App() {
  const [session,         setSession]         = useState(null);
  const [simulateLoading, setSimulateLoading] = useState(false);
  const [result,          setResult]          = useState(null);
  const [error,           setError]           = useState(null);

  async function handleSimulate(params) {
    setSimulateLoading(true);
    setError(null);
    try {
      const res = await api.simulate(params);
      setResult(res);
    } catch (e) {
      console.error("[CEP Sim] Simulate error:", e);
      setError(e.message || "Simulation failed");
    } finally {
      setSimulateLoading(false);
    }
  }

  function handleReset() {
    setSession(null);
    setResult(null);
    setError(null);
  }

  if (!session) {
    return html`
      <${ErrorBoundary}>
        <${SetupPanel} onSetupComplete=${setSession} />
      <//>
    `;
  }

  return html`
    <${ErrorBoundary}>
      <div class="flex flex-col min-h-screen">
        <${Header} session=${session} onReset=${handleReset} />
        <div class="flex flex-1 overflow-hidden" style=${{ height: "calc(100vh - 53px)" }}>

          <aside class="w-80 shrink-0 border-r border-slate-800 overflow-y-auto p-5 flex flex-col">
            <${ErrorBoundary}>
              <${ConfigPanel}
                session=${session}
                onSimulate=${handleSimulate}
                loading=${simulateLoading}
              />
            <//>
          </aside>

          <main class="flex-1 overflow-y-auto p-6">
            ${error ? html`
              <div class="mb-4 p-3 bg-red-900/40 border border-red-700 rounded-lg text-red-300 text-sm font-mono">
                ${error}
              </div>
            ` : null}
            <${ErrorBoundary}>
              <${ResultsPanel} result=${result} session=${session} />
            <//>
          </main>

        </div>
      </div>
    <//>
  `;
}

const root = createRoot(document.getElementById("root"));
root.render(html`<${App} />`);
