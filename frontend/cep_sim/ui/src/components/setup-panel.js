import { h, htm, useState, useEffect } from "../deps.js";
import { api } from "../api.js";
const html = htm.bind(h);

export function SetupPanel({ onSetupComplete }) {
  const [configs, setConfigs]     = useState([]);
  const [selected, setSelected]   = useState("uk");
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);

  useEffect(() => {
    api.configs().then(setConfigs).catch(() => {});
  }, []);

  async function handleSetup() {
    setLoading(true);
    setError(null);
    try {
      const session = await api.setup({ config_key: selected });
      onSetupComplete(session);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return html`
    <div class="min-h-screen flex items-center justify-center p-8">
      <div class="w-full max-w-md">

        <div class="text-center mb-10">
          <div class="inline-flex items-center gap-2 mb-4">
            <div class="w-8 h-8 rounded-lg bg-brand-500 flex items-center justify-center text-slate-900 font-bold text-sm">C</div>
            <span class="text-xl font-semibold tracking-tight">CEP Simulator</span>
          </div>
          <p class="text-slate-400 text-sm">Campaign scenario planning powered by memory-based brand associations</p>
        </div>

        <div class="bg-slate-800 rounded-2xl p-6 border border-slate-700">
          <h2 class="text-sm font-medium text-slate-300 mb-4 uppercase tracking-wider">Select market</h2>

          <div class="space-y-3 mb-6">
            ${configs.map(c => html`
              <label key=${c.key} class="flex items-center gap-3 p-3 rounded-xl border cursor-pointer transition-all ${selected === c.key ? 'border-brand-500 bg-brand-500/10' : 'border-slate-600 hover:border-slate-500'}">
                <input type="radio" name="config" value=${c.key}
                  checked=${selected === c.key}
                  onChange=${() => setSelected(c.key)}
                  class="accent-brand-500" />
                <div>
                  <div class="text-sm font-medium">${c.label}</div>
                  <div class="text-xs text-slate-400">${c.country} market</div>
                </div>
              </label>
            `)}
          </div>

          ${error && html`
            <div class="mb-4 p-3 bg-red-900/40 border border-red-700 rounded-lg text-red-300 text-sm">${error}</div>
          `}

          <button
            onClick=${handleSetup}
            disabled=${loading}
            class="w-full py-3 rounded-xl font-medium text-sm transition-all ${loading ? 'bg-slate-700 text-slate-400 cursor-not-allowed' : 'bg-brand-500 hover:bg-brand-600 text-slate-900'}">
            ${loading
              ? html`<span class="flex items-center justify-center gap-2"><span class="spinner"></span> Loading data…</span>`
              : "Load Market Data →"}
          </button>

          <p class="mt-3 text-center text-xs text-slate-500">Loads survey data and computes baseline recall</p>
        </div>

      </div>
    </div>
  `;
}
