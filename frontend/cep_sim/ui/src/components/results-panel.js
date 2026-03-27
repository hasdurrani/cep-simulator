import { h, htm, useState, useEffect, useMemo, useRef } from "../deps.js";
import { api } from "../api.js";
const html = htm.bind(h);

// ── Plotly chart wrapper ───────────────────────────────────────────────────

function PlotlyChart({ spec, fallbackSrc, fallbackAlt }) {
  const ref = useRef(null);

  useEffect(() => {
    if (!spec || spec === "{}") return;
    if (!ref.current) return;
    if (!window.Plotly) {
      console.error("[CEP Sim] window.Plotly not found — is plotly CDN loaded?");
      return;
    }
    try {
      const fig = JSON.parse(spec);
      window.Plotly.react(ref.current, fig.data || [], fig.layout || {}, {
        responsive: true,
        displayModeBar: false,
      });
    } catch (e) {
      console.error("[CEP Sim] Plotly render failed:", e);
    }
  }, [spec]);

  if (!spec || spec === "{}") {
    if (fallbackSrc) {
      return html`<img src=${fallbackSrc} class="w-full rounded-xl border border-slate-700" alt=${fallbackAlt || ""} />`;
    }
    return html`<p class="text-slate-500 text-xs italic">No chart data — check backend logs</p>`;
  }

  return html`<div ref=${ref} class="w-full rounded-xl overflow-hidden" style=${{ minHeight: "300px" }} />`;
}

// ── Inner sub-tabs (used inside the Simulate tab) ─────────────────────────
const SIM_TABS = [
  { key: "flight",      label: "Flight Simulator" },
  { key: "memory_map",  label: "Memory Map" },
  { key: "calibration", label: "Calibration" },
];

const CHART_FILE = {
  flight:      "campaign_flight_simulator.png",
  memory_map:  "memory_map_comparison.png",
  calibration: "calibration_dashboard.png",
};

// ── Shared components ─────────────────────────────────────────────────────

function MetricBadge({ label, value, color }) {
  return html`
    <div class="bg-slate-800 rounded-xl px-4 py-3 border border-slate-700">
      <div class="text-xs text-slate-500 mb-0.5">${label}</div>
      <div class="text-lg font-semibold ${color || 'text-slate-300'}">${value}</div>
    </div>
  `;
}

function FlightTable({ rows }) {
  if (!rows || !rows.length) {
    return html`<p class="text-slate-500 text-sm">No data</p>`;
  }
  return html`
    <div class="overflow-x-auto mt-4">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-slate-700">
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Brand</th>
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Pre</th>
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Post</th>
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Δ Recall</th>
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Rank pre</th>
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Rank post</th>
            <th class="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Rank Δ</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map((row, i) => {
            const delta = parseFloat(row.delta) || 0;
            const rankDelta = parseInt(row.rank_delta) || 0;
            const deltaColor = delta > 0 ? "text-brand-400" : delta < 0 ? "text-red-400" : "text-slate-400";
            const rankColor  = rankDelta > 0 ? "text-brand-400" : rankDelta < 0 ? "text-red-400" : "text-slate-500";
            const rankArrow  = rankDelta > 0 ? "↑" : rankDelta < 0 ? "↓" : "–";
            return html`
              <tr key=${i} class="border-b border-slate-800 hover:bg-slate-800/50">
                <td class="py-2 px-3 font-medium text-slate-200">${row.brand_name || "—"}</td>
                <td class="py-2 px-3 text-slate-400">${((parseFloat(row.recall_pre) || 0) * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 text-slate-300">${((parseFloat(row.recall_post) || 0) * 100).toFixed(1)}%</td>
                <td class="py-2 px-3 font-semibold ${deltaColor}">${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)}pp</td>
                <td class="py-2 px-3 text-slate-500">${row.rank_pre || "—"}</td>
                <td class="py-2 px-3 text-slate-500">${row.rank_post || "—"}</td>
                <td class="py-2 px-3 ${rankColor}">${rankArrow}${Math.abs(rankDelta)}</td>
              </tr>
            `;
          })}
        </tbody>
      </table>
    </div>
  `;
}

// ── Inline recall heatmap ─────────────────────────────────────────────────

function cellColor(pct, maxPct) {
  if (maxPct === 0) return "transparent";
  const intensity = pct / maxPct;
  // Interpolate from slate-800 (low) → brand-500 (high)
  const r = Math.round(30  + intensity * (34  - 30));
  const g = Math.round(41  + intensity * (197 - 41));
  const b = Math.round(59  + intensity * (94  - 59));
  const a = 0.15 + intensity * 0.85;
  return `rgba(${r},${g},${b},${a})`;
}

function RecallHeatmap({ matrix }) {
  const [sortCol,    setSortCol]    = useState(null);   // null = default order
  const [hoveredRow, setHoveredRow] = useState(null);
  const [hoveredCol, setHoveredCol] = useState(null);

  const scenarios = matrix.scenarios;

  // Global max for colour scale
  const maxPct = useMemo(() => {
    let m = 0;
    for (const row of matrix.rows) for (const v of row.values) if (v > m) m = v;
    return m;
  }, [matrix]);

  // Sorted rows
  const rows = useMemo(() => {
    if (sortCol === null) return matrix.rows;
    const idx = scenarios.indexOf(sortCol);
    return [...matrix.rows].sort((a, b) => b.values[idx] - a.values[idx]);
  }, [matrix, sortCol, scenarios]);

  const thBase  = "px-2 py-1.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider whitespace-nowrap cursor-pointer select-none hover:text-slate-200 transition-colors";
  const tdBase  = "px-2 py-1 text-[11px] text-center font-mono transition-colors";

  return html`
    <div class="overflow-auto rounded-xl border border-slate-700">
      <table class="w-full border-collapse text-xs" style=${{ minWidth: "600px" }}>
        <thead>
          <tr class="border-b border-slate-700 bg-slate-900/80 sticky top-0 z-10">
            <th class="px-3 py-1.5 text-left text-[10px] font-semibold text-slate-400 uppercase tracking-wider sticky left-0 bg-slate-900 z-20 min-w-32">
              Brand
            </th>
            ${scenarios.map((s, i) => html`
              <th key=${s}
                onClick=${() => setSortCol(sortCol === s ? null : s)}
                class="${thBase} ${sortCol === s ? 'text-brand-400' : ''} ${hoveredCol === i ? 'text-slate-200' : ''}">
                ${s.replace(/_/g, " ")}
                ${sortCol === s ? html`<span class="ml-0.5 text-brand-400">↓</span>` : ""}
              </th>
            `)}
          </tr>
        </thead>
        <tbody>
          ${rows.map((row, ri) => html`
            <tr key=${row.brand}
              onMouseEnter=${() => setHoveredRow(ri)}
              onMouseLeave=${() => setHoveredRow(null)}
              class="border-b border-slate-800/60 ${hoveredRow === ri ? 'bg-slate-800/40' : ''}">
              <td class="px-3 py-1 text-[11px] font-medium text-slate-300 sticky left-0 bg-slate-900 whitespace-nowrap z-10 ${hoveredRow === ri ? 'bg-slate-800' : ''}">
                ${row.brand}
              </td>
              ${row.values.map((v, ci) => html`
                <td key=${ci}
                  onMouseEnter=${() => setHoveredCol(ci)}
                  onMouseLeave=${() => setHoveredCol(null)}
                  title="${row.brand} · ${scenarios[ci].replace(/_/g," ")} · ${v.toFixed(1)}%"
                  class="${tdBase} ${hoveredCol === ci ? 'ring-1 ring-inset ring-slate-500' : ''}"
                  style=${{ background: cellColor(v, maxPct), color: v / maxPct > 0.5 ? "#0f172a" : "#94a3b8" }}>
                  ${v > 0 ? v.toFixed(1) : ""}
                </td>
              `)}
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

// ── Market tab ────────────────────────────────────────────────────────────

function MarketTab({ session }) {
  const [baselineData, setBaselineData] = useState(null);
  const [loading,      setLoading]      = useState(false);
  const [error,        setError]        = useState(null);

  useEffect(() => {
    if (!session) return;
    setLoading(true);
    setError(null);
    api.baseline(session.session_id)
      .then(data => {
        setBaselineData(data);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message || "Failed to load baseline data");
        setLoading(false);
      });
  }, [session && session.session_id]);

  if (loading) {
    return html`
      <div class="flex items-center justify-center h-48 text-slate-400 text-sm gap-3">
        <svg class="animate-spin w-5 h-5 text-brand-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
        </svg>
        Loading market baseline…
      </div>
    `;
  }

  if (error) {
    return html`
      <div class="p-4 bg-red-900/40 border border-red-700 rounded-lg text-red-300 text-sm font-mono">
        ${error}
      </div>
    `;
  }

  if (!baselineData) return null;

  const mae = session && session.mae;
  const maeStr = mae != null ? mae.toFixed(3) : "—";

  return html`
    <div class="flex flex-col gap-5">

      <!-- 2×2 summary stats -->
      <div class="grid grid-cols-4 gap-2">
        <${MetricBadge} label="Respondents" value=${(baselineData.respondent_count || 0).toLocaleString()} />
        <${MetricBadge} label="Brands"      value=${baselineData.brand_count || "—"} />
        <${MetricBadge} label="Scenarios"   value=${baselineData.scenario_count || "—"} />
        <${MetricBadge} label="MAE"         value=${maeStr} />
      </div>

      <!-- Brand situation heatmap -->
      <div>
        <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Brand × Scenario Recall</h3>
        ${baselineData.brand_scenario_matrix
          ? html`<${RecallHeatmap} matrix=${baselineData.brand_scenario_matrix} />`
          : html`<p class="text-slate-500 text-sm">No matrix data</p>`
        }
      </div>

      <!-- Brand leaderboard -->
      <div>
        <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Brand Recall Leaderboard</h3>
        <img
          src=${baselineData.artifact_base_url + "/brand_leaderboard.png"}
          class="w-full rounded-xl border border-slate-700"
          alt="Brand recall leaderboard"
        />
      </div>

    </div>
  `;
}

// ── Simulate tab ──────────────────────────────────────────────────────────

function SimulateTab({ result }) {
  const [activeTab, setActiveTab] = useState("flight");

  if (!result) {
    return html`
      <div class="h-full flex flex-col items-center justify-center text-center p-12">
        <div class="w-16 h-16 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center text-3xl mb-4">▶</div>
        <h3 class="text-lg font-medium text-slate-300 mb-2">No simulation yet</h3>
        <p class="text-slate-500 text-sm max-w-xs">Select a brand and purchase occasions on the left, then run a simulation.</p>
      </div>
    `;
  }

  const sessionId      = result.session_id || "";
  const brandName      = result.brand_name || "—";
  const runId          = result.run_id || "";
  const cepLabels      = Array.isArray(result.focal_cep_labels) ? result.focal_cep_labels : [];
  const secLabels      = Array.isArray(result.secondary_cep_labels) ? result.secondary_cep_labels : [];
  const flightTable    = Array.isArray(result.flight_table) ? result.flight_table : [];
  const metrics        = result.metrics || {};
  const artifactBase   = result.artifact_base_url || `/api/artifacts/${sessionId}`;
  const charts         = result.plotly_charts || {};
  console.log("[CEP Sim] plotly_charts keys:", Object.keys(charts), "| Plotly available:", !!window.Plotly);

  const lift      = metrics.focal_brand_lift;
  const liftStr   = lift != null ? `${lift >= 0 ? "+" : ""}${(lift * 100).toFixed(2)}pp` : "—";
  const liftColor = lift != null ? (lift > 0 ? "text-brand-400" : "text-red-400") : "text-slate-300";

  const rankPre   = metrics.focal_brand_rank_pre;
  const rankPost  = metrics.focal_brand_rank_post;
  const rankStr   = (rankPre != null && rankPost != null) ? `${rankPre} → ${rankPost}` : "—";
  const rankColor = (rankPre != null && rankPost != null) ? (rankPost <= rankPre ? "text-brand-400" : "text-red-400") : "text-slate-300";

  const maeVal    = metrics.mae;
  const maeStr    = maeVal != null ? maeVal.toFixed(3) : "—";
  const maeColor  = maeVal != null ? (maeVal < 0.05 ? "text-brand-400" : "text-amber-400") : "text-slate-300";

  const rhoVal    = metrics.median_spearman;
  const rhoStr    = rhoVal != null ? rhoVal.toFixed(3) : "—";
  const rhoColor  = rhoVal != null ? (rhoVal > 0.6 ? "text-brand-400" : "text-amber-400") : "text-slate-300";

  return html`
    <div class="flex flex-col h-full">

      <!-- Run header -->
      <div class="mb-4 flex items-start justify-between gap-3">
        <div class="min-w-0">
          <h2 class="font-semibold text-slate-100 mb-1.5">${brandName}</h2>
          <div class="flex flex-col gap-1">
            ${cepLabels.map((lbl, i) => html`
              <div key=${"p" + i} class="flex items-start gap-2">
                <span class="shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[9px] font-semibold bg-brand-500/20 text-brand-400 border border-brand-500/30 leading-none">P</span>
                <span class="text-xs text-slate-300 leading-snug min-w-0 break-words">${lbl}</span>
              </div>
            `)}
            ${secLabels.map((lbl, i) => html`
              <div key=${"s" + i} class="flex items-start gap-2">
                <span class="shrink-0 mt-0.5 px-1.5 py-0.5 rounded text-[9px] font-semibold bg-blue-500/20 text-blue-400 border border-blue-500/30 leading-none">S</span>
                <span class="text-xs text-slate-400 leading-snug min-w-0 break-words">${lbl}</span>
              </div>
            `)}
            ${cepLabels.length === 0 && secLabels.length === 0 ? html`<span class="text-xs text-slate-500">custom</span>` : null}
          </div>
        </div>
        <span class="text-[10px] text-slate-600 font-mono shrink-0">${runId.slice(0, 8)}</span>
      </div>

      <!-- Metrics row -->
      <div class="grid grid-cols-4 gap-2 mb-5">
        <${MetricBadge} label="Ad lift"           value=${liftStr}  color=${liftColor}  />
        <${MetricBadge} label="Rank pre → post"   value=${rankStr}  color=${rankColor}  />
        <${MetricBadge} label="MAE"               value=${maeStr}   color=${maeColor}   />
        <${MetricBadge} label="Median Spearman ρ" value=${rhoStr}   color=${rhoColor}   />
      </div>

      <!-- Sub-tabs -->
      <div class="flex border-b border-slate-700 mb-4">
        ${SIM_TABS.map(t => html`
          <button key=${t.key}
            onClick=${() => setActiveTab(t.key)}
            class="px-4 py-2 text-sm font-medium transition-all ${activeTab === t.key ? 'tab-active' : 'text-slate-500 hover:text-slate-300'}">
            ${t.label}
          </button>
        `)}
      </div>

      <!-- Sub-tab content -->
      <div class="flex-1 overflow-auto">
        ${activeTab === "flight" ? html`
          <div>
            <${PlotlyChart}
              spec=${charts.flight}
              fallbackSrc=${artifactBase + "/" + CHART_FILE.flight}
              fallbackAlt="Flight simulator"
            />
            <${FlightTable} rows=${flightTable} />
          </div>
        ` : null}
        ${activeTab === "memory_map" ? html`
          <${PlotlyChart}
            spec=${charts.memory_map}
            fallbackSrc=${artifactBase + "/" + CHART_FILE.memory_map}
            fallbackAlt="Memory map"
          />
        ` : null}
        ${activeTab === "calibration" ? html`
          <${PlotlyChart}
            spec=${charts.calibration}
            fallbackSrc=${artifactBase + "/" + CHART_FILE.calibration}
            fallbackAlt="Calibration dashboard"
          />
        ` : null}
      </div>

    </div>
  `;
}

// ── Top-level ResultsPanel with Market / Simulate tabs ────────────────────

export function ResultsPanel({ result, session }) {
  const [topTab, setTopTab] = useState("market");

  // Auto-switch to Simulate tab when a new result arrives
  useEffect(() => {
    if (result) setTopTab("simulate");
  }, [result]);

  return html`
    <div class="flex flex-col h-full">

      <!-- Top-level tab bar -->
      <div class="flex border-b border-slate-700 mb-4">
        <button
          onClick=${() => setTopTab("market")}
          class="px-4 py-2 text-sm font-medium transition-all ${topTab === "market" ? 'tab-active' : 'text-slate-500 hover:text-slate-300'}">
          Market
        </button>
        <button
          onClick=${() => setTopTab("simulate")}
          class="px-4 py-2 text-sm font-medium transition-all ${topTab === "simulate" ? 'tab-active' : 'text-slate-500 hover:text-slate-300'}">
          Simulate
        </button>
      </div>

      <!-- Tab content -->
      <div class="flex-1 overflow-auto">
        ${topTab === "market" ? html`<${MarketTab} session=${session} />` : null}
        ${topTab === "simulate" ? html`<${SimulateTab} result=${result} />` : null}
      </div>

    </div>
  `;
}
