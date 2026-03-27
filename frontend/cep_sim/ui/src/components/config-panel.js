import { h, htm, useState, useMemo } from "../deps.js";
const html = htm.bind(h);

const CHANNELS = ["digital_video", "social_media", "tv", "ooh", "radio", "cinema"];
const EMOTIONS  = ["social_warmth", "confidence", "pride", "fun", "depth", "nostalgia"];

function CepFamilyGroup({ family, ceps, focalIds, secondaryIds, onToggleFocal, onToggleSecondary }) {
  const [open, setOpen] = useState(true);
  return html`
    <div class="mb-2">
      <button onClick=${() => setOpen(!open)}
        class="w-full flex items-center justify-between py-1.5 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-300">
        <span>${family}</span>
        <span class="text-slate-600">${open ? "▲" : "▼"}</span>
      </button>
      ${open && html`
        <div class="space-y-1">
          ${ceps.map(cep => {
            const isFocal     = focalIds.includes(cep.cep_id);
            const isSecondary = secondaryIds.includes(cep.cep_id);
            return html`
              <div key=${cep.cep_id}
                class="flex items-center gap-2 px-2 py-1.5 rounded-lg text-xs ${isFocal ? 'bg-brand-500/15 border border-brand-500/40' : isSecondary ? 'bg-blue-500/10 border border-blue-500/30' : 'hover:bg-slate-700/50'}">
                <div class="flex-1 text-slate-300 leading-tight">${cep.cep_description}</div>
                <div class="flex gap-1 shrink-0">
                  <button onClick=${() => onToggleFocal(cep.cep_id)}
                    class="px-1.5 py-0.5 rounded text-[10px] font-medium transition-all ${isFocal ? 'bg-brand-500 text-slate-900' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'}">
                    P
                  </button>
                  <button onClick=${() => onToggleSecondary(cep.cep_id)}
                    class="px-1.5 py-0.5 rounded text-[10px] font-medium transition-all ${isSecondary ? 'bg-blue-500 text-white' : 'bg-slate-700 text-slate-400 hover:bg-slate-600'}">
                    S
                  </button>
                </div>
              </div>
            `;
          })}
        </div>
      `}
    </div>
  `;
}

export function ConfigPanel({ session, onSimulate, loading }) {
  const [brandId,          setBrandId]          = useState(session.default_brand_id);
  const [focalIds,         setFocalIds]          = useState(session.default_focal_cep_ids);
  const [secondaryIds,     setSecondaryIds]      = useState(session.default_secondary_cep_ids);
  const [brandingClarity,  setBrandingClarity]   = useState(0.9);
  const [attentionWeight,  setAttentionWeight]   = useState(1.0);
  const [channel,          setChannel]           = useState("digital_video");
  const [emotion,          setEmotion]           = useState("social_warmth");
  const [brandOpen,        setBrandOpen]         = useState(true);
  const [cepsOpen,         setCepsOpen]          = useState(true);
  const [adOpen,           setAdOpen]            = useState(true);

  function toggleFocal(cepId) {
    setFocalIds(prev =>
      prev.includes(cepId) ? prev.filter(x => x !== cepId) : [...prev, cepId]
    );
    // Remove from secondary if being added to focal
    setSecondaryIds(prev => prev.filter(x => x !== cepId));
  }

  function toggleSecondary(cepId) {
    setSecondaryIds(prev =>
      prev.includes(cepId) ? prev.filter(x => x !== cepId) : [...prev, cepId]
    );
    // Remove from focal if being added to secondary
    setFocalIds(prev => prev.filter(x => x !== cepId));
  }

  const focalLabel = useMemo(() => {
    if (!focalIds.length) return "custom";
    return focalIds
      .map(id => {
        for (const ceps of Object.values(session.cep_families)) {
          const found = ceps.find(c => c.cep_id === id);
          if (found) return found.cep_label || found.cep_id;
        }
        return id;
      })
      .join(" + ");
  }, [focalIds, session.cep_families]);

  function handleRun() {
    onSimulate({
      session_id:           session.session_id,
      brand_id:             brandId,
      focal_cep_ids:        focalIds,
      secondary_cep_ids:    secondaryIds,
      focal_scenario_label: focalLabel,
      branding_clarity:     brandingClarity,
      attention_weight:     attentionWeight,
      channel,
      emotion,
    });
  }

  const canRun = focalIds.length > 0 && !!brandId;

  return html`
    <div class="flex flex-col h-full">

      <!-- Brand -->
      <div class="border-b border-slate-800">
        <button onClick=${() => setBrandOpen(!brandOpen)}
          class="w-full flex items-center justify-between py-2.5 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-300">
          <span>Brand</span>
          <span class="text-slate-600">${brandOpen ? "▲" : "▼"}</span>
        </button>
        ${brandOpen && html`
          <div class="pb-3">
            <select
              value=${brandId}
              onChange=${e => setBrandId(e.target.value)}
              class="w-full px-3 py-2 rounded-lg text-sm">
              ${session.brands.map(b => html`
                <option key=${b.brand_id} value=${b.brand_id}>${b.brand_name}</option>
              `)}
            </select>
          </div>
        `}
      </div>

      <!-- CEP Picker -->
      <div class="border-b border-slate-800">
        <button onClick=${() => setCepsOpen(!cepsOpen)}
          class="w-full flex items-center justify-between py-2.5 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-300">
          <span class="whitespace-nowrap">Purchase occasions</span>
          <span class="text-slate-600">${cepsOpen ? "▲" : "▼"}</span>
        </button>
        ${cepsOpen && html`
          <div class="pb-3">
            <div class="flex gap-3 text-[10px] text-slate-500 mb-2">
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded bg-brand-500 inline-block"></span>Primary</span>
              <span class="flex items-center gap-1"><span class="w-2.5 h-2.5 rounded bg-blue-500 inline-block"></span>Secondary</span>
            </div>
            ${focalIds.length === 0 && html`
              <p class="text-xs text-amber-400 mb-2 px-1">Select at least one primary occasion (P)</p>
            `}
            <div class="overflow-y-auto pr-1" style=${{ maxHeight: "40vh" }}>
              ${Object.entries(session.cep_families).map(([family, ceps]) => html`
                <${CepFamilyGroup}
                  key=${family}
                  family=${family}
                  ceps=${ceps}
                  focalIds=${focalIds}
                  secondaryIds=${secondaryIds}
                  onToggleFocal=${toggleFocal}
                  onToggleSecondary=${toggleSecondary}
                />
              `)}
            </div>
          </div>
        `}
      </div>

      <!-- Ad Settings -->
      <div class="border-b border-slate-800">
        <button onClick=${() => setAdOpen(!adOpen)}
          class="w-full flex items-center justify-between py-2.5 text-xs font-semibold text-slate-400 uppercase tracking-wider hover:text-slate-300">
          <span>Ad settings</span>
          <span class="text-slate-600">${adOpen ? "▲" : "▼"}</span>
        </button>
        ${adOpen && html`
          <div class="pb-4 space-y-3">
            <div>
              <div class="flex justify-between text-xs text-slate-400 mb-1">
                <span>Branding clarity</span><span class="text-slate-300 font-medium">${brandingClarity.toFixed(2)}</span>
              </div>
              <input type="range" min="0.1" max="1" step="0.05"
                value=${brandingClarity}
                onInput=${e => setBrandingClarity(parseFloat(e.target.value))}
                class="w-full" />
            </div>

            <div>
              <div class="flex justify-between text-xs text-slate-400 mb-1">
                <span>Attention weight</span><span class="text-slate-300 font-medium">${attentionWeight.toFixed(2)}</span>
              </div>
              <input type="range" min="0.5" max="2" step="0.1"
                value=${attentionWeight}
                onInput=${e => setAttentionWeight(parseFloat(e.target.value))}
                class="w-full" />
            </div>

            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-xs text-slate-400 mb-1">Channel</label>
                <select value=${channel} onChange=${e => setChannel(e.target.value)} class="w-full px-2 py-1.5 rounded-lg text-xs">
                  ${CHANNELS.map(c => html`<option key=${c} value=${c}>${c.replace("_", " ")}</option>`)}
                </select>
              </div>
              <div>
                <label class="block text-xs text-slate-400 mb-1">Emotion</label>
                <select value=${emotion} onChange=${e => setEmotion(e.target.value)} class="w-full px-2 py-1.5 rounded-lg text-xs">
                  ${EMOTIONS.map(e => html`<option key=${e} value=${e}>${e.replace("_", " ")}</option>`)}
                </select>
              </div>
            </div>
          </div>
        `}
      </div>

      <!-- Run button -->
      <button
        onClick=${handleRun}
        disabled=${loading || !canRun}
        class="mt-4 w-full py-3 rounded-xl font-semibold text-sm transition-all ${loading || !canRun ? 'bg-slate-700 text-slate-500 cursor-not-allowed' : 'bg-brand-500 hover:bg-brand-600 text-slate-900'}">
        ${loading
          ? html`<span class="flex items-center justify-center gap-2"><span class="spinner"></span>Running simulation…</span>`
          : "▶  Run simulation"}
      </button>

    </div>
  `;
}
