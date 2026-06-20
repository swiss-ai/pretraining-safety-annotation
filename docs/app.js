/* Charter generator-eval results page. Reads window.CHARTER_DATA (data.js). */
(function () {
  "use strict";
  const D = window.CHARTER_DATA;
  if (!D) { document.body.innerHTML = "<p style='padding:40px'>data.js failed to load.</p>"; return; }

  // ---- palette (model-raising identity) ----
  const C = {
    black: "#001219", deepTeal: "#005f73", teal: "#0a9396", lightTeal: "#94d2bd",
    cream: "#e9d8a6", amber: "#ee9b00", rust: "#bb3e03", darkRed: "#9b2226",
    grid: "#d9d2c4", gray700: "#4d4d4d", gray500: "#808080",
  };
  const FONT = "Inter, Helvetica, Arial, sans-serif";

  // base plotly layout mirroring mra.apply_plotly()
  function baseLayout(extra) {
    return Object.assign({
      paper_bgcolor: "white", plot_bgcolor: "white",
      font: { family: FONT, size: 12, color: C.black },
      margin: { l: 64, r: 28, t: 18, b: 56 },
      xaxis: { showgrid: true, gridcolor: C.grid, gridwidth: 0.6, linecolor: C.black, linewidth: 0.8, ticks: "outside", zerolinecolor: C.grid },
      yaxis: { showgrid: true, gridcolor: C.grid, gridwidth: 0.6, linecolor: C.black, linewidth: 0.8, ticks: "outside", zerolinecolor: C.grid },
      legend: { bgcolor: "rgba(0,0,0,0)", orientation: "h", x: 0, y: 1.08, xanchor: "left", yanchor: "bottom" },
      hoverlabel: { font: { family: FONT } },
    }, extra || {});
  }
  const CONFIG = { displayModeBar: false, responsive: true };

  // ---- formatting helpers ----
  const pct = (x) => (x == null ? "—" : (x * 100).toFixed(1) + "%");
  const gpuFmt = (g) => (g >= 100000 ? Math.round(g / 1000) + "K" : (g / 1000).toFixed(1) + "K");
  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ============================ HEADLINE PLOT ============================
  function renderHeadline() {
    const traces = D.models.map((m) => ({
      x: [m.throughput.samples_per_sec],
      y: [m.overall.mean_agg],
      name: m.label,
      mode: "markers",
      type: "scatter",
      marker: { size: 17, color: m.color, line: { color: "white", width: 1.5 } },
      hovertemplate:
        `<b>${m.label}</b> (${m.prompt})<br>` +
        `${m.throughput.samples_per_sec.toFixed(2)} samples/s · ~${gpuFmt(m.throughput.gpu_hours)} GPU-h<br>` +
        `mean ${m.overall.mean_agg.toFixed(3)} · accept ${pct(m.overall.accept)}<extra></extra>`,
    }));

    // place each label so the three don't collide
    const offs = {
      "qwen3.6-35b-a3b": { ay: -52, ax: 0 },
      "gemma-4-31b": { ay: 2, ax: 92 },
      "gemma-4-26b-a4b": { ay: 56, ax: 0 },
    };
    const annotations = D.models.map((m) => {
      const o = offs[m.id] || { ay: -50, ax: 0 };
      return {
        x: Math.log10(m.throughput.samples_per_sec), y: m.overall.mean_agg,
        xref: "x", yref: "y", ax: o.ax, ay: o.ay, showarrow: true,
        arrowhead: 0, arrowwidth: 1, arrowcolor: C.gray500,
        align: o.ax > 0 ? "left" : "center",
        font: { family: FONT, size: 11.5, color: C.black },
        text: `<b>${m.label}</b><br>${m.throughput.samples_per_sec.toFixed(2)} smp/s · ~${gpuFmt(m.throughput.gpu_hours)} GPU-h<br>${m.overall.mean_agg.toFixed(2)} · ${pct(m.overall.accept)} accept`,
        bgcolor: "rgba(255,255,255,0.82)", bordercolor: m.color, borderwidth: 1, borderpad: 4,
      };
    });

    const layout = baseLayout({
      annotations,
      showlegend: false,
      margin: { l: 66, r: 36, t: 14, b: 58 },
      xaxis: Object.assign(baseLayout().xaxis, {
        type: "log", title: { text: "Generation throughput — samples / sec (log), higher = faster & cheaper →", font: { size: 12.5 } },
        range: [Math.log10(0.15), Math.log10(4.7)],
        tickvals: [0.2, 0.5, 1, 2, 3], ticktext: ["0.2", "0.5", "1", "2", "3"],
      }),
      yaxis: Object.assign(baseLayout().yaxis, {
        title: { text: "Aggregate judge quality (mean, 1–5)", font: { size: 12.5 } },
        range: [4.35, 4.60], dtick: 0.05,
      }),
    });
    Plotly.newPlot("plot-headline", traces, layout, CONFIG);
  }

  function headlineDesc() {
    const by = {}; D.models.forEach((m) => (by[m.id] = m));
    const q = by["qwen3.6-35b-a3b"], g31 = by["gemma-4-31b"], g26 = by["gemma-4-26b-a4b"];
    const ratio = (g31.throughput.gpu_hours / q.throughput.gpu_hours).toFixed(0);
    document.getElementById("headline-desc").innerHTML =
      `<strong>What we see.</strong> All three generators land within ~0.06 of each other on judge-aggregate ` +
      `quality (${g26.overall.mean_agg.toFixed(2)}–${g31.overall.mean_agg.toFixed(2)} out of 5) — quality is tightly clustered. ` +
      `Throughput, by contrast, spans <strong>${ratio}×</strong>. ` +
      `<strong>${q.label}</strong> delivers essentially the same quality (${q.overall.mean_agg.toFixed(2)}, ${pct(q.overall.accept)} accept) ` +
      `at ${q.throughput.samples_per_sec.toFixed(2)} samples/s / ~${gpuFmt(q.throughput.gpu_hours)} GPU-hours — by far the cheapest. ` +
      `<strong>${g31.label}</strong> edges out the top quality (${g31.overall.mean_agg.toFixed(2)}, ${pct(g31.overall.accept)}), but as a ` +
      `dense model with no active-parameter reduction it runs at just ${g31.throughput.samples_per_sec.toFixed(2)} samples/s / ` +
      `~${gpuFmt(g31.throughput.gpu_hours)} GPU-hours — <strong>${ratio}× the compute for +${(g31.overall.mean_agg - q.overall.mean_agg).toFixed(2)} quality</strong>. ` +
      `<strong>${g26.label}</strong> (A4B MoE) sits in between at ~${gpuFmt(g26.throughput.gpu_hours)} GPU-hours. ` +
      `For a 100M-document scale run, ${q.label} is the clear pick: near-top quality at a fraction of the cost.`;
  }

  // ============================ PER-LANGUAGE BARS ============================
  let langMetric = "accept";
  function renderLangs() {
    const langs = D.lang_order;
    const labels = langs.map((l) => D.lang_label[l]);
    const traces = D.models.map((m) => ({
      x: labels,
      y: langs.map((l) => {
        const s = m.by_lang[l];
        if (!s || s.accept == null) return null;
        return langMetric === "accept" ? s.accept * 100 : s.mean_agg;
      }),
      name: m.label,
      type: "bar",
      marker: { color: m.color },
      hovertemplate: `<b>${m.label}</b><br>%{x}: %{y:.1f}` + (langMetric === "accept" ? "%" : "") + `<extra></extra>`,
    }));
    const yaxis = langMetric === "accept"
      ? { title: { text: "Accept rate", font: { size: 12.5 } }, range: [70, 100], ticksuffix: "%" }
      : { title: { text: "Mean aggregate (1–5)", font: { size: 12.5 } }, range: [4.0, 4.7] };
    const layout = baseLayout({
      barmode: "group", bargap: 0.28, bargroupgap: 0.08,
      margin: { l: 56, r: 20, t: 30, b: 48 },
      xaxis: Object.assign(baseLayout().xaxis, { showgrid: false }),
      yaxis: Object.assign(baseLayout().yaxis, yaxis),
    });
    Plotly.react("plot-langs", traces, layout, CONFIG);
  }

  function langsDesc() {
    document.getElementById("bars-sub").textContent =
      "English is the full dclm-en bench; the six others are fw2-multi; edge cases are the curated hard set.";
    document.getElementById("langs-desc").innerHTML =
      `<strong>What we see.</strong> Quality is even across languages for Qwen (no language below 90% accept), while both ` +
      `Gemma models dip on <strong>Japanese</strong> (83–85%). The <strong>edge cases</strong> are the hardest split for every model — ` +
      `Qwen holds up best (87.6%), Gemma-4-31B is close (84.4%), and Gemma-4-26B-A4B struggles most (75.2%).`;
  }

  // ============================ SUMMARY TABLE ============================
  function renderTable() {
    const head =
      `<thead><tr><th>Model</th><th>Prompt</th><th>Arch</th><th>Throughput</th><th>GPU-hours</th>` +
      `<th>Quality (mean)</th><th>Accept (4k)</th><th>Edge accept</th></tr></thead>`;
    const rows = D.models.map((m) =>
      `<tr><td class="model-cell"><span class="swatch" style="background:${m.color}"></span>${esc(m.label)}</td>` +
      `<td>${esc(m.prompt)}</td><td>${esc(m.arch)}</td>` +
      `<td>${m.throughput.samples_per_sec.toFixed(2)} smp/s</td>` +
      `<td>~${m.throughput.gpu_hours.toLocaleString()}</td>` +
      `<td>${m.overall.mean_agg.toFixed(3)}</td>` +
      `<td>${pct(m.overall.accept)}</td>` +
      `<td>${pct(m.by_lang.edge.accept)}</td></tr>`).join("");
    document.getElementById("summary-table").innerHTML = head + "<tbody>" + rows + "</tbody>";
    document.getElementById("summary-foot").textContent =
      `${D.bench_note} GPU-hours extrapolated to the full corpus on a 4-GPU GH200 node, client concurrency 1024, ` +
      `all on the ${D.reflection_policy} reflection cutoff. Judge: ${D.judge}; accept = aggregate ≥ ${D.accept_threshold}.`;
  }

  // ============================ INSPECTOR ============================
  function initInspector() {
    const selModel = document.getElementById("sel-model");
    const selLang = document.getElementById("sel-lang");
    const selFilter = document.getElementById("sel-filter");

    D.models.forEach((m) => selModel.add(new Option(m.label, m.id)));
    selLang.add(new Option("All languages", "all"));
    D.lang_order.forEach((l) => selLang.add(new Option(D.lang_label[l], l)));

    [selModel, selLang, selFilter].forEach((s) => s.addEventListener("change", renderList));
    renderList();
  }

  function currentSamples() {
    const model = document.getElementById("sel-model").value;
    const lang = document.getElementById("sel-lang").value;
    const filt = document.getElementById("sel-filter").value;
    return D.samples.filter((s) =>
      s.model === model &&
      (lang === "all" || s.subset === lang) &&
      (filt === "all" || (filt === "accept" ? s.accepted : !s.accepted)));
  }

  function renderList() {
    const list = document.getElementById("record-list");
    const rows = currentSamples();
    if (!rows.length) {
      list.innerHTML = `<li class="list-empty">No records for this selection.</li>`;
      document.getElementById("record-detail").innerHTML = `<p class="empty-hint">No records.</p>`;
      return;
    }
    list.innerHTML = rows.map((s, i) => {
      const snip = (s.reflection_1p || s.text || "").slice(0, 150);
      return `<li class="record-row" data-i="${i}">
        <div class="row-top">
          <span class="badge lang">${esc(D.lang_label[s.subset] || s.subset)}</span>
          <span class="badge ${s.accepted ? "accept" : "reject"}">${s.accepted ? "accept" : "reject"}</span>
          <span class="badge score">agg ${s.aggregate.toFixed(1)}</span>
        </div>
        <div class="snippet">${esc(snip)}</div>
      </li>`;
    }).join("");

    Array.from(list.querySelectorAll(".record-row")).forEach((el) => {
      el.addEventListener("click", () => {
        list.querySelectorAll(".record-row").forEach((r) => r.classList.remove("is-active"));
        el.classList.add("is-active");
        renderDetail(rows[+el.dataset.i]);
      });
    });
    // auto-open the first
    list.querySelector(".record-row").classList.add("is-active");
    renderDetail(rows[0]);
  }

  function scoreChip(k, label, v, isAgg) {
    return `<div class="score-chip ${isAgg ? "agg" : ""}"><span class="k">${label}</span><span class="v">${v == null ? "—" : v}</span></div>`;
  }

  function docWithMarker(text, rp, truncated) {
    const cut = Math.min(rp == null ? text.length : rp, text.length);
    const before = esc(text.slice(0, cut));
    const after = esc(text.slice(cut));
    let html = before;
    if (after.length) {
      html += `<span class="rp-marker">— reflection point (model read up to here) —</span>` +
        `<span style="color:var(--gray-500)">${after}</span>`;
    } else {
      html += `<span class="rp-marker">— reflection written after the full text above —</span>`;
    }
    if (truncated) html += `<span style="color:var(--gray-500)"> …(document truncated for display)</span>`;
    return html;
  }

  function renderDetail(s) {
    const sc = s.scores || {};
    const elems = (s.charter_elements || []).join(", ") || "none cited";
    const html =
      `<div class="detail-head">
         <h3>${esc(D.lang_label[s.subset] || s.subset)}</h3>
         <span class="badge ${s.accepted ? "accept" : "reject"}">${s.accepted ? "accepted" : "rejected"}</span>
       </div>
       <div class="detail-meta">
         item ${esc(String(s.item_id).slice(0, 28))} · safety score ${s.safety_score ?? "—"} ·
         ${s.output_tokens ?? "—"} output tokens · charter cited: ${esc(elems)}
       </div>
       <div class="scorebar">
         ${scoreChip("relevance", "Relevance", sc.relevance)}
         ${scoreChip("specificity", "Specificity", sc.specificity)}
         ${scoreChip("charter_grounding", "Charter", sc.charter_grounding)}
         ${scoreChip("voice_tone", "Voice/Tone", sc.voice_tone)}
         ${scoreChip("aggregate", "Aggregate", s.aggregate.toFixed(2), true)}
       </div>
       <div class="block">
         <h4>Reflection (model output)</h4>
         <div class="reflection-box">${esc(s.reflection_1p)}</div>
       </div>
       <div class="block">
         <h4>Judge verdict</h4>
         <div class="reasoning-box">${esc(s.reasoning)}</div>
       </div>
       <div class="block">
         <h4>Source document</h4>
         <div class="doc-text">${docWithMarker(s.text, s.reflection_point, s.text_truncated)}</div>
       </div>
       <div class="block">
         <details class="analysis"><summary>Model analysis (scratchpad)</summary>
           <div class="analysis-box">${esc(s.analysis)}</div>
         </details>
       </div>`;
    document.getElementById("record-detail").innerHTML = html;
  }

  // ============================ TABS / META ============================
  function initTabs() {
    document.querySelectorAll(".tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        const tab = btn.dataset.tab;
        document.getElementById("tab-overview").hidden = tab !== "overview";
        document.getElementById("tab-inspector").hidden = tab !== "inspector";
        if (tab === "overview") { Plotly.Plots.resize("plot-headline"); Plotly.Plots.resize("plot-langs"); }
      });
    });
    document.getElementById("metric-toggle").addEventListener("click", (e) => {
      const b = e.target.closest(".toggle-btn"); if (!b) return;
      document.querySelectorAll(".toggle-btn").forEach((x) => x.classList.remove("is-active"));
      b.classList.add("is-active");
      langMetric = b.dataset.metric;
      renderLangs();
    });
  }

  function initMeta() {
    document.getElementById("meta-judge").textContent = D.judge;
    document.getElementById("meta-policy").textContent = D.reflection_policy;
    document.getElementById("meta-foot").textContent =
      `Reflection annotation pipeline · ${D.models.length} candidate generators · judged by ${D.judge}.`;
  }

  // ---- go ----
  initMeta();
  initTabs();
  renderHeadline();
  headlineDesc();
  renderLangs();
  langsDesc();
  renderTable();
  initInspector();
})();
