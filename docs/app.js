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
  let headlineMetric = "accept"; // "accept" | "quality"
  function renderHeadline() {
    const isAcc = headlineMetric === "accept";
    const yval = (m) => (isAcc ? m.overall.accept * 100 : m.overall.mean_agg);
    const traces = D.models.map((m) => ({
      x: [m.throughput.samples_per_sec],
      y: [yval(m)],
      name: m.label,
      mode: "markers",
      type: "scatter",
      marker: { size: 17, color: m.color, line: { color: "white", width: 1.5 } },
      hoverinfo: "skip",
    }));

    // place each label so the three don't collide (same ordering holds for both metrics)
    const offs = {
      "qwen3.6-35b-a3b": { ay: -52, ax: 0 },
      "gemma-4-31b": { ay: 2, ax: 92 },
      "gemma-4-26b-a4b": { ay: 56, ax: 0 },
    };
    const annotations = D.models.map((m) => {
      const o = offs[m.id] || { ay: -50, ax: 0 };
      const lead = isAcc
        ? `${pct(m.overall.accept)} · ${m.overall.mean_agg.toFixed(2)}`
        : `${m.overall.mean_agg.toFixed(2)} · ${pct(m.overall.accept)}`;
      return {
        x: Math.log10(m.throughput.samples_per_sec), y: yval(m),
        xref: "x", yref: "y", ax: o.ax, ay: o.ay, showarrow: true,
        arrowhead: 0, arrowwidth: 1, arrowcolor: C.gray500,
        align: o.ax > 0 ? "left" : "center",
        font: { family: FONT, size: 11.5, color: C.black },
        text: `<b>${m.label}</b><br>${m.throughput.samples_per_sec.toFixed(2)} smp/s · ~${gpuFmt(m.throughput.gpu_hours)} GPU-h<br>${lead}`,
        bgcolor: "rgba(255,255,255,0.82)", bordercolor: m.color, borderwidth: 1, borderpad: 4,
      };
    });

    const yaxis = isAcc
      ? { title: { text: "Accept rate", font: { size: 12.5 } }, range: [86, 94], ticksuffix: "%" }
      : { title: { text: "Mean aggregate (1–5)", font: { size: 12.5 } }, range: [4.35, 4.60], dtick: 0.05 };

    const layout = baseLayout({
      annotations,
      showlegend: false,
      hovermode: false,
      margin: { l: 66, r: 36, t: 14, b: 58 },
      xaxis: Object.assign(baseLayout().xaxis, {
        type: "log", title: { text: "Generation throughput — samples/sec per node (4× GH200, log), higher = faster & cheaper →", font: { size: 12.5 } },
        range: [Math.log10(0.15), Math.log10(4.7)],
        tickvals: [0.2, 0.5, 1, 2, 3], ticktext: ["0.2", "0.5", "1", "2", "3"],
      }),
      yaxis: Object.assign(baseLayout().yaxis, yaxis),
    });
    Plotly.react("plot-headline", traces, layout, CONFIG);
  }

  function headlineDesc() {
    const by = {}; D.models.forEach((m) => (by[m.id] = m));
    const q = by["qwen3.6-35b-a3b"], g31 = by["gemma-4-31b"], g26 = by["gemma-4-26b-a4b"];
    const ratio = (g31.throughput.gpu_hours / q.throughput.gpu_hours).toFixed(0);
    document.getElementById("headline-desc").innerHTML =
      `<strong>What we see.</strong> Quality is tightly clustered ` +
      `(${g26.overall.mean_agg.toFixed(2)}–${g31.overall.mean_agg.toFixed(2)} of 5) while throughput spans <strong>${ratio}×</strong>. ` +
      `<strong>${q.label}</strong> is the clear scale pick — near-top quality at ~${gpuFmt(q.throughput.gpu_hours)} GPU-h, ` +
      `vs ~${gpuFmt(g31.throughput.gpu_hours)} for the dense <strong>${g31.label}</strong> (best quality, ${ratio}× the cost) ` +
      `and ~${gpuFmt(g26.throughput.gpu_hours)} for ${g26.label}.`;
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
      `<thead><tr><th>Model</th><th>Prompt</th><th>Arch</th><th>Throughput / node</th><th>GPU-h (100M)</th>` +
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
      `${D.bench_note} Throughput is ${D.throughput_unit}; GPU-h is the estimate for ` +
      `${Math.round(D.scale_samples / 1e6)}M samples on a ${D.node_gpus}-GPU GH200 node (client concurrency 1024), ` +
      `all on the ${D.reflection_policy} reflection cutoff. Judge: ${D.judge_model}; accept = aggregate ≥ ${D.accept_threshold}.`;
  }

  // ============================ ACCEPT BY SAFETY SCORE ============================
  let safetyMetric = "accept";
  function renderSafety() {
    const isAcc = safetyMetric === "accept";
    const order = D.safety_order;
    const labels = order.map((s) => D.safety_label[String(s)]);
    const traces = D.models.map((m) => ({
      x: labels,
      y: order.map((s) => {
        const st = m.by_safety[String(s)];
        if (!st || st.accept == null) return null;
        return isAcc ? st.accept * 100 : st.mean_agg;
      }),
      customdata: order.map((s) => (m.by_safety[String(s)] || {}).n || 0),
      name: m.label, type: "bar", marker: { color: m.color },
      hovertemplate: `<b>${m.label}</b><br>%{x}: %{y:.1f}` + (isAcc ? "%" : "") + ` (n=%{customdata})<extra></extra>`,
    }));
    const yaxis = isAcc
      ? { title: { text: "Accept rate", font: { size: 12.5 } }, range: [80, 100], ticksuffix: "%" }
      : { title: { text: "Mean aggregate (1–5)", font: { size: 12.5 } }, range: [4.0, 4.8] };
    const layout = baseLayout({
      barmode: "group", bargap: 0.42, bargroupgap: 0.08,
      margin: { l: 56, r: 20, t: 30, b: 40 },
      xaxis: Object.assign(baseLayout().xaxis, { showgrid: false }),
      yaxis: Object.assign(baseLayout().yaxis, yaxis),
    });
    Plotly.react("plot-safety", traces, layout, CONFIG);
  }

  function safetyDesc() {
    document.getElementById("safety-sub").textContent =
      "The scale corpus is pre-filtered to safety ≥ 4, so only borderline (4) and clearly-safe (5) documents appear.";
    document.getElementById("safety-desc").innerHTML =
      `<strong>What we see.</strong> Counter-intuitively, the <strong>borderline safety-4 documents are accepted more often</strong> ` +
      `(94–99%) than the clearly-safe safety-5 ones (90–91%), consistently across all three models. Borderline text carries real ` +
      `value tensions for the reflection to engage with, so the model writes a more specific, relevant annotation; on wholly benign ` +
      `text the most honest reflection is often "nothing at stake", which the judge scores a little lower. The safety-4 slice is ` +
      `small (~250–270 items), so its rate is noisier.`;
  }

  // ============================ REFLECTION LENGTH ============================
  function renderLength() {
    const centers = D.length_bin_centers;
    const traces = D.models.map((m) => ({
      x: centers, y: m.length.hist_pct,
      name: m.label, type: "scatter", mode: "lines", line: { color: m.color, width: 2 },
      hovertemplate: `<b>${m.label}</b><br>~%{x} tok: %{y:.1f}% of reflections<extra></extra>`,
    }));
    const cap = D.length_cap;
    const layout = baseLayout({
      margin: { l: 56, r: 20, t: 24, b: 50 },
      xaxis: Object.assign(baseLayout().xaxis, {
        title: { text: `Reflection length (${D.length_tokenizer} tokens)`, font: { size: 12.5 } },
        range: [0, 380],
      }),
      yaxis: Object.assign(baseLayout().yaxis, { title: { text: "% of reflections", font: { size: 12.5 } }, rangemode: "tozero" }),
      shapes: [{ type: "line", x0: cap, x1: cap, yref: "paper", y0: 0, y1: 1, line: { color: C.gray500, width: 1.2, dash: "dash" } }],
      annotations: [{ x: cap, yref: "paper", y: 1, yanchor: "bottom", showarrow: false, text: `${cap}-token cutoff`, font: { size: 11, color: C.gray700 } }],
    });
    Plotly.react("plot-length", traces, layout, CONFIG);

    const head = `<thead><tr><th>Model</th><th>n</th><th>Mean</th><th>Median</th><th>p90</th><th>p95</th><th>Max</th><th>&gt;200</th><th>&gt;256</th></tr></thead>`;
    const rows = D.models.map((m) => {
      const L = m.length;
      return `<tr><td class="model-cell"><span class="swatch" style="background:${m.color}"></span>${esc(m.label)}</td>` +
        `<td>${L.n}</td><td>${L.mean}</td><td>${L.median}</td><td>${L.p90}</td><td>${L.p95}</td><td>${L.max}</td>` +
        `<td>${pct(L.pct_over_200)}</td><td>${pct(L.pct_over_256)}</td></tr>`;
    }).join("");
    document.getElementById("length-table").innerHTML = head + "<tbody>" + rows + "</tbody>";
    document.getElementById("stats-tok").textContent = D.length_tokenizer;
    document.getElementById("stats-cap").textContent = D.length_cap;
    document.getElementById("stats-ask").textContent = D.length_prompt_ask;
    document.getElementById("length-foot").textContent =
      `Reflection text only (excludes the analysis scratchpad and JSON structure), over the 4k benchmark. ` +
      `Median reflections sit well under the ${D.length_cap}-token cutoff; the prompt asks for ≤${D.length_prompt_ask}, ` +
      `which kept lengths down, and only a thin tail approaches the cutoff.`;
  }

  // ============================ CHARTER CITATIONS ============================
  function renderCitations() {
    const traces = D.models.map((m) => ({
      x: D.citation_buckets, y: m.citations.dist_pct,
      name: m.label, type: "bar", marker: { color: m.color },
      hovertemplate: `<b>${m.label}</b><br>%{x} citation(s): %{y:.1f}% of reflections<extra></extra>`,
    }));
    const layout = baseLayout({
      barmode: "group", bargap: 0.3, bargroupgap: 0.08,
      margin: { l: 56, r: 20, t: 24, b: 46 },
      xaxis: Object.assign(baseLayout().xaxis, { showgrid: false, title: { text: "Charter [X.Y] citations per reflection", font: { size: 12.5 } } }),
      yaxis: Object.assign(baseLayout().yaxis, { title: { text: "% of reflections", font: { size: 12.5 } }, rangemode: "tozero" }),
    });
    Plotly.react("plot-citations", traces, layout, CONFIG);
    document.getElementById("cite-sub").textContent =
      "How many charter [X.Y] citations each reflection carries (grouped brackets counted individually), over the 4k benchmark.";
    document.getElementById("cite-desc").innerHTML =
      `<strong>What we see.</strong> About 40% of reflections cite nothing — the honest "nothing at stake" response on benign text — ` +
      `while the rest cite one to several charter elements. Mean citations per reflection: ` +
      D.models.map((m) => `<strong>${esc(m.label)}</strong> ${m.citations.mean}`).join(", ") +
      `; Gemma-4-31B engages the charter most densely, Qwen3.6 the most sparingly.`;
  }

  // ============================ INSPECTOR ============================
  let INSP = null;               // full per-generation data, loaded lazily from inspector.js
  let inspLoading = false;
  const RENDER_CAP = 400;

  function initInspector() {
    const selModel = document.getElementById("sel-model");
    const selLang = document.getElementById("sel-lang");
    const selSafety = document.getElementById("sel-safety");
    D.models.forEach((m) => selModel.add(new Option(m.label, m.id)));
    selLang.add(new Option("All languages", "all"));
    D.lang_order.forEach((l) => selLang.add(new Option(D.lang_label[l], l)));
    selSafety.add(new Option("Any", "all"));
    (D.safety_order || []).slice().reverse().forEach((s) =>
      selSafety.add(new Option(D.safety_label[String(s)] || ("Safety " + s), String(s))));
    ["sel-model", "sel-lang", "sel-filter", "sel-safety", "sel-cites"].forEach((id) =>
      document.getElementById(id).addEventListener("change", renderList));
    let t;
    document.getElementById("sel-search").addEventListener("input", () => { clearTimeout(t); t = setTimeout(renderList, 180); });
  }

  function ensureInspector(cb) {
    if (INSP) { cb(); return; }
    if (inspLoading) return;
    inspLoading = true;
    const s = document.createElement("script");
    s.src = "inspector.js";
    s.onload = () => { INSP = (window.INSPECTOR || {}).records || []; inspLoading = false; cb(); };
    s.onerror = () => {
      inspLoading = false;
      document.getElementById("record-detail").innerHTML =
        `<p class="empty-hint">Could not load inspector.js — serve the page over http (e.g. <code>python -m http.server -d docs</code>).</p>`;
    };
    document.head.appendChild(s);
  }

  function currentRecords() {
    const model = document.getElementById("sel-model").value;
    const lang = document.getElementById("sel-lang").value;
    const verdict = document.getElementById("sel-filter").value;
    const safety = document.getElementById("sel-safety").value;
    const cites = document.getElementById("sel-cites").value;
    const q = document.getElementById("sel-search").value.trim().toLowerCase();
    return INSP.filter((r) =>
      r.model === model &&
      (lang === "all" || r.language === lang) &&
      (verdict === "all" || (verdict === "accept" ? r.accept : !r.accept)) &&
      (safety === "all" || String(r.safety_score) === safety) &&
      (cites === "all" || (cites === "3" ? r.n_cites >= 3 : r.n_cites === +cites)) &&
      (!q || (r.reflection_1p || "").toLowerCase().includes(q) || (r.text || "").toLowerCase().includes(q)));
  }

  function renderList() {
    if (!INSP) { ensureInspector(renderList); return; }
    const list = document.getElementById("record-list");
    const all = currentRecords();
    const rows = all.slice(0, RENDER_CAP);
    document.getElementById("list-count").textContent = all.length > RENDER_CAP
      ? `showing first ${RENDER_CAP} of ${all.length.toLocaleString()} matching — refine to narrow`
      : `${all.length.toLocaleString()} matching`;
    if (!rows.length) {
      list.innerHTML = `<li class="list-empty">No generations match these filters.</li>`;
      document.getElementById("record-detail").innerHTML = `<p class="empty-hint">No matches.</p>`;
      return;
    }
    list.innerHTML = rows.map((r, i) => {
      const snip = (r.reflection_1p || r.text || "").slice(0, 150);
      return `<li class="record-row" data-i="${i}">
        <div class="row-top">
          <span class="badge lang">${esc(D.lang_label[r.language] || r.language)}</span>
          <span class="badge ${r.accept ? "accept" : "reject"}">${r.accept ? "accept" : "reject"}</span>
          <span class="badge score">agg ${r.aggregate.toFixed(1)}</span>
          <span class="badge score">${r.n_cites} cite${r.n_cites === 1 ? "" : "s"}</span>
        </div>
        <div class="snippet">${esc(snip)}</div>
      </li>`;
    }).join("");
    Array.from(list.querySelectorAll(".record-row")).forEach((el) => {
      el.addEventListener("click", () => {
        list.querySelectorAll(".record-row").forEach((x) => x.classList.remove("is-active"));
        el.classList.add("is-active");
        renderDetail(rows[+el.dataset.i]);
      });
    });
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
    document.getElementById("record-detail").innerHTML =
      `<div class="detail-head">
         <h3>${esc(D.lang_label[s.language] || s.language)}</h3>
         <span class="badge ${s.accept ? "accept" : "reject"}">${s.accept ? "accepted" : "rejected"}</span>
       </div>
       <div class="detail-meta">item ${esc(String(s.item_id).slice(0, 28))} · safety ${s.safety_score ?? "—"} · ${s.n_cites} charter citation${s.n_cites === 1 ? "" : "s"}</div>
       <div class="scorebar">
         ${scoreChip("relevance", "Relevance", sc.relevance)}
         ${scoreChip("specificity", "Specificity", sc.specificity)}
         ${scoreChip("charter_grounding", "Charter", sc.charter_grounding)}
         ${scoreChip("voice_tone", "Voice/Tone", sc.voice_tone)}
         ${scoreChip("aggregate", "Aggregate", s.aggregate.toFixed(2), true)}
       </div>
       <div class="block"><h4>Reflection (model output)</h4><div class="reflection-box">${esc(s.reflection_1p)}</div></div>
       <div class="block"><h4>Judge verdict</h4><div class="reasoning-box">${esc(s.reasoning)}</div></div>
       <div class="block"><h4>Source document</h4><div class="doc-text">${docWithMarker(s.text, s.reflection_point, s.text_truncated)}</div></div>`;
  }

  // ============================ PROMPTS ============================
  let promptModelId = D.models[0].id;
  function renderPrompts() {
    const tabs = document.getElementById("prompt-tabs");
    tabs.innerHTML = D.models.map((m) =>
      `<button class="prompt-pill ${m.id === promptModelId ? "is-active" : ""}" data-id="${m.id}">
         <span class="swatch" style="background:${m.color}"></span>${esc(m.label)} &middot; ${esc(m.prompt)}
       </button>`).join("");
    tabs.querySelectorAll(".prompt-pill").forEach((el) =>
      el.addEventListener("click", () => { promptModelId = el.dataset.id; renderPrompts(); }));
    const m = D.models.find((x) => x.id === promptModelId);
    document.getElementById("prompt-path").textContent = m.prompt_file;
    document.getElementById("prompt-body").textContent = m.prompt_text;
  }

  // ============================ REVIEWER–JUDGE AGREEMENT ============================
  function renderAgreement() {
    const a = D.agreement;
    if (!a) return;
    document.getElementById("agree-sub").textContent =
      `Each of ${a.n} human accept/reject thumbs vs. the ${a.judge_model} verdict (${a.judge_prompt}) re-run on the exact reviewed reflection.`;
    document.getElementById("agree-pct").textContent = (a.agreement * 100).toFixed(1) + "%";
    document.getElementById("agree-lab").textContent = `agreement · ${Math.round(a.agreement * a.n)}/${a.n} reviews`;
    document.getElementById("agree-prev").textContent = `previous judge (${a.old_judge_prompt}): ${(a.agreement_old * 100).toFixed(1)}%`;

    const c = a.confusion;
    const cell = (v, ok) => `<td class="cm-cell ${ok ? "cm-agree" : "cm-dis"}">${v}</td>`;
    document.getElementById("confusion").innerHTML =
      `<thead><tr><th></th><th>Judge: accept</th><th>Judge: reject</th></tr></thead><tbody>` +
      `<tr><th>Human: accept</th>${cell(c.aa, true)}${cell(c.ar, false)}</tr>` +
      `<tr><th>Human: reject</th>${cell(c.ra, false)}${cell(c.rr, true)}</tr></tbody>`;

    const dis = c.ar + c.ra;
    document.getElementById("agree-desc").innerHTML =
      `<strong>What we see.</strong> The judge matches human reviewers on <strong>${(a.agreement * 100).toFixed(1)}%</strong> of ` +
      `${a.n} reviews (up from ${(a.agreement_old * 100).toFixed(1)}% for the previous ${a.old_judge_prompt}). All ${dis} disagreements ` +
      `run the same way: the <strong>judge is stricter</strong> — it rejected ${c.ar} reflections the reviewers accepted, and never ` +
      `accepted one a reviewer rejected (${c.ra}). So on this set it has perfect precision on rejects and errs conservatively. ` +
      `The reviews span ${a.n_reviewers} reviewers and were collected on an earlier generation set (qwen3.5 / gemma v5–v6), so this ` +
      `gauges judge calibration rather than the final generators.`;

    renderReviewList();
  }

  function currentReviews() {
    const f = document.getElementById("rev-filter").value;
    return D.reviews.filter((r) => f === "all" || (f === "agree" ? r.agree : !r.agree));
  }

  function renderReviewList() {
    const list = document.getElementById("review-list");
    const rows = currentReviews();
    if (!rows.length) {
      list.innerHTML = `<li class="list-empty">No reviews for this filter.</li>`;
      document.getElementById("review-detail").innerHTML = `<p class="empty-hint">None.</p>`;
      return;
    }
    list.innerHTML = rows.map((r, i) => {
      const snip = (r.reflection_1p || r.text || "").slice(0, 150);
      const hj = `H:${r.human_verdict[0].toUpperCase()} · J:${r.judge_decision[0].toUpperCase()}`;
      return `<li class="record-row" data-i="${i}">
        <div class="row-top">
          <span class="badge lang">${esc(D.lang_label[r.language] || r.language)}</span>
          <span class="badge ${r.agree ? "accept" : "reject"}">${r.agree ? "agree" : "disagree"}</span>
          <span class="badge score">${hj}</span>
        </div>
        <div class="snippet">${esc(snip)}</div>
      </li>`;
    }).join("");
    Array.from(list.querySelectorAll(".record-row")).forEach((el) => {
      el.addEventListener("click", () => {
        list.querySelectorAll(".record-row").forEach((x) => x.classList.remove("is-active"));
        el.classList.add("is-active");
        renderReviewDetail(rows[+el.dataset.i]);
      });
    });
    list.querySelector(".record-row").classList.add("is-active");
    renderReviewDetail(rows[0]);
  }

  function renderReviewDetail(r) {
    const sc = r.judge_scores || {};
    const agg = (typeof r.judge_aggregate === "number") ? r.judge_aggregate.toFixed(2) : "—";
    const html =
      `<div class="detail-head">
         <h3>${esc(D.lang_label[r.language] || r.language)}</h3>
         <span class="badge ${r.agree ? "accept" : "reject"}">${r.agree ? "agree" : "disagree"}</span>
       </div>
       <div class="detail-meta">${esc(r.gen_alias)} · safety ${r.safety_score ?? "—"} · reviewer ${esc(r.human_reviewer || "—")}</div>
       <div class="verdict-row">
         <div class="verdict-box"><span class="vk">Human</span><span class="badge ${r.human_verdict === "accept" ? "accept" : "reject"}">${r.human_verdict}</span></div>
         <div class="verdict-box"><span class="vk">Judge</span><span class="badge ${r.judge_decision === "accept" ? "accept" : "reject"}">${r.judge_decision}</span><span class="badge score">agg ${agg}</span></div>
       </div>
       <div class="scorebar">
         ${scoreChip("relevance", "Relevance", sc.relevance)}
         ${scoreChip("specificity", "Specificity", sc.specificity)}
         ${scoreChip("charter_grounding", "Charter", sc.charter_grounding)}
         ${scoreChip("voice_tone", "Voice/Tone", sc.voice_tone)}
         ${scoreChip("aggregate", "Aggregate", agg, true)}
       </div>
       <div class="block"><h4>Reflection</h4><div class="reflection-box">${esc(r.reflection_1p)}</div></div>` +
      (r.human_reason ? `<div class="block"><h4>Reviewer note</h4><div class="reasoning-box">${esc(r.human_reason)}</div></div>` : "") +
      `<div class="block"><h4>Judge verdict</h4><div class="reasoning-box">${esc(r.judge_reasoning)}</div></div>
       <div class="block"><h4>Source document</h4><div class="doc-text">${docWithMarker(r.text, r.reflection_point, r.text_truncated)}</div></div>
       <div class="block"><details class="analysis"><summary>Model analysis (scratchpad)</summary><div class="analysis-box">${esc(r.analysis)}</div></details></div>`;
    document.getElementById("review-detail").innerHTML = html;
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
        document.getElementById("tab-stats").hidden = tab !== "stats";
        document.getElementById("tab-agreement").hidden = tab !== "agreement";
        document.getElementById("tab-prompts").hidden = tab !== "prompts";
        if (tab === "overview") {
          ["plot-headline", "plot-langs", "plot-safety"].forEach((p) => Plotly.Plots.resize(p));
        }
        if (tab === "stats") { Plotly.Plots.resize("plot-length"); Plotly.Plots.resize("plot-citations"); }
        if (tab === "inspector") renderList();
      });
    });

    const revFilter = document.getElementById("rev-filter");
    if (revFilter) revFilter.addEventListener("change", renderReviewList);

    document.getElementById("prompt-copy").addEventListener("click", () => {
      const m = D.models.find((x) => x.id === promptModelId);
      const btn = document.getElementById("prompt-copy");
      if (navigator.clipboard) navigator.clipboard.writeText(m.prompt_text);
      btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy"), 1200);
    });
    function wireToggle(id, cb) {
      const grp = document.getElementById(id);
      grp.addEventListener("click", (e) => {
        const b = e.target.closest(".toggle-btn"); if (!b) return;
        grp.querySelectorAll(".toggle-btn").forEach((x) => x.classList.remove("is-active"));
        b.classList.add("is-active");
        cb(b.dataset.metric);
      });
    }
    wireToggle("metric-toggle", (m) => { langMetric = m; renderLangs(); });
    wireToggle("headline-toggle", (m) => { headlineMetric = m; renderHeadline(); });
    wireToggle("safety-toggle", (m) => { safetyMetric = m; renderSafety(); });
  }

  function initMeta() {
    document.getElementById("meta-judge").textContent = D.judge_model;
    document.getElementById("meta-foot").textContent =
      `Reflection annotation pipeline · ${D.models.length} candidate generators · judged by ${D.judge_model}.`;
  }

  // ---- go ----
  initMeta();
  initTabs();
  renderHeadline();
  headlineDesc();
  renderLangs();
  langsDesc();
  renderSafety();
  safetyDesc();
  renderTable();
  renderLength();
  renderCitations();
  initInspector();
  renderPrompts();
  if (D.agreement) renderAgreement();
  else document.querySelector('.tab[data-tab="agreement"]').hidden = true;
})();
