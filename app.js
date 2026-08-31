(function () {
  "use strict";

  const bundle = window.SAF_MODEL_BUNDLE;
  const validation = window.SAF_VALIDATION;
  const core = window.SAFPredictorCore;
  if (!bundle || !core) {
    document.body.innerHTML = "<p style='padding:30px;font-family:Arial'>Model files could not be loaded. Keep the complete release directory intact and reopen index.html.</p>";
    return;
  }
  const predictor = core.createPredictor(bundle);
  const batchState = { rows: [], results: [], exportRows: [] };

  const modelLabels = {
    RF: "Random Forest",
    SVR: "Support vector regression",
    XGBoost: "XGBoost",
    "derived: predicted Y01 × predicted Y03": "Derived: Y01 × Y03",
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatNumber(value, digits) {
    if (!Number.isFinite(value)) return "—";
    return Number(value).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function setMessage(element, text, type = "info") {
    element.innerHTML = text ? `<div class="message message-${type}">${escapeHtml(text)}</div>` : "";
  }

  function renderInputs() {
    const grid = document.getElementById("input-grid");
    grid.innerHTML = bundle.input_order.map((code) => {
      const meta = bundle.input_meta[code];
      const range = bundle.input_ranges[code];
      return `
        <label class="field">
          <span><b>${code} · ${escapeHtml(meta.name)}</b><small>${escapeHtml(meta.unit)}</small></span>
          <input id="input-${code}" name="${code}" type="number" step="${meta.step}" placeholder="Development-set median: ${formatNumber(range.median, meta.step === 1 ? 0 : 3)}" autocomplete="off">
          <div class="field-help">${escapeHtml(meta.definition)}</div>
          <div class="range-line"><span>Central P5–P95: ${formatNumber(range.q05, 3)}–${formatNumber(range.q95, 3)}</span><span>Observed range: ${formatNumber(range.min, 3)}–${formatNumber(range.max, 3)}</span></div>
        </label>`;
    }).join("");
  }

  function collectSingleInput() {
    return Object.fromEntries(bundle.input_order.map((code) => [code, document.getElementById(`input-${code}`).value]));
  }

  function renderApplicability(result) {
    const ad = result.applicability;
    const color = "var(--teal)";
    const nearest = ad.nearest.map((row, index) => `
      <div class="neighbour"><strong>${index + 1}. ${escapeHtml(row.record_id)}</strong><span>AD distance = ${row.distance.toFixed(3)}</span></div>`).join("");
    document.getElementById("applicability-card").innerHTML = `
      <div class="ad-card">
        <div class="ad-score" style="--score:${ad.percentile.toFixed(1)}%; --teal:${color}">
          <strong>${ad.percentile.toFixed(1)}</strong><span>Applicability-domain (AD) distance percentile</span>
        </div>
        <div class="ad-copy">
          <p class="eyebrow">APPLICABILITY DOMAIN</p>
          <h2>${escapeHtml(ad.label)}</h2>
          <p>Nearest-neighbour AD distance: ${ad.distance.toFixed(4)}. The percentile is calculated against the development-set reference-distance distribution.</p>
        </div>
        <div><p class="eyebrow">NEAREST DE-IDENTIFIED DEVELOPMENT VECTORS</p><div class="neighbour-list">${nearest}</div></div>
      </div>`;
  }

  function evidenceLabel(target, counts) {
    const total = Object.values(counts).reduce((sum, value) => sum + value, 0);
    if (counts.constructed) return "constructed: 100%";
    const source = counts["source-supported"] || 0;
    const derived = counts["formula-derived"] || 0;
    return `source-supported: ${Math.round(1000 * source / total) / 10}%; formula-derived: ${Math.round(1000 * derived / total) / 10}%`;
  }

  function distanceBandLabel(band) {
    if (band === "at_or_above_95") return "≥95th percentile of the AD reference-distance distribution";
    if (band === "90_to_below_95") return "90th–<95th percentile of the AD reference-distance distribution";
    return "<90th percentile of the AD reference-distance distribution";
  }

  function propertyCard(target, output) {
    const meta = bundle.output_meta[target];
    const position = Math.max(0, Math.min(100, output.developmentRangePosition));
    const model = modelLabels[output.model] || output.model;
    const caution = target === "Y04"
      ? `<div class="property-caution"><strong>Y04 caution</strong><span>Y04 performed poorly on the internal holdout (R² = −0.021; MAE = 25.2 °C; n = 10). Do not use Y04 alone to rank or select compounds; experimental confirmation is required.</span></div>`
      : "";
    return `
      <article class="property-card">
        <div class="property-top"><span class="property-code">${target}</span><span class="property-model">${escapeHtml(model)}</span></div>
        <h3>${escapeHtml(meta.name)}</h3>
        <div><span class="property-value">${formatNumber(output.point, meta.digits)}</span><span class="property-unit">${escapeHtml(meta.unit)}</span></div>
        <div class="distribution" style="--position:${position}%"></div>
        <div class="evidence">Position relative to the development-set range (distribution reference only)</div>
        <div class="interval-row">
          <div><span>90% empirical prediction interval</span><strong>${formatNumber(output.lower90, meta.digits)} – ${formatNumber(output.upper90, meta.digits)}</strong></div>
          <div><span>95% empirical prediction interval</span><strong>${formatNumber(output.lower95, meta.digits)} – ${formatNumber(output.upper95, meta.digits)}</strong></div>
          <div><span>Author curation categories in the development labels</span><strong>${escapeHtml(evidenceLabel(target, output.provenance))}</strong></div>
        </div>
        ${caution}
      </article>`;
  }

  function renderSingleResult(result) {
    renderApplicability(result);
    const groups = [
      { title: "Energy and density", subtitle: "Combustion-energy and volumetric properties", targets: ["Y01", "Y03", "Y02"] },
      { title: "Phase transition and volatility", subtitle: "Thermal-transition properties", targets: ["Y04", "Y05", "Y06"] },
      { title: "Flow behaviour", subtitle: "Viscosity at 40 °C", targets: ["Y07"] },
    ];
    document.getElementById("prediction-groups").innerHTML = groups.map((group) => `
      <section class="property-section">
        <div class="property-section-title"><h2>${group.title}</h2><span>${group.subtitle}</span></div>
        <div class="property-grid">${group.targets.map((target) => propertyCard(target, result.outputs[target])).join("")}</div>
      </section>`).join("");
    const warnings = result.warnings.length
      ? result.warnings.map((warning) => `<div class="message message-warning">${escapeHtml(warning)}</div>`).join("")
      : `<div class="message message-info">No range or consistency warnings were detected. Experimental confirmation is still required.</div>`;
    document.getElementById("prediction-warnings").innerHTML = `<div class="warning-list">${warnings}</div>`;
    document.getElementById("single-results").classList.remove("is-hidden");
  }

  function setupSingle() {
    document.getElementById("fill-median").addEventListener("click", () => {
      bundle.input_order.forEach((code) => { document.getElementById(`input-${code}`).value = bundle.input_ranges[code].median; });
      document.getElementById("hydrocarbon-confirm").checked = true;
      setMessage(document.getElementById("single-message"), "Median values were loaded for interface testing; this combination may not describe a real molecule.", "info");
    });
    document.getElementById("single-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const slot = document.getElementById("single-message");
      if (!document.getElementById("hydrocarbon-confirm").checked) {
        setMessage(slot, "Confirm that the input is a neutral, single-component pure hydrocarbon with a fully specified structure.", "error");
        return;
      }
      const result = predictor.predictOne(collectSingleInput());
      if (!result.ok) {
        setMessage(slot, result.errors.join("; "), "error");
        document.getElementById("single-results").classList.add("is-hidden");
        return;
      }
      setMessage(slot, "Predictions were computed locally; no molecular data were uploaded.", "info");
      renderSingleResult(result);
      document.getElementById("single-results").scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function parseCSV(text) {
    const rows = [];
    let row = [];
    let field = "";
    let quoted = false;
    const source = text.replace(/^\uFEFF/, "");
    for (let i = 0; i < source.length; i += 1) {
      const char = source[i];
      if (quoted) {
        if (char === '"' && source[i + 1] === '"') { field += '"'; i += 1; }
        else if (char === '"') quoted = false;
        else field += char;
      } else if (char === '"') quoted = true;
      else if (char === ",") { row.push(field); field = ""; }
      else if (char === "\n") { row.push(field.replace(/\r$/, "")); rows.push(row); row = []; field = ""; }
      else field += char;
    }
    if (field.length || row.length) { row.push(field.replace(/\r$/, "")); rows.push(row); }
    const nonEmpty = rows.filter((cells) => cells.some((cell) => cell.trim() !== ""));
    if (nonEmpty.length < 2) throw new Error("The CSV contains no prediction rows.");
    const headers = nonEmpty[0].map((cell) => cell.trim());
    const missing = bundle.input_order.filter((code) => !headers.includes(code));
    if (missing.length) throw new Error(`Missing required columns: ${missing.join(", ")}`);
    return nonEmpty.slice(1).map((cells, index) => {
      const object = { __row: index + 2 };
      headers.forEach((header, col) => { object[header] = cells[col] ?? ""; });
      return object;
    });
  }

  function previewBatch() {
    const host = document.getElementById("batch-preview");
    const rows = batchState.rows.slice(0, 5);
    host.innerHTML = `
      <p class="eyebrow" style="margin-top:18px">PREVIEW · ${batchState.rows.length} ROWS</p>
      <div class="table-scroll"><table><thead><tr><th>Row</th><th>Molecule</th>${bundle.input_order.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>${rows.map((row) => `<tr><td>${row.__row}</td><td>${escapeHtml(row.Molecule || row.Name || "")}</td>${bundle.input_order.map((c) => `<td>${escapeHtml(row[c])}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  }

  function csvEscape(value) {
    const text = String(value ?? "");
    return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }

  function makeCSV(rows) {
    if (!rows.length) return "";
    const headers = Object.keys(rows[0]);
    return "\uFEFF" + [headers.join(","), ...rows.map((row) => headers.map((h) => csvEscape(row[h])).join(","))].join("\r\n");
  }

  function downloadText(filename, text, type) {
    const blob = new Blob([text], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = filename; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function runBatch() {
    const results = predictor.predictBatch(batchState.rows);
    batchState.results = results;
    let valid = 0, invalid = 0, atOrAbove95 = 0, warningRows = 0;
    batchState.exportRows = batchState.rows.map((row, index) => {
      const result = results[index];
      const base = { Row: row.__row, Molecule: row.Molecule || row.Name || "", Formula: row.Formula || row["Molecular formula"] || "", ...Object.fromEntries(bundle.input_order.map((code) => [code, row[code]])) };
      if (!result.ok) { invalid += 1; return { ...base, Status: "ERROR", Errors: result.errors.join(" | ") }; }
      valid += 1;
      if (result.applicability.band === "at_or_above_95") atOrAbove95 += 1;
      if (result.warnings.length) warningRows += 1;
      const output = { ...base, Status: distanceBandLabel(result.applicability.band), AD_distance_percentile: result.applicability.percentile, AD_distance: result.applicability.distance, Nearest_development_record: result.applicability.nearest[0].record_id };
      for (const target of bundle.target_order) {
        const value = result.outputs[target];
        output[`${target}_pred`] = value.point;
        output[`${target}_lower90`] = value.lower90;
        output[`${target}_upper90`] = value.upper90;
        output[`${target}_lower95`] = value.lower95;
        output[`${target}_upper95`] = value.upper95;
      }
      output.Warnings = result.warnings.join(" | ");
      return output;
    });
    document.getElementById("batch-summary").innerHTML = `
      <div class="metric"><strong>${batchState.rows.length}</strong><span>Input rows</span></div>
      <div class="metric"><strong>${valid}</strong><span>Successful predictions</span></div>
      <div class="metric"><strong>${atOrAbove95}</strong><span>At or above the 95th applicability-domain (AD) distance percentile</span></div>
      <div class="metric ${invalid ? "bad" : ""}"><strong>${invalid}</strong><span>Invalid rows; ${warningRows} rows with warnings</span></div>`;
    const display = batchState.exportRows.slice(0, 100);
    document.getElementById("batch-table").innerHTML = `
      <thead><tr><th>Row</th><th>Molecule</th><th>Status</th><th>Applicability-domain (AD) distance percentile</th>${bundle.target_order.map((t) => `<th>${t} prediction</th>`).join("")}<th>Warnings/errors</th></tr></thead>
      <tbody>${display.map((row) => `<tr><td>${row.Row}</td><td>${escapeHtml(row.Molecule)}</td><td>${escapeHtml(row.Status)}</td><td>${formatNumber(Number(row.AD_distance_percentile),1)}</td>${bundle.target_order.map((t) => `<td>${formatNumber(Number(row[`${t}_pred`]), bundle.output_meta[t].digits)}</td>`).join("")}<td>${escapeHtml(row.Warnings || row.Errors || "")}</td></tr>`).join("")}</tbody>`;
    document.getElementById("batch-results").classList.remove("is-hidden");
  }

  function setupBatch() {
    document.getElementById("batch-file").addEventListener("change", async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      document.getElementById("batch-file-name").textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
      try {
        batchState.rows = parseCSV(await file.text());
        previewBatch();
        document.getElementById("run-batch").disabled = false;
        setMessage(document.getElementById("batch-message"), `Loaded ${batchState.rows.length} rows. Select “Run batch prediction” to continue.`, "info");
      } catch (error) {
        batchState.rows = [];
        document.getElementById("run-batch").disabled = true;
        setMessage(document.getElementById("batch-message"), error.message, "error");
      }
    });
    document.getElementById("run-batch").addEventListener("click", runBatch);
    document.getElementById("download-batch").addEventListener("click", () => {
      downloadText("SAF_Predict_batch_results.csv", makeCSV(batchState.exportRows), "text/csv;charset=utf-8");
    });
  }

  function renderValidation() {
    if (!validation) return;
    const metrics = validation.metrics;
    const weak = metrics.filter((row) => row.r2 < 0);
    document.getElementById("validation-overview").innerHTML = `
      <div class="metric"><strong>${validation.n_records}</strong><span>Internal holdout records</span></div>
      <div class="metric"><strong>${validation.unique_formulae}</strong><span>Unique molecular formulae</span></div>
      <div class="metric"><strong>${validation.records_with_formula_seen_in_development}/10</strong><span>Formula seen in development set</span></div>
      <div class="metric"><strong>${formatNumber(validation.ood.median_distance_percentile, 1)}</strong><span>Median applicability-domain (AD) distance percentile</span></div>`;
    document.getElementById("validation-alert").innerHTML = weak.length
      ? `<div class="message message-warning">Y04 performed poorly on the internal holdout (R² = −0.021; MAE = 25.2 °C; n = 10). Do not use Y04 alone to rank or select compounds; experimental confirmation is required.</div>`
      : "";
    document.getElementById("validation-table").innerHTML = `
      <thead><tr><th>Property</th><th>Deployed model</th><th>R²</th><th>RMSE</th><th>MAE</th><th>MedAE</th><th>Mean signed error</th><th>90% coverage</th><th>95% coverage</th></tr></thead>
      <tbody>${metrics.map((row) => `<tr>
        <td><strong>${row.target}</strong> · ${escapeHtml(row.property)}</td>
        <td>${escapeHtml(modelLabels[bundle.deployment_models[row.target]] || bundle.deployment_models[row.target])}</td>
        <td style="color:${row.r2 < 0 ? "var(--red)" : "inherit"};font-weight:800">${row.r2.toFixed(3)}</td>
        <td>${formatNumber(row.rmse, bundle.output_meta[row.target].digits)}</td>
        <td>${formatNumber(row.mae, bundle.output_meta[row.target].digits)}</td>
        <td>${formatNumber(row.median_absolute_error, bundle.output_meta[row.target].digits)}</td>
        <td>${formatNumber(row.mean_signed_error_actual_minus_predicted, bundle.output_meta[row.target].digits)}</td>
        <td>${row.coverage90_count}/${row.coverage90_n}</td><td>${row.coverage95_count}/${row.coverage95_n}</td>
      </tr>`).join("")}</tbody>`;
  }

  function renderAbout() {
    document.getElementById("model-mapping").innerHTML = bundle.target_order.map((target) => `
      <div class="model-row"><span>${target} · ${escapeHtml(bundle.output_meta[target].name)}</span><span>${escapeHtml(modelLabels[bundle.deployment_models[target]] || bundle.deployment_models[target])}</span></div>`).join("");
  }

  function setupTabs() {
    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("is-active", tab === button));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.toggle("is-active", panel.id === `tab-${button.dataset.tab}`));
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
    });
  }

  document.getElementById("version-pill").textContent = `v${bundle.version}`;
  renderInputs();
  setupTabs();
  setupSingle();
  setupBatch();
  renderValidation();
  renderAbout();
})();
