/* BFF-15 results page — renders entirely from ./data/*.json (no external libs). */
"use strict";

const $ = (id) => document.getElementById(id);
const pct = (x, d = 1) => (x == null ? "–" : (x * 100).toFixed(d) + "%");
const num = (x) => (x == null ? "–" : Number(x).toLocaleString());
const COLORS = {
  accTrain: "#38bdf8", accVal: "#34d399",
  lossTrain: "#f97316", lossVal: "#f87171",
  cnn: "#f97316", tl: "#38bdf8",
};

async function loadJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(path + " -> HTTP " + r.status);
  return r.json();
}

/* ---------- inline SVG line chart ---------- */
function lineChart(title, series, yFmt) {
  const W = 480, H = 250, padL = 46, padR = 14, padT = 14, padB = 30;
  const xs = [], ys = [];
  series.forEach((s) => s.points.forEach(([x, y]) => { xs.push(x); ys.push(y); }));
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  let yMin = Math.min(...ys), yMax = Math.max(...ys);
  if (yMin === yMax) { yMin -= 0.1; yMax += 0.1; }
  const sx = (x) => padL + ((x - xMin) / (xMax - xMin || 1)) * (W - padL - padR);
  const sy = (y) => H - padB - ((y - yMin) / (yMax - yMin || 1)) * (H - padT - padB);

  let grid = "", ticks = "";
  for (let i = 0; i <= 4; i++) {
    const yv = yMin + (i / 4) * (yMax - yMin), py = sy(yv);
    grid += `<line x1="${padL}" y1="${py}" x2="${W - padR}" y2="${py}" stroke="#263746" stroke-width="1"/>`;
    ticks += `<text x="${padL - 6}" y="${py + 3}" fill="#9fb0c0" font-size="10" text-anchor="end">${yFmt(yv)}</text>`;
  }
  let paths = "";
  series.forEach((s) => {
    const d = s.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(" ");
    paths += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    s.points.forEach(([x, y]) => { paths += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="2.2" fill="${s.color}"/>`; });
  });
  const legX = series.map((s, i) =>
    `<tspan fill="${s.color}" dx="${i ? 16 : 0}">■ ${s.name}</tspan>`).join("");
  return `<div class="chart">
    <div style="font-size:.8rem;color:#9fb0c0;margin-bottom:2px"><b>${title}</b>
      <span style="float:right">${series.map((s) => `<span style="color:${s.color}">■</span> ${s.name}`).join(" ")}</span></div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${title}">${grid}${paths}${ticks}
      <text x="${padL}" y="${H - 8}" fill="#9fb0c0" font-size="10">${xMin}</text>
      <text x="${W - padR}" y="${H - 8}" fill="#9fb0c0" font-size="10" text-anchor="end">${xMax}</text>
      <text x="${(padL + W - padR) / 2}" y="${H - 8}" fill="#9fb0c0" font-size="10" text-anchor="middle">epoch</text>
    </svg></div>`;
}

function curvesPanel(model) {
  const h = model.history;
  const ep = h.epoch;
  const zip = (arr) => arr.map((v, i) => [ep[i], v]);
  const acc = lineChart("Accuracy", [
    { name: "train", color: COLORS.accTrain, points: zip(h.accuracy) },
    { name: "val", color: COLORS.accVal, points: zip(h.val_accuracy) },
  ], (y) => y.toFixed(2));
  const loss = lineChart("Loss", [
    { name: "train", color: COLORS.lossTrain, points: zip(h.loss) },
    { name: "val", color: COLORS.lossVal, points: zip(h.val_loss) },
  ], (y) => y.toFixed(2));
  return acc + loss;
}

/* ---------- grouped bar: per-class F1 ---------- */
function f1Bar(cnn, tl, classes) {
  const W = 940, H = 260, padL = 40, padR = 12, padT = 12, padB = 60;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const group = plotW / classes.length;
  const barW = Math.min(14, group / 2 - 3);
  const sy = (v) => padT + (1 - v) * plotH;
  let bars = "", xlab = "", grid = "";
  for (let i = 0; i <= 4; i++) {
    const v = i / 4, py = sy(v);
    grid += `<line x1="${padL}" y1="${py}" x2="${W - padR}" y2="${py}" stroke="#263746"/>`;
    grid += `<text x="${padL - 5}" y="${py + 3}" fill="#9fb0c0" font-size="10" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  classes.forEach((c, i) => {
    const cx = padL + group * i + group / 2;
    const cv = (cnn[c] && cnn[c].f1) || 0, tv = (tl[c] && tl[c].f1) || 0;
    bars += `<rect x="${cx - barW - 1}" y="${sy(cv)}" width="${barW}" height="${plotH - (sy(cv) - padT)}" fill="${COLORS.cnn}"/>`;
    bars += `<rect x="${cx + 1}" y="${sy(tv)}" width="${barW}" height="${plotH - (sy(tv) - padT)}" fill="${COLORS.tl}"/>`;
    xlab += `<text x="${cx}" y="${H - padB + 12}" fill="#9fb0c0" font-size="9" text-anchor="end" transform="rotate(-55 ${cx} ${H - padB + 12})">${c}</text>`;
  });
  const leg = `<span style="color:${COLORS.cnn}">■</span> Custom CNN &nbsp; <span style="color:${COLORS.tl}">■</span> MobileNetV2`;
  return `<div class="chart"><div style="font-size:.82rem;color:#9fb0c0;margin-bottom:6px">${leg}</div>
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Per-class F1">${grid}${bars}${xlab}</svg></div>`;
}

/* ---------- confusion heatmap ---------- */
function heatmap(model, classes) {
  const m = model.confusion_matrix;
  const rowTot = m.map((r) => r.reduce((a, b) => a + b, 0));
  let html = '<table class="heat"><thead><tr><th></th>';
  classes.forEach((c) => { html += `<th class="collab"><span>${c}</span></th>`; });
  html += "</tr></thead><tbody>";
  m.forEach((row, i) => {
    html += `<tr><th class="rowlab" title="true: ${classes[i]}">${classes[i]}</th>`;
    row.forEach((v, j) => {
      const frac = rowTot[i] ? v / rowTot[i] : 0;
      const diag = i === j;
      const bg = v === 0 ? "transparent"
        : `rgba(56,189,248,${(0.10 + 0.85 * frac).toFixed(3)})`;
      const fg = frac > 0.55 ? "#08111a" : "#e8eef4";
      const outline = diag ? "outline:1px solid rgba(52,211,153,.6);outline-offset:-1px;" : "";
      html += `<td class="cell" style="background:${bg};color:${fg};${outline}" title="true ${classes[i]} → pred ${classes[j]}: ${v}">${v || ""}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table><div style='font-size:.72rem;color:#9fb0c0;margin-top:6px'>rows = true class, columns = predicted; colour = share of the row.</div>";
  return html;
}

/* ---------- comparison table ---------- */
function comparisonRows(cnn, tl) {
  const rows = [
    ["Test accuracy", "test_accuracy", 1],
    ["Top-5 accuracy", "test_top5_accuracy", 1],
    ["ROC-AUC (one-vs-rest)", "test_roc_auc_ovr", 1],
    ["Precision (macro)", "test_precision_macro", 1],
    ["Recall (macro)", "test_recall_macro", 1],
    ["F1 (macro)", "test_f1_macro", 1],
    ["Precision (weighted)", "test_precision_weighted", 1],
    ["Recall (weighted)", "test_recall_weighted", 1],
    ["F1 (weighted)", "test_f1_weighted", 1],
    ["Test loss", "test_loss", 3],
  ];
  let html = "<thead><tr><th>Metric</th><th>Custom CNN</th><th>MobileNetV2</th><th>Δ (TL − CNN)</th></tr></thead><tbody>";
  rows.forEach(([label, key, d]) => {
    const a = cnn[key], b = tl[key];
    const delta = (a != null && b != null) ? b - a : null;
    const cls = delta == null ? "" : (delta > 0 ? "pos" : "neg");
    const sign = delta > 0 ? "+" : "";
    html += `<tr><td>${label}</td><td>${d === 3 ? (a ?? 0).toFixed(3) : pct(a, 1)}</td>` +
            `<td>${d === 3 ? (b ?? 0).toFixed(3) : pct(b, 1)}</td>` +
            `<td class="${cls}">${delta == null ? "–" : sign + (d === 3 ? delta.toFixed(3) : (delta * 100).toFixed(1) + "pp")}</td></tr>`;
  });
  const extra = [
    ["Epochs run", cnn.epochs_ran, tl.epochs_ran],
    ["Train time", (cnn.train_seconds / 60).toFixed(1) + " min", (tl.train_seconds / 60).toFixed(1) + " min"],
    ["Total params", num(cnn.total_params), num(tl.total_params)],
    ["Trainable params", num(cnn.trainable_params), num(tl.trainable_params)],
  ];
  extra.forEach(([l, a, b]) => { html += `<tr><td>${l}</td><td>${a}</td><td>${b}</td><td>–</td></tr>`; });
  return html + "</tbody>";
}

/* ---------- per-class table ---------- */
function perclassTable(cnnPC, tlPC, classes) {
  let html = "<thead><tr><th>Class</th><th colspan=3>Custom CNN</th><th colspan=3>MobileNetV2</th><th>Support</th></tr>" +
    "<tr><th></th><th>P</th><th>R</th><th>F1</th><th>P</th><th>R</th><th>F1</th><th></th></tr></thead><tbody>";
  classes.forEach((c) => {
    const a = cnnPC[c] || {}, b = tlPC[c] || {};
    html += `<tr><td>${c}</td><td>${pct(a.precision, 0)}</td><td>${pct(a.recall, 0)}</td><td>${pct(a.f1, 0)}</td>` +
      `<td>${pct(b.precision, 0)}</td><td>${pct(b.recall, 0)}</td><td>${pct(b.f1, 0)}</td><td>${num(b.support)}</td></tr>`;
  });
  return html + "</tbody>";
}

/* ---------- main ---------- */
async function main() {
  let R, M, S;
  try {
    [R, M, S] = await Promise.all([
      loadJSON("./data/results.json"),
      loadJSON("./data/mcnemar.json"),
      loadJSON("./data/final_summary.json"),
    ]);
  } catch (e) {
    $("headline").innerHTML = `<div class="panel callout" style="border-left-color:#f87171">
      Could not load result data (open the site over http(s), e.g. GitHub Pages or a local server).<br>${e.message}</div>`;
    return;
  }

  const cnn = R.models.custom_cnn, tl = R.models.transfer_learning;
  const classes = R.dataset.classes;

  // header
  $("title").textContent = R.project;
  $("team").textContent = R.team.join(", ");
  const d = R.dataset;
  $("badges").innerHTML = [
    `<span class="badge"><b>${classes.length}</b> classes</span>`,
    `<span class="badge"><b>${d.split.train + d.split.val + d.split.test}</b> unique images</span>`,
    `<span class="badge">${d.img_size[0]}×${d.img_size[1]}</span>`,
    `<span class="badge">${d.license}</span>`,
    `<span class="badge">TensorFlow / Keras</span>`,
  ].join("");
  $("src-link").href = d.source;

  // headline KPIs
  $("kpi-cnn").textContent = pct(cnn.metrics.test_accuracy, 1);
  $("kpi-tl").textContent = pct(tl.metrics.test_accuracy, 1);
  $("kpi-delta").textContent = "+" + ((tl.metrics.test_accuracy - cnn.metrics.test_accuracy) * 100).toFixed(1) + "pp";
  $("kpi-p").textContent = "p " + (M.p_value < 0.001 ? M.p_value.toExponential(1) : "= " + M.p_value.toFixed(4));

  // tables & charts
  $("cmp-table").innerHTML = comparisonRows(cnn.metrics, tl.metrics);
  $("cnn-curves").innerHTML = curvesPanel(cnn);
  $("tl-curves").innerHTML = curvesPanel(tl);
  $("perclass-chart").innerHTML = f1Bar(cnn.per_class, tl.per_class, classes);
  $("perclass-table").innerHTML = perclassTable(cnn.per_class, tl.per_class, classes);
  $("cnn-cm").innerHTML = heatmap(cnn, classes);
  $("tl-cm").innerHTML = heatmap(tl, classes);

  // significance
  $("sig-box").innerHTML = `<div class="big">McNemar's test: p = ${M.p_value.toExponential(2)}</div>
    <div class="detail">On the ${num(S.transfer_learning ? R.models.transfer_learning.metrics.n_test : "")}
    frozen test images, MobileNetV2 got <b>${M.c_tl_only}</b> cases right that the Custom CNN got wrong,
    while the CNN won only <b>${M.b_cnn_only}</b> such cases in reverse.
    χ² = ${M.chi2.toFixed(1)} (continuity-corrected), so the accuracy gap is statistically significant
    rather than chance.</div>`;

  // dataset overview
  $("dataset-info").innerHTML = `
    <div class="info-block"><h3>Split &amp; augmentation</h3>
      <dl><dt>Dataset</dt><dd>${d.name}</dd>
      <dt>License</dt><dd><a href="${d.source}" target="_blank" rel="noopener">${d.license}</a></dd>
      <dt>Train / Val / Test</dt><dd>${num(d.split.train)} / ${num(d.split.val)} / ${num(d.split.test)}</dd>
      <dt>Image size</dt><dd>${d.img_size[0]} × ${d.img_size[1]}</dd>
      <dt>Augmentation</dt><dd>${d.augmentation.map((a) => `<span class="chip">${a}</span>`).join("")}</dd></dl>
    </div>
    <div class="info-block"><h3>Classes (${classes.length})</h3>
      <div class="chips">${classes.map((c) => `<span class="chip">${c}</span>`).join("")}</div>
    </div>`;
}

main();
