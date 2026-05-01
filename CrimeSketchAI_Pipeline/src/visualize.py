"""
visualize.py — CrimeSketch AI Pipeline
Reads results/metrics.json (produced by run_pipeline.py) and generates
a fully self-contained interactive HTML dashboard with Chart.js.

Every chart is data-driven from computed JSON — no hardcoded values.
"""

import json
import os
from typing import Dict


CATEGORIES_DISPLAY = {
    "frontal":   "Frontal Sketches",
    "semi":      "Semi-Profile",
    "low_qual":  "Low Quality",
    "high_det":  "High Detail",
    "composite": "Composite Tools",
    "synthetic": "Synthetic Styled",
}

CMC_LABELS = [f"R-{i}" for i in range(1, 21)]


def _color_for_model(name: str) -> str:
    name_l = name.lower()
    if "crimesketch" in name_l or "★" in name:
        return "#00e5ff"
    if any(x in name_l for x in ["arcface", "cosface", "magface", "adaface", "transface"]):
        return "#6bcb77"
    return "#8b949e"


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def generate_dashboard(results: Dict, output_path: str, run_info: Dict = None):
    """
    Generate HTML dashboard from computed results dict.
    results: {model_key: evaluate_model() output}
    """
    run_info = run_info or {}
    model_keys = [k for k in results if "error" not in results[k]
                  and "rank_accuracies" in results[k]]

    # Sort by Rank-1 ascending for charts
    model_keys.sort(key=lambda k: results[k]["rank_accuracies"].get("rank_1", 0))

    model_names = [results[k]["model_name"] for k in model_keys]
    rank1_vals  = [results[k]["rank_accuracies"].get("rank_1", 0) for k in model_keys]
    rank5_vals  = [results[k]["rank_accuracies"].get("rank_5", 0) for k in model_keys]
    latency_vals= [results[k]["latency"]["mean_ms"] for k in model_keys]
    auc_vals    = [results[k]["roc"]["auc"] for k in model_keys]
    ap_vals     = [results[k]["pr"]["ap"] for k in model_keys]
    colors      = [_color_for_model(n) for n in model_names]

    # Bar colours (fill / border)
    bar_fill   = [_rgba(c, 0.85 if c == "#00e5ff" else 0.65) for c in colors]
    bar_fill5  = [_rgba(c, 0.25) for c in colors]
    bar_border = colors

    # CMC datasets (select a subset for clarity)
    cmc_keys  = model_keys  # all models
    cmc_data_js = json.dumps([
        {
            "label": results[k]["model_name"],
            "data": results[k]["cmc_curve"],
            "borderColor": _color_for_model(results[k]["model_name"]),
            "borderWidth": 3 if "★" in results[k]["model_name"] else 1.5,
            "borderDash": [] if "★" in results[k]["model_name"] else [4, 3],
            "pointRadius": 3 if "★" in results[k]["model_name"] else 1.5,
            "tension": 0.35,
            "fill": False,
        }
        for k in cmc_keys
    ])

    # ROC datasets (top 4 models + best baseline)
    best_baseline = [k for k in model_keys if "★" not in results[k]["model_name"]]
    roc_keys = (best_baseline[-3:] if len(best_baseline) >= 3 else best_baseline) + \
               [k for k in model_keys if "★" in results[k]["model_name"]]
    roc_data_js = json.dumps([
        {
            "label": f"{results[k]['model_name']}  (AUC={results[k]['roc']['auc']})",
            "data": [{"x": round(f, 3), "y": round(t, 3)}
                     for f, t in zip(results[k]["roc"]["fpr"], results[k]["roc"]["tpr"])],
            "showLine": True,
            "borderColor": _color_for_model(results[k]["model_name"]),
            "backgroundColor": "transparent",
            "borderWidth": 3 if "★" in results[k]["model_name"] else 1.5,
            "borderDash": [] if "★" in results[k]["model_name"] else [3, 3],
            "pointRadius": 0,
            "tension": 0.4,
        }
        for k in roc_keys
    ])

    # PR datasets (same selection)
    pr_data_js = json.dumps([
        {
            "label": f"{results[k]['model_name']}  (AP={results[k]['pr']['ap']})",
            "data": [{"x": round(r, 3), "y": round(p, 3)}
                     for r, p in zip(results[k]["pr"]["recall"],
                                     results[k]["pr"]["precision"])],
            "showLine": True,
            "borderColor": _color_for_model(results[k]["model_name"]),
            "backgroundColor": "transparent",
            "borderWidth": 3 if "★" in results[k]["model_name"] else 1.5,
            "borderDash": [] if "★" in results[k]["model_name"] else [3, 3],
            "pointRadius": 0,
            "tension": 0.4,
        }
        for k in roc_keys
    ])

    # Scatter (accuracy vs latency)
    scatter_data_js = json.dumps([
        {
            "label": results[k]["model_name"],
            "data": [{"x": round(results[k]["latency"]["mean_ms"], 2),
                      "y": round(results[k]["rank_accuracies"].get("rank_1", 0), 2),
                      "r": max(4, (results[k]["rank_accuracies"].get("rank_5", 0) - 60) / 2.5)}],
            "backgroundColor": _rgba(_color_for_model(results[k]["model_name"]), 0.30),
            "borderColor": _color_for_model(results[k]["model_name"]),
            "borderWidth": 3 if "★" in results[k]["model_name"] else 1.5,
        }
        for k in model_keys
    ])

    # Category radar — CrimeSketch AI vs best baseline
    best_b_key = best_baseline[-1] if best_baseline else None
    cs_key     = next((k for k in model_keys if "★" in results[k]["model_name"]), None)
    cats = list(CATEGORIES_DISPLAY.keys())
    cat_labels = list(CATEGORIES_DISPLAY.values())
    radar_datasets = []
    for k, style in [(cs_key, "ours"), (best_b_key, "base")]:
        if k is None:
            continue
        pc = results[k].get("per_category", {})
        radar_datasets.append({
            "label": results[k]["model_name"],
            "data": [pc.get(c, 0) for c in cats],
            "borderColor": "#00e5ff" if style == "ours" else "#6bcb77",
            "backgroundColor": _rgba("#00e5ff" if style == "ours" else "#6bcb77",
                                     0.12 if style == "ours" else 0.06),
            "borderWidth": 2.5 if style == "ours" else 1.8,
            "pointRadius": 4 if style == "ours" else 3,
        })
    radar_js = json.dumps({"labels": cat_labels, "datasets": radar_datasets})

    # Distance distribution (CrimeSketch AI)
    dist_data = {}
    if cs_key:
        dist_data = results[cs_key].get("distance_dist", {})
    dist_js = json.dumps(dist_data)

    # Latency histogram (CrimeSketch AI)
    lat_hist = {}
    if cs_key:
        vals = results[cs_key]["latency"].get("values_ms", [])
        if vals:
            import numpy as np
            counts, edges = np.histogram(vals, bins=15)
            lat_hist = {
                "labels": [f"{(edges[i]+edges[i+1])/2:.1f}" for i in range(len(counts))],
                "data":   counts.tolist(),
                "threshold": results[cs_key]["latency"].get("mean_ms", 3),
            }
    lat_js = json.dumps(lat_hist)

    # Rank breakdown bars (CrimeSketch AI)
    rank_breakdown = {}
    if cs_key:
        rank_breakdown = {
            str(r): results[cs_key]["rank_accuracies"].get(f"rank_{r}", 0)
            for r in [1, 2, 3, 4, 5, 10, 20]
        }
    rank_js = json.dumps(rank_breakdown)

    # Ablation table data (from CrimeSketch AI result — computed from run)
    # The ablation data is stored in results["ablation"] if available
    ablation_js = json.dumps(results.get("ablation", {}))

    # Summary stats
    n_probe   = results[model_keys[0]]["n_probe"] if model_keys else 0
    n_gallery = results[model_keys[0]]["n_gallery"] if model_keys else 0
    dataset   = run_info.get("dataset_mode", "unknown")
    run_date  = run_info.get("run_date", "")

    # ── Generate HTML ─────────────────────────────────────────────────────

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CrimeSketch AI — IEEE FICV 2026 — Computed Results</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  :root{{--bg:#0d1117;--sf:#161b22;--bd:#30363d;--ac:#00e5ff;--ok:#6bcb77;--warn:#ffd93d;--bad:#ff6b6b;--tx:#e6edf3;--mt:#8b949e;}}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--tx);font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;}}
  header{{background:var(--sf);border-bottom:1px solid var(--bd);padding:14px 26px;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px;}}
  .badge{{background:var(--ac);color:#000;font-size:9px;font-weight:700;padding:2px 7px;border-radius:3px;margin-right:8px;}}
  .badge2{{background:var(--ok);color:#000;font-size:9px;font-weight:700;padding:2px 7px;border-radius:3px;}}
  h1{{font-size:16px;font-weight:700;color:var(--ac);margin:4px 0 2px;}}
  .sub{{font-size:11px;color:var(--mt);}}
  .computed-note{{font-size:11px;color:var(--ok);padding:3px 8px;background:rgba(107,203,119,0.08);border:1px solid rgba(107,203,119,0.3);border-radius:4px;margin-top:4px;}}
  .tab-bar{{display:flex;background:var(--sf);border-bottom:1px solid var(--bd);padding:0 26px;overflow-x:auto;}}
  .tab{{padding:9px 16px;font-size:12px;color:var(--mt);cursor:pointer;border-bottom:2px solid transparent;background:none;border-left:none;border-right:none;border-top:none;white-space:nowrap;}}
  .tab.active{{color:var(--ac);border-bottom-color:var(--ac);}}
  .panel{{display:none;padding:18px 26px 26px;}}
  .panel.active{{display:block;}}
  .metrics-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:0;}}
  .metric{{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:12px 14px;text-align:center;}}
  .metric .val{{font-size:22px;font-weight:700;color:var(--ac);line-height:1.1;}}
  .metric .lbl{{font-size:10px;color:var(--mt);margin-top:3px;}}
  .metric .delta{{font-size:10px;color:var(--ok);margin-top:2px;}}
  .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
  .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;}}
  .card{{background:var(--sf);border:1px solid var(--bd);border-radius:8px;padding:16px;}}
  .card.full{{grid-column:1/-1;}}
  .card.wide{{grid-column:span 2;}}
  .ct{{font-size:12px;font-weight:600;margin-bottom:2px;}}
  .cs{{font-size:10px;color:var(--mt);margin-bottom:12px;}}
  canvas{{width:100%!important;}}
  .bar-row{{display:flex;align-items:center;gap:8px;margin:5px 0;}}
  .bar-lbl{{font-size:11px;color:var(--mt);width:64px;text-align:right;}}
  .bar-bg{{flex:1;height:8px;background:rgba(255,255,255,0.04);border-radius:4px;overflow:hidden;}}
  .bar-f{{height:100%;border-radius:4px;background:linear-gradient(90deg,#0099cc,#00e5ff);}}
  .bar-val{{font-size:11px;font-weight:600;color:var(--ac);width:42px;}}
  table.mt{{width:100%;border-collapse:collapse;font-size:12px;}}
  table.mt th{{padding:8px 10px;text-align:left;color:var(--mt);font-weight:500;border-bottom:1px solid var(--bd);font-size:11px;}}
  table.mt td{{padding:8px 10px;border-bottom:1px solid rgba(48,54,61,0.4);}}
  .pill{{display:inline-block;padding:1px 6px;border-radius:8px;font-size:9px;font-weight:700;}}
  .pill.ours{{background:rgba(0,229,255,0.15);color:var(--ac);}}
  .pill.new{{background:rgba(107,203,119,0.15);color:var(--ok);}}
  .pill.base{{background:rgba(139,148,158,0.12);color:var(--mt);}}
  .bi{{display:flex;align-items:center;gap:6px;}}
  .bb{{flex:1;height:5px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;max-width:100px;}}
  .bf{{height:100%;border-radius:3px;}}
  .note{{font-size:11px;color:var(--mt);padding:8px 12px;background:rgba(0,229,255,0.05);border-left:3px solid var(--ac);border-radius:0 4px 4px 0;margin-bottom:16px;}}
  .rank-grid{{display:grid;grid-template-columns:repeat(7,1fr);gap:10px;}}
  .rank-cell{{text-align:center;background:rgba(0,229,255,0.05);border:1px solid rgba(0,229,255,0.15);border-radius:6px;padding:10px 4px;}}
  .rank-cell .rv{{font-size:16px;font-weight:700;color:var(--ac);}}
  .rank-cell .rl{{font-size:9px;color:var(--mt);margin-top:2px;}}
</style>
</head>
<body>
<header>
  <div>
    <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px;">
      <span class="badge">IEEE FICV 2026</span>
      <span class="badge2">✓ COMPUTED LOCALLY</span>
    </div>
    <h1>CrimeSketch AI — Programmatically Computed Results</h1>
    <p class="sub">All metrics computed from live model inference · {n_probe} probes · {n_gallery} gallery · Dataset: {dataset}</p>
    <p class="computed-note">⚡ Every value on this dashboard was computed by run_pipeline.py on your local system — no hardcoded numbers · Run date: {run_date}</p>
  </div>
</header>

<div class="tab-bar">
  <button class="tab active" onclick="showTab('overview',this)">📊 Overview</button>
  <button class="tab" onclick="showTab('bar',this)">📈 Bar + CMC</button>
  <button class="tab" onclick="showTab('roc',this)">🎯 ROC + PR</button>
  <button class="tab" onclick="showTab('scatter',this)">⚡ Latency</button>
  <button class="tab" onclick="showTab('model',this)">🔬 Model Profile</button>
  <button class="tab" onclick="showTab('table',this)">📋 Full Table</button>
</div>

<!-- ── OVERVIEW ── -->
<div id="panel-overview" class="panel active">
  <div id="metricsStrip" class="metrics-strip" style="margin-bottom:18px;"></div>
  <div class="grid2">
    <div class="card">
      <div class="ct">Rank-N Breakdown — CrimeSketch AI</div>
      <div class="cs">Computed on {n_probe} test probes</div>
      <div id="rankBars"></div>
    </div>
    <div class="card">
      <div class="ct">Per-Category Rank-1 Accuracy</div>
      <div class="cs">6 forensic sketch categories · computed from test set</div>
      <canvas id="radarChart" height="240"></canvas>
    </div>
  </div>
</div>

<!-- ── BAR + CMC ── -->
<div id="panel-bar" class="panel">
  <div class="grid2">
    <div class="card">
      <div class="ct">Fig 2 — Rank-1 &amp; Rank-5 Identification Accuracy</div>
      <div class="cs">Computed programmatically on test split · {n_probe} probe sketches</div>
      <canvas id="barChart" height="300"></canvas>
    </div>
    <div class="card">
      <div class="ct">Fig 3 — CMC Curve (Rank-1 to Rank-20)</div>
      <div class="cs">Cumulative match rate · computed via FAISS L2 retrieval</div>
      <canvas id="cmcChart" height="300"></canvas>
    </div>
  </div>
</div>

<!-- ── ROC + PR ── -->
<div id="panel-roc" class="panel">
  <div class="grid2">
    <div class="card">
      <div class="ct">Fig 4 — ROC Curve (genuine vs impostor pairs)</div>
      <div class="cs">AUC computed from verification pair sampling · top models shown</div>
      <canvas id="rocChart" height="280"></canvas>
    </div>
    <div class="card">
      <div class="ct">Fig 5 — Precision-Recall Curve</div>
      <div class="cs">Average Precision computed from sampled pairs</div>
      <canvas id="prChart" height="280"></canvas>
    </div>
  </div>
</div>

<!-- ── SCATTER + LATENCY ── -->
<div id="panel-scatter" class="panel">
  <div class="grid2">
    <div class="card">
      <div class="ct">Fig 6 — Rank-1 Accuracy vs Query Latency</div>
      <div class="cs">Bubble radius ∝ Rank-5 accuracy · latency measured over 50 trials</div>
      <canvas id="scatterChart" height="300"></canvas>
    </div>
    <div class="card">
      <div class="ct">Fig 7 — CrimeSketch AI Query Latency Distribution</div>
      <div class="cs">Histogram of per-query FAISS retrieval times</div>
      <canvas id="latChart" height="300"></canvas>
    </div>
  </div>
</div>

<!-- ── MODEL PROFILE ── -->
<div id="panel-model" class="panel">
  <div class="note">Figures below show CrimeSketch AI-specific characteristics computed from local inference.</div>
  <div class="grid2">
    <div class="card">
      <div class="ct">Fig 8 — L2 Embedding Distance Distribution</div>
      <div class="cs">Genuine (same subject) vs Impostor pairs · clear separation enables thresholding</div>
      <canvas id="distChart" height="240"></canvas>
    </div>
    <div class="card">
      <div class="ct">Rank-N Summary — CrimeSketch AI</div>
      <div class="cs">All computed from test split</div>
      <div id="rankGrid" class="rank-grid" style="margin-top:10px;"></div>
    </div>
  </div>
</div>

<!-- ── TABLE ── -->
<div id="panel-table" class="panel">
  <div class="note">TABLE I (Extended) — All values computed programmatically by run_pipeline.py · <strong style="color:var(--ac)">Cyan = CrimeSketch AI</strong> · <strong style="color:var(--ok)">Green = new models</strong></div>
  <div class="card">
    <table class="mt" id="modelTable"></table>
  </div>
</div>

<script>
Chart.defaults.color='#8b949e';
Chart.defaults.borderColor='rgba(48,54,61,0.5)';
Chart.defaults.font.family="'Segoe UI',system-ui,sans-serif";

// ── Data from Python ──────────────────────────────────────────────────────
const modelNames  = {json.dumps(model_names)};
const rank1Vals   = {json.dumps(rank1_vals)};
const rank5Vals   = {json.dumps(rank5_vals)};
const latencyVals = {json.dumps(latency_vals)};
const aucVals     = {json.dumps(auc_vals)};
const apVals      = {json.dumps(ap_vals)};
const colors      = {json.dumps(colors)};
const barFill     = {json.dumps(bar_fill)};
const barFill5    = {json.dumps(bar_fill5)};
const barBorder   = {json.dumps(bar_border)};
const cmcDatasets = {cmc_data_js};
const rocDatasets = {roc_data_js};
const prDatasets  = {pr_data_js};
const scatterDatasets = {scatter_data_js};
const radarData   = {radar_js};
const distData    = {dist_js};
const latData     = {lat_js};
const rankBreak   = {rank_js};

// ── Metrics Strip ────────────────────────────────────────────────────────
const csIdx = modelNames.findIndex(n => n.includes('★') || n.toLowerCase().includes('crimesketch'));
const fbIdx = modelNames.slice(0, csIdx >= 0 ? csIdx : modelNames.length - 1)
                        .reduceRight((a, b, i) => a !== -1 ? a : i, -1);
const csR1  = csIdx >= 0 ? rank1Vals[csIdx] : 0;
const csR5  = csIdx >= 0 ? rank5Vals[csIdx] : 0;
const csLat = csIdx >= 0 ? latencyVals[csIdx] : 0;
const csAUC = csIdx >= 0 ? aucVals[csIdx] : 0;
const csAP  = csIdx >= 0 ? apVals[csIdx] : 0;
const prevR1= fbIdx >= 0 ? rank1Vals[fbIdx] : 0;

const strip = document.getElementById('metricsStrip');
strip.innerHTML = [
  {{val: csR1.toFixed(1)+'%', lbl:'Rank-1 Accuracy', delta:'↑ +'+(csR1-prevR1).toFixed(1)+' pp vs 2nd best'}},
  {{val: csR5.toFixed(1)+'%', lbl:'Rank-5 Accuracy', delta:'Computed on test split'}},
  {{val: csLat.toFixed(1)+' ms', lbl:'Avg Query Latency', delta:'FAISS L2 · CPU', color:'var(--ok)'}},
  {{val: csAUC.toFixed(3), lbl:'AUC-ROC', delta:'Genuine vs impostor pairs', color:'var(--warn)'}},
  {{val: csAP.toFixed(3), lbl:'Avg Precision (AP)', delta:'Precision-Recall curve'}},
].map(m => `<div class="metric"><div class="val" style="color:${{m.color||'var(--ac)'}}">${{m.val}}</div><div class="lbl">${{m.lbl}}</div><div class="delta">${{m.delta}}</div></div>`).join('');

// ── Rank Bars ─────────────────────────────────────────────────────────────
const maxRank = Math.max(...Object.values(rankBreak));
document.getElementById('rankBars').innerHTML = Object.entries(rankBreak).map(([r, v]) =>
  `<div class="bar-row"><div class="bar-lbl">Rank-${{r}}</div><div class="bar-bg"><div class="bar-f" style="width:${{(v/maxRank*100).toFixed(1)}}%"></div></div><div class="bar-val">${{v.toFixed(1)}}%</div></div>`
).join('');

// ── Bar Chart ─────────────────────────────────────────────────────────────
new Chart(document.getElementById('barChart').getContext('2d'), {{
  type:'bar',
  data:{{
    labels:modelNames,
    datasets:[
      {{label:'Rank-1 (%)',data:rank1Vals,backgroundColor:barFill,borderColor:barBorder,borderWidth:2,borderRadius:3}},
      {{label:'Rank-5 (%)',data:rank5Vals,backgroundColor:barFill5,borderColor:barBorder,borderWidth:1.5,borderRadius:3}},
    ]
  }},
  options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:10}}}}}},tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.parsed.y.toFixed(2)}}%`}}}}}},scales:{{x:{{ticks:{{font:{{size:9}},maxRotation:40,minRotation:30}},grid:{{display:false}}}},y:{{min:40,max:100,ticks:{{callback:v=>v+'%'}},title:{{display:true,text:'Accuracy (%)',font:{{size:10}}}}}}}}
}});

// ── CMC Chart ─────────────────────────────────────────────────────────────
const cmcLabels = {json.dumps([f"R-{i}" for i in range(1, 21)])};
new Chart(document.getElementById('cmcChart').getContext('2d'), {{
  type:'line',
  data:{{labels:cmcLabels,datasets:cmcDatasets}},
  options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:9}},padding:6}}}},tooltip:{{callbacks:{{label:c=>` ${{c.dataset.label}}: ${{c.parsed.y.toFixed(2)}}%`}}}}}},scales:{{x:{{grid:{{color:'rgba(48,54,61,0.4)'}}}},y:{{min:40,max:100,ticks:{{callback:v=>v+'%'}},title:{{display:true,text:'Cumulative Accuracy (%)',font:{{size:10}}}}}}}}
}});

// ── ROC ───────────────────────────────────────────────────────────────────
new Chart(document.getElementById('rocChart').getContext('2d'), {{
  type:'scatter',
  data:{{datasets:[{{label:'Random (AUC=0.50)',data:[{{x:0,y:0}},{{x:1,y:1}}],showLine:true,borderColor:'rgba(139,148,158,0.2)',borderDash:[5,5],borderWidth:1,pointRadius:0}},...rocDatasets]}},
  options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:9}},padding:6}}}}}},scales:{{x:{{min:0,max:1,title:{{display:true,text:'False Positive Rate',font:{{size:10}}}},ticks:{{callback:v=>v.toFixed(1)}}}},y:{{min:0,max:1,title:{{display:true,text:'True Positive Rate',font:{{size:10}}}},ticks:{{callback:v=>v.toFixed(1)}}}}}}
}});

// ── PR ────────────────────────────────────────────────────────────────────
new Chart(document.getElementById('prChart').getContext('2d'), {{
  type:'scatter',
  data:{{datasets:prDatasets}},
  options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:9}},padding:6}}}}}},scales:{{x:{{min:0,max:1,title:{{display:true,text:'Recall',font:{{size:10}}}},ticks:{{callback:v=>v.toFixed(1)}}}},y:{{min:0,max:1,title:{{display:true,text:'Precision',font:{{size:10}}}},ticks:{{callback:v=>v.toFixed(1)}}}}}}
}});

// ── Scatter ───────────────────────────────────────────────────────────────
new Chart(document.getElementById('scatterChart').getContext('2d'), {{
  type:'bubble',
  data:{{datasets:scatterDatasets}},
  options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{label:c=>{{const n=modelNames[c.datasetIndex];return[` ${{n}}`,` Rank-1: ${{rank1Vals[c.datasetIndex]?.toFixed(2)}}%`,` Latency: ${{latencyVals[c.datasetIndex]?.toFixed(1)}} ms`];}}}}}}}},scales:{{x:{{title:{{display:true,text:'Avg Query Latency (ms) — lower is better',font:{{size:10}}}},ticks:{{callback:v=>v+' ms'}}}},y:{{min:30,max:100,title:{{display:true,text:'Rank-1 Accuracy (%) — higher is better',font:{{size:10}}}},ticks:{{callback:v=>v+'%'}}}}}}
}});

// ── Radar ─────────────────────────────────────────────────────────────────
if(radarData.datasets && radarData.datasets.length > 0) {{
  new Chart(document.getElementById('radarChart').getContext('2d'), {{
    type:'radar',
    data:radarData,
    options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:10}}}}}}}},scales:{{r:{{min:0,max:100,ticks:{{stepSize:20,font:{{size:8}},backdropColor:'transparent'}},pointLabels:{{font:{{size:9}}}},grid:{{color:'rgba(48,54,61,0.6)'}},angleLines:{{color:'rgba(48,54,61,0.6)'}}}}}}
  }});
}}

// ── Distance Distribution ─────────────────────────────────────────────────
if(distData.bin_centres) {{
  new Chart(document.getElementById('distChart').getContext('2d'), {{
    type:'line',
    data:{{
      labels:distData.bin_centres.map(v=>v.toFixed(2)),
      datasets:[
        {{label:'Genuine pairs (same subject)',data:distData.genuine,borderColor:'#00e5ff',backgroundColor:'rgba(0,229,255,0.12)',fill:true,borderWidth:2.5,pointRadius:0,tension:0.45}},
        {{label:'Impostor pairs',data:distData.impostor,borderColor:'#ff6b6b',backgroundColor:'rgba(255,107,107,0.08)',fill:true,borderWidth:2,pointRadius:0,tension:0.45}},
      ]
    }},
    options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{position:'top',labels:{{boxWidth:10,font:{{size:10}}}}}}}},scales:{{x:{{title:{{display:true,text:'L2 Distance (embedding space)',font:{{size:10}}}},ticks:{{maxTicksLimit:8,font:{{size:9}}}}}},y:{{title:{{display:true,text:'Density',font:{{size:10}}}}}}}}
  }});
}}

// ── Latency Histogram ─────────────────────────────────────────────────────
if(latData.labels) {{
  const thr = latData.threshold || 3;
  new Chart(document.getElementById('latChart').getContext('2d'), {{
    type:'bar',
    data:{{
      labels:latData.labels,
      datasets:[{{
        label:'Query count',
        data:latData.data,
        backgroundColor:latData.labels.map(v=>parseFloat(v)<=thr?'rgba(0,229,255,0.65)':'rgba(255,107,107,0.5)'),
        borderColor:latData.labels.map(v=>parseFloat(v)<=thr?'#00e5ff':'#ff6b6b'),
        borderWidth:1.5,borderRadius:3
      }}]
    }},
    options:{{responsive:true,animation:{{duration:700}},plugins:{{legend:{{display:false}}}},scales:{{x:{{title:{{display:true,text:`Latency (ms) · cyan ≤ ${{thr.toFixed(1)}} ms mean`,font:{{size:10}}}},grid:{{display:false}}}},y:{{title:{{display:true,text:'Queries',font:{{size:10}}}}}}}}
  }});
}}

// ── Rank Grid ─────────────────────────────────────────────────────────────
document.getElementById('rankGrid').innerHTML = Object.entries(rankBreak)
  .map(([r, v]) => `<div class="rank-cell"><div class="rv">${{v.toFixed(1)}}%</div><div class="rl">Rank-${{r}}</div></div>`)
  .join('');

// ── Model Table ───────────────────────────────────────────────────────────
const isNew = n => ['arcface','cosface','magface','adaface','transface'].some(x=>n.toLowerCase().includes(x));
const maxR1 = Math.max(...rank1Vals);
let th = `<thead><tr><th>#</th><th>Method</th><th>Type</th><th>Rank-1 (%)</th><th>Rank-5 (%)</th><th>AUC</th><th>Latency</th><th>Δ vs Baseline</th></tr></thead><tbody>`;
const baseR1 = rank1Vals[0] || 0;
modelNames.forEach((n, i) => {{
  const isOurs = n.includes('★') || n.toLowerCase().includes('crimesketch');
  const pill = isOurs ? '<span class="pill ours">★ Ours</span>' : isNew(n) ? '<span class="pill new">New</span>' : '<span class="pill base">Baseline</span>';
  const bc   = isOurs ? '#00e5ff' : isNew(n) ? '#6bcb77' : '#555e6a';
  const delta = (rank1Vals[i] - baseR1).toFixed(1);
  th += `<tr style="${{isOurs?'color:var(--ac);font-weight:700;':isNew(n)?'color:var(--ok);':''}}">
    <td style="color:var(--mt)">${{i+1}}</td><td>${{n}}</td><td>${{pill}}</td>
    <td><div class="bi"><span>${{rank1Vals[i].toFixed(2)}}%</span><div class="bb"><div class="bf" style="width:${{(rank1Vals[i]/maxR1*100).toFixed(1)}}%;background:${{bc}}"></div></div></div></td>
    <td>${{rank5Vals[i].toFixed(2)}}%</td><td>${{aucVals[i].toFixed(4)}}</td>
    <td>${{latencyVals[i].toFixed(1)}} ms</td>
    <td style="color:${{parseFloat(delta)>0?'var(--ok)':'var(--mt)'}}">+${{delta}} pp</td>
  </tr>`;
}});
document.getElementById('modelTable').innerHTML = th + '</tbody>';

function showTab(id, el) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + id).classList.add('active');
  el.classList.add('active');
}}
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n[Visualize] Dashboard written → {output_path}")
    return output_path
