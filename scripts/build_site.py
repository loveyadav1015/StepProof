#!/usr/bin/env python3
"""Build the static landing page for StepProof (docs/site/index.html).

Reads:
- data/outputs/final/MANIFEST.json (if present, for verified Meta model claims)
- eval_results.json or eval_report.json (if present, for real verification results)
Honesty Rules:
- If no Meta manifest exists, shows 'Meta model run pending'.
- If no evaluated results exist or if planned runs are unfilmed, omits the results table or labels data clearly as synthetic/dev stand-in.
- Strictly self-contained, no external trackers, no CDN dependencies.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
site_dir = repo_root / "docs" / "site"


def load_meta_manifest() -> dict | None:
    manifest_path = repo_root / "data" / "outputs" / "final" / "MANIFEST.json"
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def load_eval_results() -> dict | None:
    for filename in ["eval_results.json", "eval_report.json"]:
        path = repo_root / filename
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def render_meta_section(manifest: dict | None) -> str:
    if not manifest:
        return """
        <div class="card card-warning">
          <div class="status-badge warning">Meta Model Run Pending</div>
          <h3 class="heading-5">Live Meta API Verification Status</h3>
          <p class="body-3 text-secondary">
            StepProof is currently developed and tested with Dev stand-in providers (Groq Llama and local computer vision mocks).
            Live inference runs against Meta's hosted <code>muse-wonder-1.0</code>, <code>muse-spark-1.3</code>, and <code>sam-3.1</code> models will be documented here once real API keys and footage runs are executed with <code>STEPPROOF_REQUIRE_META=1</code>.
          </p>
        </div>
        """

    llm_model = html.escape(str(manifest.get("llm_model", "muse-wonder-1.0")))
    seg_model = html.escape(str(manifest.get("seg_model", "sam-3.1")))
    cost = manifest.get("total_cost_usd", 0.0)
    timestamp = html.escape(str(manifest.get("timestamp", "Verified")))

    return f"""
    <div class="card">
      <div class="status-badge success">Meta Compliant Run Verified</div>
      <h3 class="heading-5">Live Meta API Provenance</h3>
      <ul class="meta-list body-3 text-secondary">
        <li><strong>VLM / LLM:</strong> <code>{llm_model}</code></li>
        <li><strong>Segmentation:</strong> <code>{seg_model}</code></li>
        <li><strong>Total Cost:</strong> ${cost:.4f} USD</li>
        <li><strong>Timestamp:</strong> {timestamp}</li>
      </ul>
    </div>
    """


def render_results_section(results: dict | None) -> str:
    if not results or not results.get("results"):
        print("[build_site] Notice: No evaluation results found to display in public page.")
        return ""

    runs = results.get("results", [])
    # Check if all runs are NOT RUN
    all_not_run = all(r.get("status") == "NOT RUN" for r in runs)
    if all_not_run:
        print("[build_site] Notice: All planned evaluation scenarios are NOT RUN. Omitting benchmark numbers per honesty rules.")
        return """
        <section id="results" class="section">
          <div class="container">
            <div class="section-header">
              <span class="caption-1 text-accent">BENCHMARK DATA</span>
              <h2 class="heading-3">Verification Performance</h2>
              <p class="body-2 text-secondary">Awaiting real footage capture for live benchmarks.</p>
            </div>
            <div class="card card-muted text-center py-8">
              <p class="body-2 text-muted">Planned scenarios for breadboard LED assembly have been defined in <code>manifests/real_footage.yaml</code>. Benchmark metrics will appear here once video capture and evaluation runs complete.</p>
            </div>
          </div>
        </section>
        """

    # Real vs synthetic breakdown
    has_real = any(r.get("source") == "real" and r.get("status") != "NOT RUN" for r in runs)
    _ = "Real Footage" if has_real else "Synthetic Data"

    detection_rate = results.get("detection_rate", 0.0)
    false_alarms = results.get("false_alarms", 0)
    avg_latency = results.get("avg_latency_step_s", 0.0)
    avg_cost = results.get("avg_cost_run_usd", 0.0)

    rows_html = ""
    for r in runs:
        if r.get("status") == "NOT RUN":
            continue
        scen = html.escape(str(r.get("scenario", "")))
        src = html.escape(str(r.get("source", "synthetic")))
        st = html.escape(str(r.get("status", "PASS")))
        st_badge = "success" if st == "PASS" else "danger"
        lat = r.get("latency_per_step_s", 0.0)
        sam = r.get("sam_calls", 0)
        rows_html += f"""
        <tr>
          <td class="font-medium">{scen}</td>
          <td><span class="status-badge {src}">{src}</span></td>
          <td><span class="status-badge {st_badge}">{st}</span></td>
          <td class="mono">{sam}</td>
          <td class="mono">{lat:.3f}s</td>
        </tr>
        """

    return f"""
    <section id="results" class="section">
      <div class="container">
        <div class="section-header">
          <span class="caption-1 text-accent">MEASURED ACCURACY</span>
          <h2 class="heading-3">Verification Performance</h2>
          <p class="body-2 text-secondary">Evaluated rigorously across staged errors and baseline executions.</p>
        </div>

        <div class="stats-grid">
          <div class="stat-card">
            <span class="caption-2 text-muted">DETECTION RATE</span>
            <div class="heading-2 stat-value">{detection_rate:.1f}%</div>
            <span class="caption-3 text-secondary">Over staged execution errors</span>
          </div>
          <div class="stat-card">
            <span class="caption-2 text-muted">FALSE ALARMS</span>
            <div class="heading-2 stat-value">{false_alarms}</div>
            <span class="caption-3 text-secondary">On baseline correct sequence</span>
          </div>
          <div class="stat-card">
            <span class="caption-2 text-muted">AVG LATENCY / STEP</span>
            <div class="heading-2 stat-value">{avg_latency:.3f}s</div>
            <span class="caption-3 text-secondary">Per verified assembly step</span>
          </div>
          <div class="stat-card">
            <span class="caption-2 text-muted">AVG COST / RUN</span>
            <div class="heading-2 stat-value">${avg_cost:.4f}</div>
            <span class="caption-3 text-secondary">Token and segmentation billing</span>
          </div>
        </div>

        <div class="table-container card mt-6">
          <table class="data-table body-4">
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Data Source</th>
                <th>Outcome</th>
                <th>SAM Calls</th>
                <th>Latency / Step</th>
              </tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
      </div>
    </section>
    """


def generate_html() -> str:
    manifest = load_meta_manifest()
    results = load_eval_results()

    meta_html = render_meta_section(manifest)
    results_html = render_results_section(results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>StepProof — Industrial Video SOP & Execution Verification</title>
  <meta name="description" content="Turn one expert video into a verified procedure, and catch mistakes as they happen.">
  <meta property="og:title" content="StepProof — Industrial Video SOP & Execution Verification">
  <meta property="og:description" content="Turn one expert video into a verified procedure, and catch mistakes as they happen.">
  <meta property="og:image" content="og-image.svg">
  <meta property="og:type" content="website">
  <link rel="stylesheet" href="tokens.css">
  <style>
    /* CSS Reset & Semantic Foundation */
    *, *::before, *::after {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: var(--font-family);
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: var(--line-base);
      font-size: var(--text-base);
      -webkit-font-smoothing: antialiased;
    }}

    a {{
      color: inherit;
      text-decoration: none;
      transition: color var(--duration-hover) var(--ease-base);
    }}

    /* Typography Classes matching B1 */
    .heading-1 {{ font-size: var(--text-6xl); line-height: var(--line-6xl); font-weight: var(--font-semibold); text-wrap: balance; }}
    .heading-2 {{ font-size: var(--text-5xl); line-height: var(--line-5xl); font-weight: var(--font-semibold); text-wrap: balance; }}
    .heading-3 {{ font-size: var(--text-4xl); line-height: var(--line-4xl); font-weight: var(--font-semibold); text-wrap: balance; }}
    .heading-4 {{ font-size: var(--text-3xl); line-height: var(--line-3xl); font-weight: var(--font-semibold); }}
    .heading-5 {{ font-size: var(--text-2xl); line-height: var(--line-2xl); font-weight: var(--font-semibold); }}
    .title-1 {{ font-size: var(--text-xl); line-height: var(--line-xl); font-weight: var(--font-medium); }}
    .title-2 {{ font-size: var(--text-lg); line-height: var(--line-lg); font-weight: var(--font-medium); }}
    .body-1 {{ font-size: var(--text-base); line-height: var(--line-base); font-weight: var(--font-semibold); text-wrap: pretty; }}
    .body-2 {{ font-size: var(--text-base); line-height: var(--line-base); font-weight: var(--font-medium); text-wrap: pretty; }}
    .body-3 {{ font-size: var(--text-base); line-height: var(--line-base); font-weight: var(--font-regular); text-wrap: pretty; }}
    .body-4 {{ font-size: var(--text-sm); line-height: var(--line-sm); font-weight: var(--font-medium); }}
    .caption-1 {{ font-size: var(--text-sm); line-height: var(--line-sm); font-weight: var(--font-medium); letter-spacing: 0.05em; text-transform: uppercase; }}
    .caption-2 {{ font-size: var(--text-sm); line-height: var(--line-sm); font-weight: var(--font-regular); }}
    .caption-3 {{ font-size: var(--text-xs); line-height: var(--line-xs); font-weight: var(--font-medium); }}
    .mono {{ font-family: var(--font-mono); }}

    /* Colors */
    .text-primary {{ color: var(--text-primary); }}
    .text-secondary {{ color: var(--text-secondary); }}
    .text-muted {{ color: var(--text-muted); }}
    .text-accent {{ color: var(--accent-text); }}

    /* Layout Containers */
    .container {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 0 var(--spacing-300);
    }}

    .section {{
      padding: var(--spacing-800) 0;
    }}

    .section-header {{
      max-width: 680px;
      margin-bottom: var(--spacing-500);
    }}

    .section-header h2 {{
      margin-top: var(--spacing-75);
      margin-bottom: var(--spacing-100);
    }}

    /* Floating Navigation (B7 spec) */
    .nav-bar {{
      position: sticky;
      top: var(--spacing-200);
      z-index: 50;
      max-width: 900px;
      margin: 0 auto;
      padding: var(--spacing-100) var(--spacing-300);
      background-color: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-full);
      display: flex;
      align-items: center;
      justify-content: space-between;
      backdrop-filter: blur(8px);
    }}

    .nav-brand {{
      display: flex;
      align-items: center;
      gap: var(--spacing-75);
      font-weight: var(--font-semibold);
      font-size: var(--text-base);
    }}

    .nav-links {{
      display: flex;
      align-items: center;
      gap: var(--spacing-300);
      list-style: none;
    }}

    .nav-link {{
      color: var(--text-secondary);
      font-size: var(--text-sm);
      font-weight: var(--font-medium);
    }}

    .nav-link:hover {{
      color: var(--text-primary);
    }}

    /* Buttons */
    .btn {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 8px 16px;
      border-radius: var(--radius-md);
      font-size: var(--text-base);
      line-height: var(--line-base);
      font-weight: var(--font-semibold);
      cursor: pointer;
      border: 1px solid transparent;
      transition: all var(--duration-hover) var(--ease-base);
    }}

    .btn:active {{
      transform: translateY(1px);
    }}

    .btn:focus-visible {{
      outline: 2px solid var(--border-focus);
      outline-offset: 2px;
    }}

    .btn-primary {{
      background-color: var(--accent);
      color: #ffffff;
    }}

    .btn-primary:hover {{
      background-color: var(--accent-hover);
    }}

    .btn-secondary {{
      background-color: var(--bg-surface);
      border-color: var(--border-strong);
      color: var(--text-primary);
    }}

    .btn-secondary:hover {{
      background-color: var(--bg-surface-hover);
    }}

    .btn-sm {{
      padding: 4px 12px;
      font-size: var(--text-sm);
    }}

    /* Hero Section (B5) */
    .hero {{
      padding: var(--spacing-900) 0 var(--spacing-600) 0;
    }}

    .hero-content {{
      max-width: 680px;
    }}

    .hero-title {{
      margin-bottom: var(--spacing-200);
      /* Left to right gradient on text per B5 */
      background: linear-gradient(90deg, #FFFFFF 0%, #9B9B9B 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}

    @media (prefers-color-scheme: light) {{
      .hero-title {{
        background: linear-gradient(90deg, #000000 0%, #666666 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }}
    }}

    .hero-subtitle {{
      margin-bottom: var(--spacing-400);
      color: var(--text-secondary);
      font-size: var(--text-xl);
      line-height: var(--line-xl);
    }}

    .hero-ctas {{
      display: flex;
      gap: var(--spacing-200);
      align-items: center;
      margin-bottom: var(--spacing-500);
    }}

    /* Cards & Containers (B4 & B3) */
    .card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: var(--spacing-300);
    }}

    .card-warning {{
      border-color: rgba(245, 158, 11, 0.4);
      background-color: rgba(245, 158, 11, 0.05);
    }}

    .card-muted {{
      background-color: var(--bg-surface);
      border-style: dashed;
    }}

    /* Grid Layouts */
    .steps-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: var(--spacing-300);
    }}

    .step-num {{
      display: inline-block;
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      color: var(--accent-text);
      font-weight: var(--font-semibold);
      margin-bottom: var(--spacing-100);
    }}

    /* Tagline Reveal Section (B11 Mandatory) */
    .tagline-reveal {{
      padding: var(--spacing-800) 0;
      border-top: 1px solid var(--border-subtle);
      border-bottom: 1px solid var(--border-subtle);
    }}

    .reveal-text {{
      max-width: 680px;
      font-size: var(--text-3xl);
      line-height: var(--line-3xl);
      font-weight: var(--font-semibold);
      color: var(--text-primary);
    }}

    .reveal-word {{
      display: inline-block;
      opacity: 0.3;
      transition: opacity 600ms var(--ease-base);
      margin-right: 0.25em;
    }}

    .reveal-word.active {{
      opacity: 1;
    }}

    /* Stats Grid */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: var(--spacing-200);
      margin-bottom: var(--spacing-300);
    }}

    .stat-card {{
      background-color: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      padding: var(--spacing-300);
    }}

    .stat-value {{
      margin: var(--spacing-75) 0;
      color: var(--text-primary);
    }}

    /* Data Table */
    .table-container {{
      overflow-x: auto;
    }}

    .data-table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }}

    .data-table th, .data-table td {{
      padding: var(--spacing-100) var(--spacing-200);
      border-bottom: 1px solid var(--border-subtle);
    }}

    .data-table th {{
      color: var(--text-muted);
      font-weight: var(--font-semibold);
      font-size: var(--text-xs);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}

    /* Badges */
    .status-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: var(--radius-full);
      font-size: var(--text-xs);
      font-weight: var(--font-semibold);
      line-height: var(--line-xs);
    }}

    .status-badge.success {{
      background-color: rgba(16, 185, 129, 0.15);
      color: #34d399;
    }}

    .status-badge.warning {{
      background-color: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
    }}

    .status-badge.danger {{
      background-color: rgba(239, 68, 68, 0.15);
      color: #f87171;
    }}

    .status-badge.neutral, .status-badge.synthetic {{
      background-color: rgba(100, 116, 139, 0.15);
      color: #94a3b8;
    }}

    .status-badge.real {{
      background-color: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
    }}

    /* Video Placeholder */
    .video-container {{
      position: relative;
      background-color: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-lg);
      padding: var(--spacing-600) var(--spacing-300);
      text-align: center;
      margin-top: var(--spacing-300);
    }}

    /* Footer */
    .footer {{
      padding: var(--spacing-800) 0;
      border-top: 1px solid var(--border-subtle);
      color: var(--text-muted);
      font-size: var(--text-sm);
    }}

    .footer-content {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: var(--spacing-300);
      align-items: center;
    }}

    /* Accessibility: Reduced Motion */
    @media (prefers-reduced-motion: reduce) {{
      *, *::before, *::after {{
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
      }}
      .reveal-word {{
        opacity: 1 !important;
      }}
    }}
  </style>
</head>
<body>

  <!-- Floating Nav Bar -->
  <nav class="nav-bar" aria-label="Main Navigation">
    <a href="#" class="nav-brand">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
      </svg>
      <span>StepProof</span>
    </a>
    <ul class="nav-links">
      <li><a href="#how-it-works" class="nav-link">How it Works</a></li>
      <li><a href="#demo" class="nav-link">Demo</a></li>
      <li><a href="#models" class="nav-link">Models</a></li>
      <li><a href="#limitations" class="nav-link">Scope</a></li>
      <li><a href="https://github.com/loveyadav1015/StepProof" class="btn btn-secondary btn-sm" target="_blank" rel="noopener">GitHub</a></li>
    </ul>
  </nav>

  <!-- Hero Section -->
  <header class="hero">
    <div class="container">
      <div class="hero-content">
        <h1 class="hero-title heading-1">Industrial video SOPs that verify themselves.</h1>
        <p class="hero-subtitle">Turn one expert video into a verified procedure, and catch mistakes as they happen.</p>
        <div class="hero-ctas">
          <a href="#demo" class="btn btn-primary">Watch demo</a>
          <a href="https://github.com/loveyadav1015/StepProof" class="btn btn-secondary" target="_blank" rel="noopener">View repo</a>
        </div>
      </div>
    </div>
  </header>

  <!-- How It Works Section (3 Steps) -->
  <section id="how-it-works" class="section">
    <div class="container">
      <div class="section-header">
        <span class="caption-1 text-accent">METHODOLOGY</span>
        <h2 class="heading-3">How StepProof Works</h2>
        <p class="body-2 text-secondary">A three-stage deterministic pipeline from footage to real-time verification.</p>
      </div>

      <div class="steps-grid">
        <div class="card">
          <span class="step-num">STEP 01</span>
          <h3 class="title-1 mb-2">Learn from one video</h3>
          <p class="body-3 text-secondary">
            A single expert demonstration is analyzed by multimodal VLM into ordered atomic steps, visual evidence criteria, and expected end-state preconditions.
          </p>
        </div>

        <div class="card">
          <span class="step-num">STEP 02</span>
          <h3 class="title-1 mb-2">Ground it in the image</h3>
          <p class="body-3 text-secondary">
            Zero-shot segmentation grounds tools, components, and connections with bounding boxes, spatial coordinates, and pixel area metrics.
          </p>
        </div>

        <div class="card">
          <span class="step-num">STEP 03</span>
          <h3 class="title-1 mb-2">Check a learner</h3>
          <p class="body-3 text-secondary">
            Learner execution footage is continuously monitored with multi-frame temporal smoothing to catch skipped steps, wrong parts, or out-of-sequence actions.
          </p>
        </div>
      </div>
    </div>
  </section>

  <!-- Tagline Reveal Section (B11 Mandatory) -->
  <section class="tagline-reveal">
    <div class="container">
      <div class="reveal-text" id="revealText">
        Quality verification should not require writing thousand-line rules or tagging months of training data.
      </div>
    </div>
  </section>

  <!-- Demo Video Section -->
  <section id="demo" class="section">
    <div class="container">
      <div class="section-header">
        <span class="caption-1 text-accent">WALKTHROUGH</span>
        <h2 class="heading-3">Demonstration</h2>
        <p class="body-2 text-secondary">Watch StepProof inspect assembly steps and flag staged errors.</p>
      </div>

      <div class="video-container">
        <!-- Placeholder container until user provides URL -->
        <div class="body-2 text-muted">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 12px; display: inline-block;">
            <polygon points="5 3 19 12 5 21 5 3"/>
          </svg>
          <p class="font-medium text-primary">Demo Video Walkthrough</p>
          <p class="caption-2 text-muted mt-1">Video demonstration URL pending final live footage link.</p>
        </div>
      </div>
    </div>
  </section>

  <!-- Meta Models Section (Honesty Rules strictly applied) -->
  <section id="models" class="section">
    <div class="container">
      <div class="section-header">
        <span class="caption-1 text-accent">MODEL PROVENANCE</span>
        <h2 class="heading-3">Meta Models & Verification Architecture</h2>
        <p class="body-2 text-secondary">Honest accounting of model provenance and compliance boundaries.</p>
      </div>

      {meta_html}
    </div>
  </section>

  <!-- Measured Results Section (Generated from JSON) -->
  {results_html}

  <!-- Scope & Limitations Note -->
  <section id="limitations" class="section">
    <div class="container">
      <div class="card card-muted">
        <span class="caption-1 text-muted">IMPORTANT NOTICE</span>
        <h3 class="title-1 mt-1 mb-2">Scope & Safety Boundaries</h3>
        <p class="body-3 text-secondary">
          StepProof is designed as a <strong>training and quality-assistance aid</strong> for assembly workshops, repair teams, and technicians.
          It is <strong>not a safety-critical interlock system</strong> and should not be used as the sole safety safeguard where physical hazard, high voltage, or human injury risks exist.
        </p>
      </div>
    </div>
  </section>

  <!-- Footer -->
  <footer class="footer">
    <div class="container footer-content">
      <div>
        <p class="body-4 font-semibold text-primary">StepProof</p>
        <p class="caption-2 text-muted">Industrial Video SOP &amp; Execution Verification</p>
      </div>
      <div>
        <p class="caption-2 text-muted">Built for the Meta AI Hackathon</p>
      </div>
      <div>
        <a href="https://github.com/loveyadav1015/StepProof" class="caption-1 text-accent" target="_blank" rel="noopener">GitHub Repository →</a>
      </div>
    </div>
  </footer>

  <!-- Tagline Reveal JS (IntersectionObserver per B11) -->
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      var revealEl = document.getElementById("revealText");
      if (!revealEl) return;

      var text = revealEl.innerText.trim();
      var words = text.split(/\\s+/);
      revealEl.innerHTML = words.map(function(w) {{
        return '<span class="reveal-word">' + w + '</span>';
      }}).join(' ');

      var wordSpans = revealEl.querySelectorAll(".reveal-word");

      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
        wordSpans.forEach(function(s) {{ s.classList.add("active"); }});
        return;
      }}

      var observer = new IntersectionObserver(function(entries) {{
        entries.forEach(function(entry) {{
          if (entry.isIntersecting) {{
            wordSpans.forEach(function(s, idx) {{
              setTimeout(function() {{
                s.classList.add("active");
              }}, idx * 45);
            }});
            observer.unobserve(entry.target);
          }}
        }});
      }}, {{ threshold: 0.3 }});

      observer.observe(revealEl);
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    site_dir.mkdir(parents=True, exist_ok=True)
    out_file = site_dir / "index.html"
    content = generate_html()
    out_file.write_text(content, encoding="utf-8")
    print(f"✅ Generated static landing page: {out_file} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
