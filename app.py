#!/usr/bin/env python3
"""
FastAPI interface for Sprint Intelligence Multi-Agent System.
Research project: Intelligent Sprint Analysis Using Agentic System for Startup Projects
Florida Polytechnic University – Department of Computer Science
"""

import asyncio
import json
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.agents.state import OrchestratorState, GitHubIssue
from src.agents.orchestrator import MasterOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("sprint-ui")

orchestrator: Optional[MasterOrchestrator] = None
last_result: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    logger.info("Initializing MasterOrchestrator …")
    try:
        orchestrator = MasterOrchestrator()
        logger.info("MasterOrchestrator ready ✓")
    except Exception as exc:
        logger.error("Orchestrator init failed: %s", exc)
        orchestrator = None
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Sprint Intelligence Agent",
    description=(
        "Multi-agent LLM system for real-time, "
        "explainable sprint health assessments"
    ),
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=HTML_PAGE, status_code=200)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "orchestrator_ready": orchestrator is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/analyze")
async def analyze(request: Request):
    global last_result
    if orchestrator is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    "Orchestrator not initialised. "
                    "Check Ollama connection."
                )
            },
        )

    body = await request.json()
    repositories = body.get("repositories", [])
    eval_mode = body.get("eval_mode", "resilient")

    if not repositories:
        return JSONResponse(
            status_code=400,
            content={"error": "repositories list is required"},
        )

    try:
        now = datetime.now()
        input_state = OrchestratorState(
            repositories=repositories,
            eval_mode=eval_mode,
            milestone_data={
                "created_at": (
                    now - timedelta(days=14)
                ).isoformat(),
                "due_on": now.isoformat(),
            },
        )

        loop = asyncio.get_event_loop()
        result_state = await loop.run_in_executor(
            None, orchestrator.invoke, input_state,
        )

        result_dict = result_state.dict()

        def _safe(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError

        result_json = json.loads(
            json.dumps(result_dict, default=_safe)
        )
        last_result = result_json
        return JSONResponse(content={
            "status": "success", "result": result_json
        })

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Analysis failed: %s\n%s", exc, tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": tb},
        )


@app.post("/api/analyze/mock")
async def analyze_mock(request: Request):
    global last_result
    if orchestrator is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": (
                    "Orchestrator not initialised. "
                    "Check Ollama connection."
                )
            },
        )

    body = await request.json()
    eval_mode = body.get("eval_mode", "resilient")

    try:
        now = datetime.now()
        issues = [
            GitHubIssue(
                number=1, title="Fix authentication bug",
                body="Users unable to login with OAuth",
                state="closed", labels=["bug", "priority-high"],
                created_at=(now - timedelta(days=5)).isoformat(),
            ),
            GitHubIssue(
                number=2, title="Add dark mode support",
                body="UI redesign for dark theme",
                state="open", labels=["feature", "ui"],
                created_at=(now - timedelta(days=10)).isoformat(),
            ),
            GitHubIssue(
                number=3, title="Optimize database queries",
                body="Performance improvement for slow reports",
                state="open",
                labels=["performance", "backend"],
                created_at=(now - timedelta(days=3)).isoformat(),
            ),
        ]

        input_state = OrchestratorState(
            repositories=["Mintplex-Labs/anything-llm"],
            eval_mode=eval_mode,
            github_issues=issues,
            github_prs=[],
            commits=[],
            milestone_data={
                "created_at": (
                    now - timedelta(days=7)
                ).isoformat(),
                "due_on": now.isoformat(),
            },
        )

        loop = asyncio.get_event_loop()
        result_state = await loop.run_in_executor(
            None, orchestrator.invoke, input_state,
        )

        result_dict = result_state.dict()

        def _safe(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError

        result_json = json.loads(
            json.dumps(result_dict, default=_safe)
        )
        last_result = result_json
        return JSONResponse(content={
            "status": "success", "result": result_json
        })

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Mock analysis failed: %s\n%s", exc, tb)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "traceback": tb},
        )


@app.get("/api/last-result")
async def get_last_result():
    if last_result is None:
        return JSONResponse(
            status_code=404,
            content={"error": "No analysis has been run yet."},
        )
    return JSONResponse(content=last_result)


# ═══════════════════════════════════════════════════════════════
# HTML Template
# ═══════════════════════════════════════════════════════════════

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>Sprint Intelligence — Multi-Agent Analysis System</title>
<meta name="description" content="Intelligent Sprint Analysis Using Agentic System for Startup Projects. Multi-agent LLM system with RAG-enhanced explainability for real-time sprint health assessments."/>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080c14;--bg-surface:#0f1520;--bg-card:#141c2b;--bg-input:#0b1120;
  --border:#1a2744;--border-hover:#2a3d66;--border-focus:#6366f1;
  --text:#e2e8f0;--text-sec:#94a3b8;--text-muted:#64748b;
  --accent:#6366f1;--accent-dim:#6366f120;
  --green:#10b981;--green-dim:#10b98120;
  --amber:#f59e0b;--amber-dim:#f59e0b20;
  --red:#ef4444;--red-dim:#ef444420;
  --cyan:#06b6d4;--cyan-dim:#06b6d420;
  --purple:#a855f7;--purple-dim:#a855f720;
  --r:12px;--r-lg:16px;
  --shadow:0 4px 24px rgba(0,0,0,.4);
  --tr:all .22s cubic-bezier(.4,0,.2,1);
  --font:'Inter',system-ui,sans-serif;
  --mono:'JetBrains Mono',monospace;
}
html{font-size:14px;scroll-behavior:smooth}
body{font-family:var(--font);background:var(--bg);color:var(--text);min-height:100vh}
body::before{content:'';position:fixed;inset:0;z-index:-1;
  background:
    radial-gradient(ellipse 55% 45% at 8% 15%,#6366f10c 0%,transparent 70%),
    radial-gradient(ellipse 45% 35% at 90% 80%,#06b6d40c 0%,transparent 70%),
    radial-gradient(ellipse 35% 25% at 50% 5%,#10b98108 0%,transparent 60%)}

/* Layout */
.shell{max-width:1180px;margin:0 auto;padding:1.6rem 1.5rem 4rem}

/* Header – research branded */
.header{text-align:center;margin-bottom:2rem;padding-bottom:1.5rem;border-bottom:1px solid var(--border)}
.header h1{font-size:1.65rem;font-weight:800;letter-spacing:-.02em;
  background:linear-gradient(135deg,#6366f1,#06b6d4,#10b981);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.header .subtitle{color:var(--text-sec);font-size:.88rem;margin-top:.3rem}
.header .institution{color:var(--text-muted);font-size:.76rem;margin-top:.25rem;letter-spacing:.02em}
.header .team{color:var(--text-muted);font-size:.72rem;margin-top:.2rem}

/* KPI ribbon */
.kpi-ribbon{display:flex;gap:.55rem;justify-content:center;flex-wrap:wrap;margin-top:1rem}
.kpi{display:flex;align-items:center;gap:.35rem;padding:.3rem .75rem;
  background:var(--bg-surface);border:1px solid var(--border);border-radius:20px;font-size:.72rem}
.kpi .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.kpi .label{color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.kpi .val{color:var(--text);font-weight:700}

/* Cards */
.card{background:var(--bg-card);border:1px solid var(--border);border-radius:var(--r-lg);padding:1.4rem;box-shadow:var(--shadow);transition:var(--tr)}
.card:hover{border-color:var(--border-hover)}
.card-title{font-size:.92rem;font-weight:700;margin-bottom:1rem;display:flex;align-items:center;gap:.45rem}
.card-title .icon{font-size:1.1rem}

/* Agent pipeline viz */
.pipeline{display:flex;gap:0;overflow-x:auto;padding:.5rem 0;margin-bottom:.5rem}
.pipeline::-webkit-scrollbar{height:4px}
.pipeline::-webkit-scrollbar-thumb{background:#334155;border-radius:2px}
.agent-node{display:flex;flex-direction:column;align-items:center;min-width:72px;position:relative}
.agent-node .dot{width:28px;height:28px;border-radius:50%;
  background:var(--bg-surface);border:2px solid var(--border);
  display:flex;align-items:center;justify-content:center;font-size:.75rem;
  transition:var(--tr);z-index:1}
.agent-node.active .dot{border-color:var(--accent);background:var(--accent-dim);box-shadow:0 0 12px var(--accent-dim);animation:pulse-ring 1.6s ease infinite}
.agent-node.done .dot{border-color:var(--green);background:var(--green-dim)}
.agent-node.error .dot{border-color:var(--red);background:var(--red-dim)}
.agent-node .lbl{font-size:.58rem;color:var(--text-muted);margin-top:.3rem;text-align:center;max-width:68px;line-height:1.25}
.agent-node::after{content:'';position:absolute;top:14px;left:50px;width:calc(100% - 22px);height:2px;background:var(--border)}
.agent-node:last-child::after{display:none}
.agent-node.done::after{background:var(--green)}
.agent-node.active::after{background:linear-gradient(90deg,var(--accent),var(--border))}
@keyframes pulse-ring{0%,100%{box-shadow:0 0 0 0 var(--accent-dim)}50%{box-shadow:0 0 0 6px transparent}}

/* Form */
.form-row{display:grid;grid-template-columns:1fr 200px;gap:.75rem;margin-bottom:.85rem}
@media(max-width:640px){.form-row{grid-template-columns:1fr}}
.form-group label{display:block;font-size:.7rem;font-weight:600;color:var(--text-muted);
  text-transform:uppercase;letter-spacing:.06em;margin-bottom:.25rem}
.form-group input,.form-group select{width:100%;padding:.55rem .75rem;font-size:.85rem;
  font-family:var(--font);background:var(--bg-input);color:var(--text);
  border:1px solid var(--border);border-radius:var(--r);outline:none;transition:var(--tr)}
.form-group input:focus,.form-group select:focus{border-color:var(--border-focus);
  box-shadow:0 0 0 3px #6366f118}
.form-group input::placeholder{color:var(--text-muted)}

/* Buttons */
.btn{display:inline-flex;align-items:center;gap:.4rem;padding:.6rem 1.1rem;font-size:.82rem;
  font-weight:600;font-family:var(--font);border:none;border-radius:var(--r);cursor:pointer;transition:var(--tr)}
.btn-primary{background:linear-gradient(135deg,#6366f1,#4f46e5);color:#fff;box-shadow:0 3px 12px #6366f140}
.btn-primary:hover{transform:translateY(-1px);box-shadow:0 5px 18px #6366f160}
.btn-secondary{background:var(--bg-surface);color:var(--text-sec);border:1px solid var(--border)}
.btn-secondary:hover{border-color:var(--text-muted);color:var(--text)}
.btn:disabled{opacity:.45;cursor:not-allowed;transform:none!important}
.btn-row{display:flex;gap:.6rem;flex-wrap:wrap}

/* Status */
#status-bar{max-height:0;overflow:hidden;transition:max-height .4s,padding .4s}
#status-bar.visible{max-height:600px;padding:.85rem 0 0}
.status-msg{display:flex;align-items:center;gap:.5rem;font-size:.8rem;color:var(--text-sec)}
.status-msg .pulse{width:7px;height:7px;border-radius:50%;background:var(--accent);animation:pulse-dot 1.4s ease infinite}
@keyframes pulse-dot{0%,100%{opacity:.3;transform:scale(.85)}50%{opacity:1;transform:scale(1.2)}}

/* Agent log console */
#agent-logs{margin-top:.65rem;background:var(--bg-input);border:1px solid var(--border);
  border-radius:var(--r);padding:.65rem .85rem;max-height:180px;overflow-y:auto;
  font-family:var(--mono);font-size:.7rem;line-height:1.6;color:var(--green);display:none}
#agent-logs.visible{display:block}
#agent-logs .log-line{opacity:0;animation:fi .25s forwards}
@keyframes fi{to{opacity:1}}

/* Error */
.error-banner{background:#ef444415;border:1px solid #ef444435;border-radius:var(--r);
  padding:.85rem 1rem;margin-top:.75rem;color:var(--red);font-size:.82rem;display:none}
.error-banner.visible{display:block}
.error-banner pre{margin-top:.4rem;font-family:var(--mono);font-size:.7rem;white-space:pre-wrap;color:#fca5a5}

/* ─── Results area ──────────────────────────────────────── */
#results{margin-top:1.8rem;display:none}
#results.visible{display:block}

/* Section titles */
.sec-title{font-size:.82rem;font-weight:700;color:var(--text-sec);text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:.65rem;display:flex;align-items:center;gap:.35rem}

/* Score grid */
.score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:.75rem;margin-bottom:1.4rem}
.sc{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);
  padding:.85rem 1rem;text-align:center;position:relative;overflow:hidden}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--accent),var(--cyan))}
.sc.green::before{background:linear-gradient(90deg,var(--green),#34d399)}
.sc.amber::before{background:linear-gradient(90deg,var(--amber),#fbbf24)}
.sc.red::before{background:linear-gradient(90deg,var(--red),#f87171)}
.sc .lbl{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted);font-weight:600}
.sc .val{font-size:1.65rem;font-weight:800;margin-top:.15rem}
.sc .sub{font-size:.68rem;color:var(--text-sec);margin-top:.12rem}

/* Run metrics research panel */
.metrics-panel{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.6rem;margin-bottom:1.4rem}
.metric{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);padding:.7rem .85rem}
.metric .m-label{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted);font-weight:600}
.metric .m-val{font-size:1.1rem;font-weight:700;margin-top:.1rem}
.metric .m-sub{font-size:.62rem;color:var(--text-muted);margin-top:.05rem}

/* Two-col layout for features */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1.4rem}
@media(max-width:700px){.two-col{grid-template-columns:1fr}}

/* Feature category */
.feat-cat{margin-bottom:.5rem}
.feat-cat-title{font-size:.7rem;font-weight:700;color:var(--cyan);text-transform:uppercase;letter-spacing:.04em;margin-bottom:.3rem;
  display:flex;align-items:center;gap:.3rem}
.feat-row{display:flex;justify-content:space-between;padding:.18rem 0;border-bottom:1px solid #1a274430;font-size:.72rem}
.feat-row .fname{color:var(--text-sec)}
.feat-row .fval{color:var(--text);font-weight:600;font-family:var(--mono);font-size:.68rem}

/* Item cards (risks/recs) */
.item-list{display:flex;flex-direction:column;gap:.55rem;margin-bottom:1.4rem}
.item-card{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);
  padding:.75rem .95rem;transition:var(--tr)}
.item-card:hover{border-color:var(--border-hover)}
.item-card .ih{display:flex;justify-content:space-between;align-items:center;margin-bottom:.2rem}
.item-card .it{font-weight:600;font-size:.82rem}
.badge{display:inline-block;padding:.12rem .5rem;font-size:.6rem;font-weight:700;border-radius:6px;
  text-transform:uppercase;letter-spacing:.04em}
.badge-high{background:var(--red-dim);color:var(--red)}
.badge-medium{background:var(--amber-dim);color:var(--amber)}
.badge-low{background:var(--green-dim);color:var(--green)}
.badge-llm{background:var(--purple-dim);color:var(--purple)}
.badge-fallback{background:var(--amber-dim);color:var(--amber)}
.item-card .id{font-size:.72rem;color:var(--text-sec);line-height:1.5}

/* Narrative */
#narrative-box{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);
  padding:1rem 1.2rem;font-size:.8rem;line-height:1.7;color:var(--text-sec);white-space:pre-wrap;margin-bottom:1.4rem}
#narrative-box strong{color:var(--text)}

/* Dependency graph */
.dep-graph{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);
  padding:.85rem 1rem;margin-bottom:1.4rem}
.dep-node{display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .6rem;
  background:var(--bg-card);border:1px solid var(--border);border-radius:6px;
  font-size:.7rem;font-weight:600;margin:.2rem}
.dep-edge{display:flex;align-items:center;gap:.4rem;font-size:.68rem;color:var(--text-muted);padding:.15rem 0}
.dep-edge .arrow{color:var(--cyan)}

/* RAG evidence */
.rag-panel{background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--r);
  padding:.85rem 1rem;margin-bottom:1.4rem;font-size:.75rem}

/* Accordion */
.acc-toggle{cursor:pointer;user-select:none;display:flex;align-items:center;gap:.35rem}
.acc-toggle .arrow{transition:transform .25s;font-size:.7rem}
.acc-toggle.open .arrow{transform:rotate(90deg)}
.acc-body{max-height:0;overflow:hidden;transition:max-height .35s ease}
.acc-body.open{max-height:800px}
.acc-content{margin-top:.4rem;background:var(--bg-input);border:1px solid var(--border);
  border-radius:var(--r);padding:.65rem .85rem;font-family:var(--mono);font-size:.68rem;
  line-height:1.55;color:var(--cyan);max-height:260px;overflow-y:auto}

/* Scrollbar */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#475569}

/* Responsive */
@media(max-width:640px){
  .shell{padding:1rem .85rem 3rem}
  .header h1{font-size:1.2rem}
  .score-grid{grid-template-columns:repeat(2,1fr)}
  .metrics-panel{grid-template-columns:repeat(2,1fr)}
}
</style>
</head>
<body>
<div class="shell">

<!-- ═══ Header ═══ -->
<header class="header">
  <h1>⚡ Intelligent Sprint Analysis Using Agentic System</h1>
  <p class="subtitle">Multi-Agent LLM Pipeline with RAG-Enhanced Explainability for Startup Projects</p>
  <p class="institution">Florida Polytechnic University · Department of Computer Science</p>
  <p class="team">Bibek Gupta · Saarupya Sunkara · Siwani Sah · Deepthi Reddy Chelladi</p>
  <div class="kpi-ribbon">
    <div class="kpi"><div class="dot" style="background:var(--green)"></div><span class="label">Agents</span><span class="val">11</span></div>
    <div class="kpi"><div class="dot" style="background:var(--cyan)"></div><span class="label">Features</span><span class="val">18</span></div>
    <div class="kpi"><div class="dot" style="background:var(--purple)"></div><span class="label">Modalities</span><span class="val">6</span></div>
    <div class="kpi"><div class="dot" style="background:var(--accent)"></div><span class="label">LLM</span><span class="val">Ollama</span></div>
    <div class="kpi"><div class="dot" style="background:var(--amber)"></div><span class="label">RAG</span><span class="val">ChromaDB</span></div>
    <div class="kpi"><div class="dot" style="background:var(--green)"></div><span class="label">Target F1</span><span class="val">≥0.85</span></div>
  </div>
</header>

<!-- ═══ Input ═══ -->
<div class="card" id="input-card">
  <div class="card-title"><span class="icon">🔧</span> Configure Sprint Analysis</div>

  <!-- Agent pipeline visualization -->
  <div class="pipeline" id="pipeline">
    <div class="agent-node" data-idx="0"><div class="dot">🔀</div><div class="lbl">Router</div></div>
    <div class="agent-node" data-idx="1"><div class="dot">📥</div><div class="lbl">Data Collector</div></div>
    <div class="agent-node" data-idx="2"><div class="dot">🔗</div><div class="lbl">Dep Graph</div></div>
    <div class="agent-node" data-idx="3"><div class="dot">⚙️</div><div class="lbl">Feature Eng.</div></div>
    <div class="agent-node" data-idx="4"><div class="dot">🧬</div><div class="lbl">Synthetic Gen.</div></div>
    <div class="agent-node" data-idx="5"><div class="dot">🧲</div><div class="lbl">Embedding RAG</div></div>
    <div class="agent-node" data-idx="6"><div class="dot">🤖</div><div class="lbl">LLM Reasoner</div></div>
    <div class="agent-node" data-idx="7"><div class="dot">📈</div><div class="lbl">Sprint Analyzer</div></div>
    <div class="agent-node" data-idx="8"><div class="dot">🎓</div><div class="lbl">LoRA Training</div></div>
    <div class="agent-node" data-idx="9"><div class="dot">🛡️</div><div class="lbl">Risk Assessor</div></div>
    <div class="agent-node" data-idx="10"><div class="dot">💡</div><div class="lbl">Recommender</div></div>
    <div class="agent-node" data-idx="11"><div class="dot">📝</div><div class="lbl">Explainer</div></div>
  </div>

  <div class="form-row">
    <div class="form-group">
      <label for="repo-input">GitHub Repository (owner/repo)</label>
      <input type="text" id="repo-input" placeholder="e.g. Mintplex-Labs/anything-llm" value="Mintplex-Labs/anything-llm"/>
    </div>
    <div class="form-group">
      <label for="eval-mode">Evaluation Mode</label>
      <select id="eval-mode">
        <option value="resilient" selected>Resilient</option>
        <option value="strict">Strict</option>
      </select>
    </div>
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" id="btn-analyze" onclick="runAnalysis(false)">▶ Run Full Analysis</button>
    <button class="btn btn-secondary" id="btn-mock" onclick="runAnalysis(true)">🧪 Demo (Mock Data)</button>
  </div>

  <div id="status-bar">
    <div class="status-msg"><div class="pulse"></div><span id="status-text">Ready</span></div>
    <div id="agent-logs"></div>
  </div>
  <div class="error-banner" id="error-banner"><strong>⚠ Error</strong><pre id="error-text"></pre></div>
</div>

<!-- ═══ Results ═══ -->
<div id="results">

  <!-- Sprint health scores -->
  <div class="sec-title"><span>📊</span> Sprint Health Dashboard</div>
  <div class="score-grid" id="score-grid"></div>

  <!-- Run metrics (research) -->
  <div class="sec-title"><span>🔬</span> Research Metrics</div>
  <div class="metrics-panel" id="run-metrics"></div>

  <!-- Feature breakdown -->
  <div class="sec-title"><span>⚙️</span> Multi-Modal Feature Extraction (18 Metrics, 5 Categories)</div>
  <div class="two-col" id="features-panel"></div>

  <!-- Cross-repo dependency intelligence -->
  <div class="sec-title"><span>🔗</span> Cross-Repository Dependency Intelligence</div>
  <div class="dep-graph" id="dep-graph"></div>

  <!-- RAG evidence -->
  <div class="sec-title"><span>🧲</span> RAG Context &amp; Evidence Base</div>
  <div class="rag-panel" id="rag-panel"></div>

  <!-- Risks -->
  <div class="sec-title"><span>🛡️</span> Identified Risks <span id="risk-source-badge"></span></div>
  <div class="item-list" id="risks-list"></div>

  <!-- Recommendations -->
  <div class="sec-title"><span>💡</span> Recommended Interventions <span id="rec-source-badge"></span></div>
  <div class="item-list" id="recs-list"></div>

  <!-- Narrative -->
  <div class="sec-title"><span>📝</span> Sprint Intelligence Report (Explainer Agent)</div>
  <div id="narrative-box"></div>

  <!-- Execution logs -->
  <div class="sec-title acc-toggle" onclick="toggleLogs()">
    <span class="arrow">▶</span><span>📋</span> Execution Logs &amp; Errors
  </div>
  <div class="acc-body" id="exec-body">
    <div class="acc-content" id="exec-content"></div>
  </div>
</div>
</div>

<script>
/* ── Agent stages matching orchestrator.py node order ── */
const STAGES=[
  {name:"Router",desc:"Validating input & preparing state…"},
  {name:"Data Collector",desc:"Fetching GitHub issues, PRs, commits…"},
  {name:"Dependency Graph",desc:"Mapping cross-repo dependencies…"},
  {name:"Feature Engineer",desc:"Extracting 18 metrics across 5 categories…"},
  {name:"Synthetic Data",desc:"Generating cold-start bootstrapping data…"},
  {name:"Embedding Agent",desc:"Building RAG context via ChromaDB…"},
  {name:"LLM Reasoner",desc:"Predicting completion probability via Ollama…"},
  {name:"Sprint Analyzer",desc:"Computing composite health score…"},
  {name:"LoRA Training",desc:"Continuous learning — adapter check…"},
  {name:"Risk Assessor",desc:"Identifying blockers & velocity gaps…"},
  {name:"Recommender",desc:"Generating interventions from precedent…"},
  {name:"Explainer",desc:"Writing evidence-backed narrative report…"},
];
let animTimer=null,stageIdx=0;

function resetPipeline(){
  document.querySelectorAll('.agent-node').forEach(n=>{
    n.classList.remove('active','done','error')});
}

function advancePipeline(idx){
  const nodes=document.querySelectorAll('.agent-node');
  nodes.forEach((n,i)=>{
    if(i<idx) n.classList.add('done'),n.classList.remove('active');
    else if(i===idx) n.classList.add('active'),n.classList.remove('done');
    else n.classList.remove('active','done');
  });
}

async function runAnalysis(isMock){
  const btn=document.getElementById('btn-analyze'),
    btnM=document.getElementById('btn-mock'),
    sbar=document.getElementById('status-bar'),
    stxt=document.getElementById('status-text'),
    logs=document.getElementById('agent-logs'),
    err=document.getElementById('error-banner'),
    res=document.getElementById('results');

  btn.disabled=btnM.disabled=true;
  err.classList.remove('visible');res.classList.remove('visible');
  logs.innerHTML='';logs.classList.add('visible');
  sbar.classList.add('visible');resetPipeline();stageIdx=0;

  animTimer=setInterval(()=>{
    if(stageIdx<STAGES.length){
      advancePipeline(stageIdx);
      stxt.textContent=`[${stageIdx+1}/${STAGES.length}] ${STAGES[stageIdx].name} — ${STAGES[stageIdx].desc}`;
      const l=document.createElement('div');l.className='log-line';
      l.textContent=`[${new Date().toLocaleTimeString()}] Agent ${stageIdx+1}: ${STAGES[stageIdx].name} → ${STAGES[stageIdx].desc}`;
      logs.appendChild(l);logs.scrollTop=logs.scrollHeight;
      stageIdx++;
    }
  },1600);

  const repo=document.getElementById('repo-input').value.trim();
  const mode=document.getElementById('eval-mode').value;
  const url=isMock?'/api/analyze/mock':'/api/analyze';
  const payload=isMock?{eval_mode:mode}:{repositories:[repo],eval_mode:mode};

  try{
    const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    clearInterval(animTimer);
    const d=await r.json();
    if(!r.ok||d.error) throw{message:d.error||'Unknown error',traceback:d.traceback||''};

    // finish remaining stages
    while(stageIdx<STAGES.length){
      const l=document.createElement('div');l.className='log-line';
      l.textContent=`[${new Date().toLocaleTimeString()}] Agent ${stageIdx+1}: ${STAGES[stageIdx].name} ✓`;
      logs.appendChild(l);stageIdx++;
    }
    advancePipeline(STAGES.length);
    document.querySelectorAll('.agent-node').forEach(n=>n.classList.add('done'));
    stxt.textContent='✅ Pipeline complete — all 11 agents finished';
    renderResults(d.result);
  }catch(e){
    clearInterval(animTimer);
    stxt.textContent='❌ Pipeline failed';
    document.getElementById('error-text').textContent=(e.message||e)+(e.traceback?'\n\n'+e.traceback:'');
    err.classList.add('visible');
  }finally{btn.disabled=btnM.disabled=false}
}

/* ── Render ──────────────────────────────────────────────── */
function cc(v){return v>=70?'green':v>=45?'amber':'red'}
function sl(s){return{on_track:'On Track',at_risk:'At Risk',critical:'Critical'}[s]||s||'Unknown'}
function srcBadge(s){
  if(!s) return '';
  const cls=s==='llm'?'badge-llm':s==='fallback'?'badge-fallback':'badge-low';
  return `<span class="badge ${cls}">source: ${s}</span>`;
}

function renderResults(r){
  const a=r.sprint_analysis||{};
  const health=a.health_score??0, comp=a.completion_probability??0,
    deliv=a.delivery_score??0, mom=a.momentum_score??0,
    qual=a.quality_score??0, collab=a.collaboration_score??0,
    depRisk=a.dependency_risk_score??0;

  // Score grid
  document.getElementById('score-grid').innerHTML=`
    <div class="sc ${cc(health)}"><div class="lbl">Health Score</div><div class="val">${health.toFixed(1)}</div><div class="sub">${sl(a.health_status)}</div></div>
    <div class="sc ${cc(comp)}"><div class="lbl">Completion</div><div class="val">${comp.toFixed(0)}%</div><div class="sub">Predicted probability</div></div>
    <div class="sc ${cc(deliv)}"><div class="lbl">Delivery</div><div class="val">${deliv.toFixed(1)}</div><div class="sub">Issue + PR rate</div></div>
    <div class="sc ${cc(mom)}"><div class="lbl">Momentum</div><div class="val">${mom.toFixed(1)}</div><div class="sub">Commit frequency</div></div>
    <div class="sc ${cc(qual)}"><div class="lbl">Quality</div><div class="val">${qual.toFixed(1)}</div><div class="sub">Code concentration</div></div>
    <div class="sc ${cc(collab)}"><div class="lbl">Collaboration</div><div class="val">${collab.toFixed(1)}</div><div class="sub">Author participation</div></div>
    <div class="sc ${cc(100-depRisk)}"><div class="lbl">Dep Risk</div><div class="val">${depRisk.toFixed(1)}</div><div class="sub">Cross-repo propagation</div></div>
  `;

  // Run metrics
  const rm=r.run_metrics||{};
  const cq=rm.citation_quality||{};
  const counts=rm.counts||{};
  document.getElementById('run-metrics').innerHTML=`
    <div class="metric"><div class="m-label">Latency</div><div class="m-val">${(rm.latency_seconds??0).toFixed(2)}s</div><div class="m-sub">End-to-end pipeline</div></div>
    <div class="metric"><div class="m-label">F1 Score</div><div class="m-val">${rm.f1_score!==null&&rm.f1_score!==undefined?rm.f1_score.toFixed(3):'—'}</div><div class="m-sub">Target ≥0.85</div></div>
    <div class="metric"><div class="m-label">Parse Success</div><div class="m-val">${rm.parse_success_rate!==null?(rm.parse_success_rate*100).toFixed(0)+'%':'—'}</div><div class="m-sub">LLM output quality</div></div>
    <div class="metric"><div class="m-label">Fallback Rate</div><div class="m-val">${rm.fallback_rate!==null?(rm.fallback_rate*100).toFixed(0)+'%':'—'}</div><div class="m-sub">Deterministic fallback</div></div>
    <div class="metric"><div class="m-label">Citation Quality</div><div class="m-val">${(cq.score??0).toFixed(2)}</div><div class="m-sub">${cq.non_empty_citations??0}/${cq.total_citations??0} citations</div></div>
    <div class="metric"><div class="m-label">Analysis Source</div><div class="m-val" style="font-size:.85rem">${(rm.source_breakdown||{}).analysis||'—'}</div><div class="m-sub">LLM / fallback / error</div></div>
    <div class="metric"><div class="m-label">Risk Source</div><div class="m-val" style="font-size:.85rem">${(rm.source_breakdown||{}).risk||'—'}</div><div class="m-sub">${counts.risks??0} risks detected</div></div>
    <div class="metric"><div class="m-label">Rec Source</div><div class="m-val" style="font-size:.85rem">${(rm.source_breakdown||{}).recommendation||'—'}</div><div class="m-sub">${counts.recommendations??0} recommendations</div></div>
  `;

  // Features breakdown
  const feats=r.features||{};
  const catIcons={temporal:'🕐',activity:'📊',code:'💻',risk:'⚠️',team:'👥',language:'🔤'};
  let featHTML='';
  for(const[cat,metrics] of Object.entries(feats)){
    if(!metrics||typeof metrics!=='object') continue;
    let rows='';
    for(const[k,v] of Object.entries(metrics)){
      const display=typeof v==='number'?v.toFixed(3):String(v);
      rows+=`<div class="feat-row"><span class="fname">${k.replace(/_/g,' ')}</span><span class="fval">${display}</span></div>`;
    }
    if(rows) featHTML+=`<div class="feat-cat"><div class="feat-cat-title">${catIcons[cat]||'📐'} ${cat.toUpperCase()}</div>${rows}</div>`;
  }
  document.getElementById('features-panel').innerHTML=featHTML||'<div style="color:var(--text-muted);font-size:.78rem">No features extracted</div>';

  // Dependency graph
  const dep=r.dependency_graph||{};
  const nodes=dep.nodes||[];
  const edges=dep.edges||[];
  const propagation=dep.risk_propagation||{};
  let depHTML=`<div style="margin-bottom:.5rem;font-size:.7rem;color:var(--text-muted)">Nodes: ${nodes.length} repositories · Edges: ${edges.length} dependencies</div>`;
  if(nodes.length){
    depHTML+='<div style="margin-bottom:.5rem">'+nodes.map(n=>{
      const riskP=propagation[n];
      const riskStr=riskP!==undefined?` (propagation: ${(riskP*100).toFixed(0)}%)`:'';
      return `<span class="dep-node">${n}${riskStr}</span>`;
    }).join(' ')+'</div>';
  }
  if(edges.length){
    depHTML+=edges.map(e=>`<div class="dep-edge"><span>${e.source}</span><span class="arrow">→</span><span>${e.target}</span><span style="color:var(--text-muted)">[${e.type}]</span>${e.is_blocker?'<span class="badge badge-high">blocker</span>':''}</div>`).join('');
  }else{depHTML+='<div style="font-size:.72rem;color:var(--text-muted)">No cross-repo dependencies detected (single-repo analysis)</div>'}
  document.getElementById('dep-graph').innerHTML=depHTML;

  // RAG panel
  const sims=r.similar_sprint_ids||[];
  const cites=r.evidence_citations||[];
  const synth=r.synthetic_sprints||[];
  const synthVal=r.synthetic_validation||{};
  let ragHTML=`<div style="font-size:.72rem;margin-bottom:.4rem"><strong style="color:var(--cyan)">Similar Historical Sprints (RAG Retrieval):</strong> ${sims.length?sims.join(', '):'None retrieved'}</div>`;
  ragHTML+=`<div style="font-size:.72rem;margin-bottom:.4rem"><strong style="color:var(--cyan)">Synthetic Sprints Generated:</strong> ${synth.length} scenarios · Embedded: ${r.synthetic_embedded_count||0}</div>`;
  if(Object.keys(synthVal).length){ragHTML+=`<div style="font-size:.72rem;margin-bottom:.4rem"><strong style="color:var(--cyan)">Synthetic Validation:</strong> Realism score: ${(synthVal.realism_score??0).toFixed(3)}</div>`}
  ragHTML+=`<div style="font-size:.72rem"><strong style="color:var(--cyan)">Evidence Citations:</strong> ${cites.length?cites.join(', '):'No citations in this run'}</div>`;
  document.getElementById('rag-panel').innerHTML=ragHTML;

  // Risks
  document.getElementById('risk-source-badge').innerHTML=srcBadge(r.risk_source);
  const risks=r.identified_risks||[];
  document.getElementById('risks-list').innerHTML=risks.length?risks.map(risk=>{
    const sev=risk.severity??0;const b=sev>=.7?'high':sev>=.4?'medium':'low';
    return `<div class="item-card"><div class="ih"><span class="it">${(risk.risk_type||'Risk').replace(/_/g,' ')}</span><span class="badge badge-${b}">${b} · ${(sev*100).toFixed(0)}%</span></div><div class="id">${risk.description||''}</div>${risk.affected_issues?.length?`<div class="id" style="margin-top:.2rem;color:var(--amber)">Affected issues: ${risk.affected_issues.join(', ')}</div>`:''}</div>`;
  }).join(''):'<div class="item-card"><div class="id">No risks identified in this analysis.</div></div>';

  // Recommendations
  document.getElementById('rec-source-badge').innerHTML=srcBadge(r.recommendation_source);
  const recs=r.recommendations||[];
  document.getElementById('recs-list').innerHTML=recs.length?recs.map(rec=>{
    const p=rec.priority||'medium';
    return `<div class="item-card"><div class="ih"><span class="it">${rec.title||'Recommendation'}</span><span class="badge badge-${p}">${p}</span></div><div class="id">${rec.description||''}</div>${rec.action?`<div class="id" style="margin-top:.15rem;color:var(--green)">→ ${rec.action}</div>`:''}</div>`;
  }).join(''):'<div class="item-card"><div class="id">No recommendations generated.</div></div>';

  // Narrative
  const nar=r.narrative_explanation||'No narrative generated.';
  document.getElementById('narrative-box').innerHTML=nar
    .replace(/^## (.+)$/gm,'<strong style="color:#6366f1;font-size:.88rem">$1</strong>')
    .replace(/^# (.+)$/gm,'<strong style="color:var(--text);font-size:.95rem">$1</strong>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/^- /gm,'• ');

  // Execution logs
  const eLogs=r.execution_logs||[];const errs=r.errors||[];
  document.getElementById('exec-content').innerHTML=
    eLogs.map(l=>`<div>${l}</div>`).join('')+
    (errs.length?'<div style="color:var(--red);margin-top:.3rem">── Errors ──</div>':'')+
    errs.map(e=>`<div style="color:#fca5a5">⚠ ${e}</div>`).join('');

  document.getElementById('results').classList.add('visible');
}

function toggleLogs(){
  document.querySelector('.acc-toggle').classList.toggle('open');
  document.getElementById('exec-body').classList.toggle('open');
}
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
