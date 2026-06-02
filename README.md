# RegFlow

> Event-driven, multi-agent regulatory compliance pipeline with a self-improving correction loop. Watches EU + US financial regulations, extracts machine-readable obligations, detects cross-jurisdiction conflicts, scores compliance gaps against NIST 800-53 controls and real public enforcement precedents, drafts remediation plans, and emits auditor-ready evidence packs — with layered LLM-hallucination defenses.

Built as a portfolio project to demonstrate **production-grade agentic system design** end-to-end. The full architectural spec lives in [`architecture.pdf`](architecture.pdf) (v2.3); this README documents the **operational** system that implements it.

---

## What's actually built and working

| | |
|---|---|
| **Sources** | EUR-Lex (GDPR — 281 articles ingested) + US Federal Register (5 SEC/CFTC rules, 89 articles) |
| **Enterprise context** | 111 NIST SP 800-53 Rev 5 controls (auto-imported from official OSCAL JSON) + 15 firm-specific SOPs + **10 real public enforcement actions** (BA £20M, Meta €1.2B, Amazon €746M, JPMorgan $200M, Equifax $700M, etc.) |
| **Agent 1 — Regulatory Radar** | Diff against prior version + LLM severity classification (cosmetic / minor / substantive / major). Emits `RegulatoryChangeEvent` only for substantive change. |
| **Agent 2 — Obligation Extractor** | Per-article LLM extraction with **Override Store retrieval-augmented prompting** (the headline novelty). Persists to Postgres + Neo4j + Weaviate. |
| **Agent 3 — Cross-Jurisdiction Conflict Detector** | Batch mode. Detects contradictions, overlaps, and stricter-standard relationships across EU/US obligations. Writes `(Obligation)-[:CONFLICTS_WITH]->(Obligation)` edges in Neo4j. |
| **Agent 4 — Gap Analyzer & Risk Scorer** | Maps obligations to NIST + firm controls, ties to prior enforcement precedents, computes `risk_score = enforcement_severity × business_impact × deadline_urgency`. Routes HIGH-risk to compliance approval. |
| **Agent 5 — Remediation Planner** | Generates 2-5 actions per gap with **constrained owner assignment** (only real owners from the enterprise roster), priorities, deadlines, and dependency notes. |
| **Agent 6 — Audit Evidence Generator** | Synthesizes auditor-ready evidence packs (clause citations + control links + justification narrative + open questions). **Exports to Markdown** for handover. |
| **Orchestrator** | LangGraph workflow: discovery → ingest → Agent 1 → confidence-gated routing (AUTO / NOTIFY / BLOCK) → dispatch to Agent 2 and beyond. |
| **Autonomous polling** | Prefect 3.x flow runs every 4 hours: discover new documents, ingest only what changed (content-hash idempotency), trigger the agent pipeline only for substantive changes. Zero LLM cost on quiet days. |
| **Human review** | FastAPI `POST /reviews/corrections` writes corrections to the Override Store (Postgres + Weaviate halves, joined by `vector_uuid`). Three trigger types (`orchestrator_block`, `orchestrator_notify`, `high_risk_gap`, `high_severity_conflict`, `correction_submitted`) feed one unified queue. |
| **Hallucination defense (Agent 6)** | Three-layer stack: **(L1)** citation-required generation enforced in prompt + validated against allow-lists; **(L2)** verifier second-pass LLM grades each claim against input data; **(L3)** Override Store retrieval injects past reviewer corrections as anti-examples. |

---

## Headline demonstration — the correction-retrieval loop

> ### What it claims
> A reviewer's correction to one obligation changes Agent 2's behavior on similar obligations on the next run — without retraining the LLM.

### Day 1 — Agent 2 on GDPR Article 5 (the "principles" article)

```
$ python scripts/demo_correction_loop.py
[Step 1/4] Day-1 run — Override Store is empty for this input.
[Day-1 Agent 2 output (no corrections influencing)]
  Count: 7
  All 7 typed as `governance`. Scope: "Controllers and processors must..."
```

### Day 2 — same article, after one correction submitted via API

```
[Step 2/4] Submitting reviewer correction to /reviews/corrections ...
  -> correction_id: 87c88c85-d677-43b6-99a8-af02a403c0a9

[Step 3/4] Day-2 run — Override Store now contains the reviewer correction.
agent_2.corrections_retrieved  article_ref=art_5 count=1 top_distance=5.96e-07

[Day-2 Agent 2 output (correction retrieved as few-shot)]
  Count: 7
  Item 5 now typed as `retention`     ← was `governance`
  Item 6 now typed as `security`      ← was `governance`
  Scope tightened: "Controllers must..."  ← was "Controllers AND processors"
```

**Same input, same model, different output** — driven entirely by the correction being retrieved at inference time. No fine-tuning, no model swap. **[See `scripts/demo_correction_loop.py`.](scripts/demo_correction_loop.py)**

> _Interview talking point: "We don't make the LLM deterministic — we make the **system** deterministic by accumulating corrections at the architecture level."_

---

## The three planes

```
            ┌─────────────────────────────────────────────────────────────┐
            │                EVENT PLANE                                   │
            │   Agent 1 (Regulatory Radar) — diff + severity classify     │
            │   Emits substantive changes ONLY; cosmetic dropped at source │
            └────────────────────────────┬────────────────────────────────┘
                                         │
            ┌────────────────────────────▼────────────────────────────────┐
            │                CONTROL PLANE                                 │
            │   LangGraph orchestrator:                                    │
            │     • routes AUTO / NOTIFY / BLOCK by confidence + severity  │
            │     • dispatches per-obligation work                         │
            │     • writes triggers to review_log                          │
            │   Prefect flow runs the whole thing on cron schedule         │
            └────────────────────────────┬────────────────────────────────┘
                                         │
            ┌────────────────────────────▼────────────────────────────────┐
            │                REASONING PLANE                               │
            │                                                              │
            │   Agent 2 ─► Agent 3 (batch, cross-jurisdiction)             │
            │             Agent 4 (per-obligation, gap + risk score)       │
            │             Agent 5 (per-gap, remediation plan)              │
            │             Agent 6 (per-obligation, audit evidence)         │
            │                                                              │
            │   All agents retrieve from the Override Store at inference   │
            └─────────────────────────────────────────────────────────────┘

            ┌─────────────────────────────────────────────────────────────┐
            │                SHARED MEMORY                                 │
            │   Postgres  Neo4j  Weaviate  MinIO  Enterprise Context       │
            │                                                              │
            │   Override Store ← FastAPI /reviews/corrections              │
            │   (Postgres rows + Weaviate vectors, joined by vector_uuid)  │
            └─────────────────────────────────────────────────────────────┘
```

---


## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | LangGraph | Stateful multi-agent workflows; right abstraction for agent-level state transitions |
| LLM serving | Ollama Cloud (`gpt-oss:20b-cloud`) via LiteLLM | Provider-agnostic abstraction; same code runs against local Ollama or any OpenAI-compatible endpoint |
| Embeddings | BAAI/BGE-M3 (sentence-transformers) | Multilingual; required for EU+US cross-jurisdiction work. Input length capped at 6000 chars to avoid internal chunking on CPU. |
| Vector store | Weaviate v4 | Two collections (`RegulatoryCorpus`, `OverrideStore`) with `Vectorizer.none()` — we bring our own vectors |
| Knowledge graph | Neo4j 5.25 Community | Native graph storage for cross-jurisdiction conflict edges + obligation→gap→action chain queries |
| Relational | PostgreSQL 16 | Source of truth for all structured rows + audit trail |
| Raw blob store | MinIO (S3-compatible) | Stores every fetched HTML/XML/PDF content-hashed; legal traceability |
| API | FastAPI + Pydantic | Auto-generated OpenAPI/Swagger UI for the review endpoint |
| Autonomous polling | Prefect 3.x | Cron-scheduled flow with per-task retries and a UI dashboard |
| Compliance reference | NIST SP 800-53 Rev 5 (OSCAL JSON) | Industry-standard control catalog; auto-downloaded from `usnistgov/oscal-content` |

---

## Demo walkthrough

> **Prerequisites:** Python 3.11 or 3.12, Docker, ~6GB free disk for models. See **Setup** below for first-time install.

### 1. Initialize infrastructure + ingest GDPR + ingest 5 SEC/CFTC rules

```powershell
docker compose up -d
python scripts/init_infra.py
python scripts/load_nist_800_53.py            # 111 NIST controls
python scripts/load_enterprise_context.py     # 15 firm SOPs + 10 real enforcement findings
python scripts/ingest_eurlex.py 32016R0679 --from-file data/samples/gdpr.html
python scripts/ingest_federal_register.py     # discovers + ingests 5 SEC rules via FR API
```

### 2. Drive the agent pipeline

```powershell
# Agent 1 + Agent 2 via orchestrator
python scripts/run_workflow.py 32016R0679 --only-articles
python scripts/run_workflow.py 2026-04202     # one of the SEC rules

# Agent 3 cross-jurisdiction sweep
python scripts/run_conflict_detector.py --sweep --limit 10

# Agent 4 gap analysis
python scripts/run_gap_analyzer.py --sweep --limit 10

# Agent 5 remediation plans (HIGH-risk gaps only)
python scripts/run_remediation_planner.py --sweep --risk high

# Agent 6 audit evidence packs
python scripts/run_audit_evidence.py --sweep --limit 5
```

### 3. The correction-loop demo (the headline)

```powershell
# Terminal 1
python scripts/run_api.py        # FastAPI on http://127.0.0.1:8000

# Terminal 2
python scripts/demo_correction_loop.py
```

### 4. Export an audit pack as markdown

```powershell
python scripts/export_audit_pack.py <obligation-uuid>
# writes data/exports/audit_pack_<short>.md
```

### 5. View the full system as a graph (Neo4j Browser)

Open <http://localhost:7474> (`neo4j` / `regflow_dev`), then run:

```cypher
MATCH (o:Obligation)-[:HAS_GAP]->(g:Gap)
OPTIONAL MATCH (g)-[:HAS_ACTION]->(a:RemediationAction)
OPTIONAL MATCH (o)-[:HAS_EVIDENCE]->(e:AuditEvidence)
OPTIONAL MATCH (o)-[r:CONFLICTS_WITH]-(other:Obligation)
WHERE g.risk_level = "high"
RETURN o, g, a, e, r, other
LIMIT 30
```

This single query renders the entire pipeline's output: regulation → obligation → gap → remediation actions → audit evidence → cross-jurisdiction conflicts.

### 6. Autonomous polling (production posture)

```powershell
prefect server start                          # http://localhost:4200 (UI)
python scripts/serve_ingestion_schedule.py    # fires every 4 hours
```

---

## What's deferred (and why)

| | Status | Reason |
|---|---|---|
| LLM logprob-based confidence calibration | Not built | `gpt-oss:20b-cloud` doesn't expose logprobs; architecture v3 roadmap item (lines 187-196) |
| Self-consistency sampling (3× generations for high-stakes claims) | Not built | 3× cloud quota; would deploy with paid LLM API |
| Real-time webhook ingestion (vs polling) | Not built | Only some sources support webhooks; polling works universally |
| Dashboard UI | Not built | Backend-focused project; Neo4j Browser + markdown exports cover the visualization story |
| EUR-Lex auto-ingestion in Prefect flow | Manual via `--from-file` | EUR-Lex is Cloudflare-fronted; SOAP API needs registration (multi-day process) |
| Sentence-level granular verifier | Not built | Current paragraph-level verifier is acceptable; sentence-level reduces FPs further |

These are explicitly deferred, not missing. Each has a documented reason and a credible path to v2.

---

## Setup

### 1. Python 3.11 or 3.12 (NOT 3.13 — torch wheels lag)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
```

### 2. Docker services

```powershell
docker compose up -d
docker compose ps        # wait until all 5 services show 'healthy'
```

Services started: `postgres`, `neo4j`, `weaviate`, `minio`, `ollama`. Default ports: Neo4j Browser <http://localhost:7474>, MinIO Console <http://localhost:9001>, Weaviate <http://localhost:8080>, FastAPI <http://localhost:8000>, Prefect UI <http://localhost:4200>.

### 3. Initialize stores + enterprise context

```powershell
python scripts/init_infra.py                  # creates tables, MinIO bucket, Weaviate collections, Neo4j constraints
python scripts/load_nist_800_53.py            # auto-downloads NIST OSCAL JSON, imports 111 controls
python scripts/load_enterprise_context.py     # firm SOPs + real enforcement findings
```

### 4. LLM access (one-time)

```powershell
# Authenticate to Ollama Cloud (free tier, no card)
docker exec -it regflow-ollama ollama signin
```

`config/settings.yaml` is set to `gpt-oss:20b-cloud` by default. To run fully local instead:

```powershell
docker exec -it regflow-ollama ollama pull llama3.1:8b
# uncomment the "FULL LOCAL" block in config/settings.yaml
```

### 5. Ingest sample data

```powershell
python scripts/ingest_eurlex.py 32016R0679 --from-file data/samples/gdpr.html
python scripts/ingest_federal_register.py
```

That's everything. The full agent pipeline runs against this data.

---

## Repository layout

```
src/regflow/
  agents/
    regulatory_radar/        # Agent 1: diff + severity classification
    obligation_extractor/    # Agent 2: + Override Store retrieval
    conflict_detector/       # Agent 3: batch cross-jurisdiction
    gap_analyzer/            # Agent 4: NIST mapping + risk score
    remediation_planner/     # Agent 5: constrained owner assignment
    audit_evidence/          # Agent 6: 3-layer hallucination defense
  api/                       # FastAPI human-review endpoint
  common/                    # settings, logging, shared Pydantic types, LiteLLM wrapper
  db/                        # Postgres, Weaviate, MinIO, Neo4j clients
  feeds/
    connectors/              # eur_lex, federal_register (FeedConnector Protocol)
    parsers/                 # eurlex_html, federal_register_html
    pipeline.py              # source-agnostic ingestion
  flows/                     # Prefect ingestion_flow
  orchestrator/              # LangGraph state, router, workflow
  rag/                       # Override Store retriever
config/                      # settings.yaml, feeds.yaml
data/sample_controls/        # NIST cache, firm SOPs YAML, real audit findings YAML
data/samples/                # gdpr.html (raw)
data/exports/                # generated audit pack markdown files
scripts/                     # one driver per task (ingest, run_workflow, demo_*, export_*)
tests/                       # unit / integration / e2e
docker-compose.yaml
architecture.pdf             # v2.3 design spec (source of truth)
```

---

## Known limitations (honest assessment)

- **EUR-Lex Cloudflare fronting** means automated ingestion currently falls back to `--from-file`. The Cellar API fallback is implemented but returns 404s for the documents in our feed (likely due to URL-pattern variation per CELEX number).
- **Federal Register parser** is heading-based (`<h1>`-`<h4>` + legacy `<HD>` tags) with a min-length filter (250 chars) to merge tiny sub-sections. Some FR documents with non-standard markup still produce single "Document" articles via the fallback path.
- **Self-reported LLM confidence is poorly calibrated** (architecture line 181). v3 work would add logprob-derived confidence and a verifier-model-tuned threshold per agent.
- **Override Store retrieval is unfiltered by jurisdiction by default** — a GDPR correction can be retrieved when processing an SEC obligation if semantically similar. Tunable via `correction_type` filter and `max_distance` threshold in `rag/override_retriever.py`.

---

## License

Apache-2.0. NIST SP 800-53 content is U.S. government work, public domain. Real enforcement findings reference publicly documented regulatory actions.
