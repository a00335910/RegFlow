"""Gradio dashboard for RegFlow.

Four tabs:
  1. Overview            — live counts across all stores (docs, obligations, gaps, conflicts, …)
  2. Obligation Explorer — pick an obligation, see Agent 4 gap + Agent 5 plan + Agent 6 pack
  3. Conflicts           — cross-jurisdiction Agent 3 output as a sortable table
  4. Submit Correction   — POST to the Override Store (closes the correction loop from the UI)

Run with:  python scripts/run_ui.py   (http://127.0.0.1:7860)
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

import gradio as gr
import pandas as pd
from sqlalchemy import func, select

from regflow.agents.audit_evidence.agent import render_markdown
from regflow.common.logging import get_logger
from regflow.common.types import CorrectionType
from regflow.db.postgres import (
    Article,
    AuditEvidenceRow,
    ConflictRow,
    Document,
    GapRow,
    ObligationRow,
    RemediationActionRow,
    get_session,
)
from regflow.db.vector import get_embedder, get_vector_store

log = get_logger(__name__)


# ============================================================================
# Data loaders (each returns plain Python so Gradio components can consume directly)
# ============================================================================


def overview_markdown() -> str:
    with get_session() as session:
        doc_count = session.scalar(select(func.count(Document.id))) or 0
        obl_count = session.scalar(select(func.count(ObligationRow.id))) or 0
        conflict_count = session.scalar(select(func.count(ConflictRow.id))) or 0
        action_count = session.scalar(select(func.count(RemediationActionRow.id))) or 0
        evidence_count = session.scalar(select(func.count(AuditEvidenceRow.id))) or 0
        gap_counts: dict[str, int] = dict(
            session.execute(
                select(GapRow.risk_level, func.count(GapRow.id)).group_by(GapRow.risk_level)
            ).all()
        )
        jurisdiction_breakdown = session.execute(
            select(Document.source, Document.jurisdiction, func.count(Document.id))
            .group_by(Document.source, Document.jurisdiction)
        ).all()

    jb_md = "\n".join(
        f"| {source} | {jurisdiction} | {count} |"
        for source, jurisdiction, count in jurisdiction_breakdown
    ) or "| _(no documents yet)_ | | |"

    return f"""## System overview

| Layer | Count |
|---|---|
| Documents ingested | **{doc_count}** |
| Obligations extracted (Agent 2) | **{obl_count}** |
| Cross-jurisdiction conflicts (Agent 3) | **{conflict_count}** |
| Gaps — **HIGH** risk | **{gap_counts.get('high', 0)}** |
| Gaps — MEDIUM risk | {gap_counts.get('medium', 0)} |
| Gaps — LOW risk | {gap_counts.get('low', 0)} |
| Remediation actions (Agent 5) | **{action_count}** |
| Audit evidence packs (Agent 6) | **{evidence_count}** |

### Documents by source & jurisdiction

| Source | Jurisdiction | Docs |
|---|---|---|
{jb_md}
"""


def list_obligations_choices() -> list[tuple[str, str]]:
    """Returns (label, value) tuples for a Gradio Dropdown.

    Sorted by Agent-2 confidence desc so the most reliable extractions come first.
    """
    with get_session() as session:
        rows = session.execute(
            select(
                ObligationRow.id,
                ObligationRow.jurisdiction,
                ObligationRow.regulator,
                ObligationRow.obligation_text,
            )
            .order_by(ObligationRow.confidence.desc())
            .limit(150)
        ).all()
    out: list[tuple[str, str]] = []
    for row in rows:
        oid, juris, regulator, text = row
        snippet = (text[:90] + "…") if len(text) > 90 else text
        label = f"[{juris}] {snippet}"
        out.append((label, str(oid)))
    return out


def obligation_detail(obligation_id_str: str | None) -> tuple[str, str, str, str]:
    """Returns (obligation_md, gap_md, plan_md, audit_pack_md)."""
    if not obligation_id_str:
        return ("_(pick an obligation above)_", "", "", "")

    try:
        oid = UUID(obligation_id_str)
    except (ValueError, TypeError):
        return ("_(invalid obligation id)_", "", "", "")

    with get_session() as session:
        obligation = session.get(ObligationRow, oid)
        if obligation is None:
            return ("_(obligation not found)_", "", "", "")
        article = session.get(Article, obligation.article_id)
        document = session.get(Document, obligation.document_id)

        obligation_md = _format_obligation(obligation, document, article)

        gap = session.execute(
            select(GapRow)
            .where(GapRow.obligation_id == oid)
            .order_by(GapRow.analyzed_at.desc())
            .limit(1)
        ).scalar_one_or_none()
        gap_md = _format_gap(gap) if gap else "_(no gap analysis yet — run Agent 4)_"

        actions = list(
            session.execute(
                select(RemediationActionRow)
                .where(RemediationActionRow.obligation_id == oid)
                .order_by(RemediationActionRow.priority.asc())
            ).scalars()
        )
        plan_md = _format_plan(actions) if actions else "_(no remediation plan yet — run Agent 5)_"

        evidence = session.execute(
            select(AuditEvidenceRow).where(AuditEvidenceRow.obligation_id == oid).limit(1)
        ).scalar_one_or_none()
        audit_md = (
            render_markdown(evidence, obligation, document, article)
            if evidence
            else "_(no audit pack yet — run Agent 6)_"
        )

    return obligation_md, gap_md, plan_md, audit_md


def list_conflicts_df() -> pd.DataFrame:
    with get_session() as session:
        rows = session.execute(
            select(
                ConflictRow.conflict_type,
                ConflictRow.severity,
                ConflictRow.confidence,
                ConflictRow.jurisdiction_a,
                ConflictRow.jurisdiction_b,
                ConflictRow.description,
            )
            .order_by(ConflictRow.confidence.desc())
        ).all()
    if not rows:
        return pd.DataFrame(
            columns=["conflict_type", "severity", "confidence", "from", "to", "description"]
        )
    df = pd.DataFrame(
        rows, columns=["conflict_type", "severity", "confidence", "from", "to", "description"]
    )
    df["confidence"] = df["confidence"].round(2)
    return df


def submit_correction(
    agent_id: str,
    correction_type: str,
    input_context: str,
    original_output_json: str,
    corrected_output_json: str,
    reviewer_id: str,
) -> str:
    """Posts a correction directly to both halves of the Override Store.

    Bypasses HTTP — calls the same persistence code path the FastAPI route uses.
    Means the UI works whether or not `scripts/run_api.py` is running.
    """
    if not input_context or len(input_context) < 10:
        return "✗ input_context must be at least 10 characters."
    try:
        original = json.loads(original_output_json) if original_output_json.strip() else {}
        corrected = json.loads(corrected_output_json) if corrected_output_json.strip() else {}
    except json.JSONDecodeError as e:
        return f"✗ JSON parse error: {e}"

    from datetime import datetime

    from regflow.db.postgres import CorrectionRecordRow, ReviewLogEntry

    try:
        ctype = CorrectionType(correction_type)
    except ValueError:
        return f"✗ unknown correction_type: {correction_type}"

    correction_id = uuid4()
    vector_uuid = str(correction_id)

    try:
        embedding = get_embedder().embed_one(input_context)
        get_vector_store().upsert_correction(
            correction_uuid=vector_uuid,
            agent_id=agent_id,
            correction_type=ctype.value,
            input_context=input_context,
            original_output=json.dumps(original),
            corrected_output=json.dumps(corrected),
            vector=embedding,
        )
        created_at = datetime.utcnow()
        with get_session() as session:
            session.add(
                CorrectionRecordRow(
                    id=correction_id,
                    agent_id=agent_id,
                    correction_type=ctype.value,
                    input_context=input_context,
                    original_output=original,
                    corrected_output=corrected,
                    reviewer_id=reviewer_id or "ui_reviewer",
                    vector_uuid=vector_uuid,
                    created_at=created_at,
                )
            )
            session.add(
                ReviewLogEntry(
                    trigger="correction_submitted",
                    agent_id=agent_id,
                    subject_type="correction_record",
                    subject_id=correction_id,
                    reviewer_id=reviewer_id or "ui_reviewer",
                    decision="modified",
                    notes="Submitted via Gradio UI",
                    payload={"correction_type": ctype.value},
                )
            )
        return (
            f"✓ Correction submitted.\n\n"
            f"- **correction_id**: `{correction_id}`\n"
            f"- **vector_uuid**: `{vector_uuid}`\n"
            f"- **agent_id**: `{agent_id}`\n"
            f"- **created_at**: `{created_at.isoformat()}`\n\n"
            f"Next call to `{agent_id}` on a semantically-similar input will retrieve this correction as a few-shot example."
        )
    except Exception as exc:        # noqa: BLE001 — surface anything to the UI user
        log.exception("ui.correction_submit_failed")
        return f"✗ {type(exc).__name__}: {exc}"


# ============================================================================
# Formatters (kept here so each tab's rendering is in one place)
# ============================================================================


def _format_obligation(obligation: ObligationRow, document, article) -> str:
    return f"""**Obligation:** {obligation.obligation_text}

| | |
|---|---|
| Source document | {document.title if document else '(unknown)'} (`{document.source_doc_id if document else '?'}`) |
| Article | {article.article_ref if article else '(unknown)'} |
| Jurisdiction | {obligation.jurisdiction} |
| Regulator | {obligation.regulator} |
| Type | {obligation.obligation_type} |
| Scope | {obligation.scope or '_(unspecified)_'} |
| Deadlines | {', '.join(obligation.deadlines) if obligation.deadlines else '_(none)_'} |
| Penalties | {', '.join(obligation.penalties) if obligation.penalties else '_(none)_'} |
| Agent 2 confidence | {obligation.confidence:.2f} |
"""


def _format_gap(gap: GapRow) -> str:
    matching = "\n".join(f"- {c}" for c in (gap.matching_controls or [])) or "_(none)_"
    missing = "\n".join(f"- {c}" for c in (gap.missing_or_weak_controls or [])) or "_(none)_"
    findings = "\n".join(f"- {f}" for f in (gap.related_audit_findings or [])) or "_(none)_"
    return f"""### Risk: **{gap.risk_level.upper()}**  (score: {gap.risk_score:.2f})

| Risk factor | Score |
|---|---|
| Enforcement severity | {gap.enforcement_severity:.2f} |
| Business impact | {gap.business_impact:.2f} |
| Deadline urgency | {gap.deadline_urgency:.2f} |
| Agent 4 confidence | {gap.confidence:.2f} |
| Evidence on file | {'✓' if gap.evidence_exists else '✗'} |

#### Matching controls

{matching}

#### Missing or weak controls

{missing}

#### Related audit findings & enforcement precedents

{findings}

#### Rationale

{gap.rationale or '_(none)_'}
"""


def _format_plan(actions: list[RemediationActionRow]) -> str:
    lines = ["### Remediation plan\n"]
    for a in actions:
        owner = a.suggested_owner or "**(UNASSIGNED)**"
        deadline = f"`{a.suggested_deadline}`" if a.suggested_deadline else "_(no deadline)_"
        lines.append(
            f"**[P{a.priority}]**  owner: {owner}  ·  deadline: {deadline}  ·  "
            f"confidence: {a.confidence:.2f}\n\n"
            f"{a.description}\n"
        )
        if a.proposed_control_updates:
            updates = "\n".join(f"  - {u}" for u in a.proposed_control_updates)
            lines.append(f"\n**Proposed control updates:**\n{updates}\n")
        if a.dependency_descriptions:
            deps = "\n".join(f"  - {d}" for d in a.dependency_descriptions)
            lines.append(f"\n**Dependencies:**\n{deps}\n")
        lines.append("\n---\n")
    return "\n".join(lines)


# ============================================================================
# Gradio Blocks
# ============================================================================


def build_app() -> gr.Blocks:
    with gr.Blocks(title="RegFlow Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            "# RegFlow — Multi-Agent Regulatory Compliance\n"
            "*Event-driven · Correction-retrieval-augmented · EU GDPR + US SEC/CFTC*"
        )

        # ------- Tab 1: Overview -------
        with gr.Tab("Overview"):
            overview_out = gr.Markdown(value=overview_markdown())
            refresh_overview = gr.Button("Refresh", size="sm")
            refresh_overview.click(fn=overview_markdown, outputs=overview_out)

        # ------- Tab 2: Obligation Explorer -------
        with gr.Tab("Obligation Explorer"):
            gr.Markdown(
                "Pick an obligation to see its full pipeline output: "
                "Agent 4 gap analysis, Agent 5 remediation plan, Agent 6 audit pack."
            )
            obligation_picker = gr.Dropdown(
                choices=list_obligations_choices(),
                label="Obligation",
                interactive=True,
            )
            refresh_obligations = gr.Button("Refresh obligation list", size="sm")

            with gr.Accordion("Obligation (Agent 2)", open=True):
                obligation_panel = gr.Markdown()
            with gr.Accordion("Gap Analysis (Agent 4)", open=True):
                gap_panel = gr.Markdown()
            with gr.Accordion("Remediation Plan (Agent 5)", open=False):
                plan_panel = gr.Markdown()
            with gr.Accordion("Audit Evidence Pack (Agent 6) — rendered", open=False):
                audit_panel = gr.Markdown()

            obligation_picker.change(
                fn=obligation_detail,
                inputs=obligation_picker,
                outputs=[obligation_panel, gap_panel, plan_panel, audit_panel],
            )
            refresh_obligations.click(
                fn=lambda: gr.update(choices=list_obligations_choices()),
                outputs=obligation_picker,
            )

        # ------- Tab 3: Cross-Jurisdiction Conflicts -------
        with gr.Tab("Cross-Jurisdiction Conflicts (Agent 3)"):
            gr.Markdown(
                "Conflicts detected between obligations across jurisdictions. "
                "`conflict_type` is one of `contradiction` / `overlap` / `stricter_standard`."
            )
            conflicts_table = gr.DataFrame(value=list_conflicts_df(), interactive=False)
            refresh_conflicts = gr.Button("Refresh", size="sm")
            refresh_conflicts.click(fn=list_conflicts_df, outputs=conflicts_table)

        # ------- Tab 4: Submit Correction -------
        with gr.Tab("Submit Correction (Override Store)"):
            gr.Markdown(
                "Submit a reviewer correction. The correction is embedded and stored in both halves "
                "of the Override Store. Next time the chosen agent encounters a semantically similar "
                "input, this correction is retrieved as a few-shot example.\n\n"
                "**This is the system's headline novelty made interactive.**"
            )
            with gr.Row():
                agent_dd = gr.Dropdown(
                    choices=[
                        "agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6"
                    ],
                    label="Agent being corrected",
                    value="agent_2",
                )
                ctype_dd = gr.Dropdown(
                    choices=[ct.value for ct in CorrectionType],
                    label="Correction type",
                    value=CorrectionType.WRONG_EXTRACTION.value,
                )
                reviewer_in = gr.Textbox(label="Reviewer id", value="ui_reviewer")

            input_ctx_in = gr.Textbox(
                label="Input context (article text or what the agent looked at)",
                placeholder="Paste the obligation text or article snippet the agent was analyzing…",
                lines=4,
            )
            with gr.Row():
                orig_in = gr.Code(
                    label="Original LLM output (JSON)",
                    language="json",
                    value="{}",
                    lines=8,
                )
                corr_in = gr.Code(
                    label="Corrected output (JSON)",
                    language="json",
                    value="{}",
                    lines=8,
                )

            submit_btn = gr.Button("Submit correction", variant="primary")
            submit_result = gr.Markdown()
            submit_btn.click(
                fn=submit_correction,
                inputs=[agent_dd, ctype_dd, input_ctx_in, orig_in, corr_in, reviewer_in],
                outputs=submit_result,
            )

        gr.Markdown(
            "---\n"
            "*RegFlow v0.1 · See [`architecture.pdf`](https://github.com/) for design spec · "
            "Defense layers: citation-required (L1) · verifier second pass (L2) · "
            "Override Store anti-examples (L3)*"
        )

    return app
