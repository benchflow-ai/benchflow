"""Viewer trace steps → a page the current viewer's primitives render.

> **PROVISIONAL.** Part of the IR family (`docs/trace-interop.md` §8.7). It is
> the family's only module that imports a runtime module, and nothing in a run
> path imports *it* — see §8.15.

`ir_to_view.ir_to_view_steps` projects a canonical trace onto the viewer's step
vocabulary. This module turns that step list into one HTML page, using the card
builders `benchflow.trajectories.viewer` already emits its ACP page with. The
chain is::

    <format> -> ir_from_* -> CanonicalTrace -> ir_to_view_steps -> here -> page

and the direction of the dependency is one-way: this module imports the viewer,
the viewer never imports the IR. A step list is a plain document — six kinds,
named keys — so the renderer stays a renderer.

## What it is not

It does not rebuild the steps into ACP capture events and hand them to
`_render_acp_events`. That would be a lie in the middle of the chain: the IR
holds records ACP has no type for (`oracle`, anything `unknown`), a status ACP
cannot spell, and a tool name whose *semantics* are the whole point. Forging
capture events would launder all three back into the shape the conversion
exists to stop assuming.

## What the current page gains, and why those are additions rather than fixes

Measured against `viewer._render_acp_events` on the two captured rollouts:

- an `agent_timeout` reaches **no card** there — H2's four events render three
  blocks and the word "timeout" appears nowhere on the page;
- an unrecognized record reaches no card either;
- a tool card carries kind, title and status, and **not** the tool's output.

None of that is broken: those are the four branches that renderer has. This
edge has six step kinds to place, so it places them, and declares the
difference rather than presenting it as a repair.

## Reasoning that arrives with an action

A step may carry `reasoning` without being a reasoning step — the shape ATIF
produces, since it folds a thought into the agent step it precedes. That value
is rendered **inside the same card**, above the card's own content, in the
stylesheet's existing `.thinking` style. It does not become a second card:
`ir_to_view` refuses to invent an event boundary the source never declared, and
inventing one here instead would be the same fabrication one layer down.

## The classification rule

The hue arrives already decided by `ir_to_view`, which emits a category only
when the source said the string *is* one. Here it is mapped to an accent class
by **membership in a table**, never by inspecting the string.
`viewer._tool_accent_class` — which scans the kind and then the title for
needles, so a `function_name` of `read_file` becomes the read accent — is
deliberately not imported, and a test pins that it is not.
"""

from __future__ import annotations

import html
import json
from typing import Any, NamedTuple

from benchflow.trajectories import viewer
from benchflow.trajectories.ir import CanonicalTrace, LossClass, LossReport, PathSpace
from benchflow.trajectories.ir_to_view import (
    NEUTRAL_HUE,
    VIEW_TOOL_HUES,
    ir_to_view_steps,
)

LOSS_DIRECTION = "view->html"
"""The one edge in the family with the IR on **neither** side.

:class:`~benchflow.trajectories.ir.PathSpace` is defined relative to the hub, so
its two non-hub members are read here as: ``SOURCE`` addresses the step list
this module was given, ``TARGET`` addresses the page it produced. ``HUB`` is
never used, so no record of this edge can be joined to a hub path by mistake.
"""

HUE_ACCENT: dict[str, str] = {
    "execute": "acc-bash",
    "read": "acc-read",
    "edit": "acc-edit",
    "fetch": "acc-web",
    "search": "acc-web",
    "skill": "acc-agent",
    "think": "acc-other",
    "other": "acc-other",
}
"""Display hue → the accent class the current stylesheet defines.

A table, checked by membership. Two hues map to ``acc-other`` for opposite
reasons: ``other`` because no category was observed, and ``think`` because the
stylesheet has no accent for one — the second is a real loss and is declared as
such. Adding an accent for ``think`` is a viewer decision, not ours to take
inside a converter.
"""

NEUTRAL_ACCENT = "acc-other"

DIAGNOSTIC_LABEL = "Canonical IR representation"
"""What the card for an untypeable event says it is showing.

`ir_to_view` renders those events as a serialization of the **canonical IR
event**; the IR holds no source record to show instead. A page that displayed
it under the source's own type would be claiming a document it does not have.
"""

TEXT_PREVIEW = 500
"""Message/thought/prompt cut, the same one the legacy ACP card uses."""

TOOL_OUTPUT_PREVIEW = 2000
DIAGNOSTIC_PREVIEW = 4000


def _truncate(text: str, limit: int, where: str, losses: LossReport) -> str:
    """Cut *text* to *limit* with a visible marker and a declared record.

    The cut happens **before** escaping — a marker is appended to the plain
    string, then the whole thing is escaped once — so it can never bisect an
    entity. Silence is the thing being avoided here: a page that shortens a
    tool's output without saying so is a page that lies about the run.
    """
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    losses.add(
        where,
        LossClass.NORMALIZED,
        f"shown to {limit} characters of {len(text)}; the page carries an "
        f"explicit marker for the {dropped} it does not show",
        space=PathSpace.SOURCE,
    )
    return f"{text[:limit]}\n… [truncated, {dropped} more characters]"


def _esc(text: str, limit: int, where: str, losses: LossReport) -> str:
    return html.escape(_truncate(text, limit, where, losses))


def _accent(hue: Any, where: str, losses: LossReport) -> str:
    """The accent class for an already-decided hue. No string is inspected."""
    if hue not in VIEW_TOOL_HUES:
        losses.add(
            where,
            LossClass.DROPPED,
            f"{hue!r} is outside the display vocabulary this edge was written "
            f"against; the neutral accent was used, which asserts no category",
            space=PathSpace.SOURCE,
        )
        return NEUTRAL_ACCENT
    accent = HUE_ACCENT[hue]
    if hue != NEUTRAL_HUE and accent == NEUTRAL_ACCENT:
        losses.add(
            where,
            LossClass.DROPPED,
            f"the observed category {hue!r} has no accent in the viewer's "
            f"stylesheet, so the strip stays neutral; the category itself is "
            f"still legible — it is the card's own label",
            space=PathSpace.SOURCE,
        )
    return accent


def _data_attrs(pairs: dict[str, Any]) -> str:
    """``data-*`` attributes for the values a card has no visible slot for."""
    out = []
    for name, value in pairs.items():
        if value is None:
            continue
        out.append(f' data-{name}="{html.escape(str(value), quote=True)}"')
    return "".join(out)


def _reasoning_html(step: dict[str, Any], where: str, losses: LossReport) -> str:
    """The `steps[].reasoning` block, in the stylesheet's own thinking style.

    It rides **inside** the card of the step that carried it, above that step's
    own content. Reasoning observed alongside an action is not a second event —
    `ir_to_view` refuses to invent one — so it must not become a second card
    here either.
    """
    reasoning = step.get("reasoning")
    if reasoning is None:
        return ""
    return (
        f'<div class="thinking">'
        f"{_esc(str(reasoning), TEXT_PREVIEW, f'{where}.reasoning', losses)}"
        f"</div>"
    )


def _prompt_card(
    label: str, escaped_text: str, step: dict[str, Any], where: str, losses: LossReport
) -> str:
    """The viewer's prompt card, with a reasoning block when the step has one.

    Without reasoning this *is* `viewer._prompt_block`. With it, the same
    markup gains one `.thinking` div before the message — a test pins the two
    against each other so this copy cannot drift from the original.
    """
    reasoning = _reasoning_html(step, where, losses)
    if not reasoning:
        return viewer._prompt_block(label, escaped_text)
    return (
        f'<div class="step prompt">'
        f'<div class="step-header"><span class="label prompt">{label}</span></div>'
        f'{reasoning}<div class="msg">{escaped_text}</div>'
        f"</div>"
    )


def _message_card(
    escaped_text: str, step: dict[str, Any], where: str, losses: LossReport
) -> str:
    """The viewer's agent-message card, with a reasoning block when present."""
    reasoning = _reasoning_html(step, where, losses)
    if not reasoning:
        return viewer._message_block(escaped_text)
    return (
        f'<div class="step agent">{reasoning}'
        f'<div class="msg">{escaped_text}</div></div>'
    )


def _tool_card(step: dict[str, Any], where: str, losses: LossReport) -> str:
    """A tool call: the legacy card's three fields, plus what it has no slot for.

    ``name_semantics`` rides in the metrics line *and* in a ``data-`` attribute:
    the first so a reader sees whether ``execute`` was a category or a function
    name, the second so a browser check can assert it without parsing prose.
    """
    tool = step["tool"]
    accent = _accent(tool.get("hue"), f"{where}.tool.hue", losses)
    semantics = tool.get("name_semantics")

    kind = html.escape(str(tool.get("kind", "")))
    title = html.escape(str(tool.get("title", "")))
    status = html.escape(str(tool.get("status", "")))
    meta = (
        f"{status or '—'} · name_semantics: {html.escape(str(semantics or 'unknown'))}"
    )

    if tool.get("id"):
        losses.add(
            f"{where}.tool.id",
            LossClass.DROPPED,
            "the tool call id is carried in a data- attribute; the card has no "
            "visible slot for it",
            space=PathSpace.SOURCE,
        )

    body = ""
    content = tool.get("content") or []
    if content:
        joined = "\n\n".join(str(item) for item in content)
        body = (
            f'<div class="tool-args">'
            f"{_esc(joined, TOOL_OUTPUT_PREVIEW, f'{where}.tool.content', losses)}"
            f"</div>"
        )

    attrs = _data_attrs(
        {
            "name-semantics": semantics,
            "hue": tool.get("hue"),
            "tool-id": tool.get("id") or None,
            "source-type": step.get("type"),
        }
    )
    return (
        f'<div class="step agent tool-step {accent}"{attrs}>'
        f"{_reasoning_html(step, where, losses)}"
        f'<div class="tool"><span class="tool-name">{kind}</span> {title}</div>'
        f'<div class="metrics">{meta}</div>'
        f"{body}"
        f"</div>"
    )


def _timeout_card(step: dict[str, Any], where: str, losses: LossReport) -> str:
    """A typed timeout — the kind the current renderer has no branch for."""
    info = step["timeout"]
    reason = html.escape(str(info.get("reason") or ""))
    pending = info.get("pending") or []
    details = [f"timeout_sec: {html.escape(str(info.get('timeout_sec')))}"]
    details.append(f"pending tool calls: {html.escape(str(len(pending)))}")
    if pending:
        details.append(html.escape(", ".join(str(p) for p in pending)))
    details.append(
        f"terminal trajectory complete: {html.escape(str(info.get('complete')))}"
    )
    attrs = _data_attrs({"step-kind": "timeout", "source-type": step.get("type")})
    return (
        f'<div class="step agent tool-step {NEUTRAL_ACCENT}"{attrs}>'
        f"{_reasoning_html(step, where, losses)}"
        f'<div class="tool"><span class="tool-name">agent timeout</span> {reason}</div>'
        f'<div class="metrics">{" · ".join(details)}</div>'
        f"</div>"
    )


def _diagnostic_card(step: dict[str, Any], where: str, losses: LossReport) -> str:
    """An event with no typed slot, labelled for what its body actually is."""
    source_type = html.escape(str(step.get("type") or "no source type"))
    text = str(step.get("text") or "")
    body = ""
    if text:
        body = (
            f'<div class="tool-args">'
            f"{_esc(text, DIAGNOSTIC_PREVIEW, f'{where}.text', losses)}"
            f"</div>"
        )
    attrs = _data_attrs(
        {
            "step-kind": "unknown",
            "source-type": step.get("type"),
            "diagnostic": "canonical-ir",
        }
    )
    return (
        f'<div class="step agent tool-step {NEUTRAL_ACCENT}"{attrs}>'
        f'<div class="tool"><span class="tool-name">{html.escape(DIAGNOSTIC_LABEL)}</span> '
        f"{source_type}</div>"
        f"{body}"
        f"</div>"
    )


def _provenance_card(step_count: int, source_format: str | None) -> str:
    """One line saying which renderer produced the page below it."""
    source = html.escape(source_format or "canonical trace")
    return (
        f'<div class="step" data-provenance="canonical-ir">'
        f'<div class="metrics">Rendered from the canonical Trace IR '
        f"({source}) — {step_count} steps, one per canonical event.</div>"
        f"</div>"
    )


def view_steps_to_html(
    title: str,
    steps: list[dict[str, Any]],
    result_data: dict[str, Any] | None = None,
    *,
    prompts: list[str] | None = None,
    source_format: str | None = None,
) -> tuple[str, LossReport]:
    """Render a viewer step list as one page, with what the page cannot hold.

    *prompts* follows the legacy renderer's own rule — the run's ``prompts.json``
    is shown only when the step list carries no prompt of its own, because
    showing both prints the same text twice. Prompt ordinals are a property of
    the run, not of the trace, which is why they are assigned here.

    *result_data* is ``result.json`` and reaches the page through the viewer's
    own summary card, unchanged.
    """
    losses = LossReport(direction=LOSS_DIRECTION)
    blocks: list[str] = [_provenance_card(len(steps), source_format)]

    prompt_counter = 0
    if not any(step.get("kind") == "prompt" for step in steps):
        for position, prompt in enumerate(prompts or []):
            prompt_counter += 1
            blocks.append(
                viewer._prompt_block(
                    f"PROMPT {prompt_counter}",
                    _esc(str(prompt), TEXT_PREVIEW, f"prompts[{position}]", losses),
                )
            )

    for index, step in enumerate(steps):
        where = f"steps[{index}]"
        kind = step.get("kind")

        if kind == "prompt":
            prompt_counter += 1
            blocks.append(
                _prompt_card(
                    f"PROMPT {prompt_counter}",
                    _esc(
                        str(step.get("text") or ""),
                        TEXT_PREVIEW,
                        f"{where}.text",
                        losses,
                    ),
                    step,
                    where,
                    losses,
                )
            )
        elif kind == "message":
            blocks.append(
                _message_card(
                    _esc(
                        str(step.get("text") or ""),
                        TEXT_PREVIEW,
                        f"{where}.text",
                        losses,
                    ),
                    step,
                    where,
                    losses,
                )
            )
        elif kind == "thought":
            blocks.append(
                viewer._thought_block(
                    _esc(
                        str(step.get("text") or ""),
                        TEXT_PREVIEW,
                        f"{where}.text",
                        losses,
                    )
                )
            )
        elif kind == "tool" and isinstance(step.get("tool"), dict):
            blocks.append(_tool_card(step, where, losses))
        elif kind == "timeout" and isinstance(step.get("timeout"), dict):
            blocks.append(_timeout_card(step, where, losses))
        else:
            # Everything else — 'unknown', and a typed kind whose payload is
            # missing — becomes a labelled diagnostic. No step is skipped.
            blocks.append(_diagnostic_card(step, where, losses))
            if kind not in (None, "unknown"):
                losses.add(
                    where,
                    LossClass.NORMALIZED,
                    f"a {kind!r} step carrying no {kind!r} payload was rendered "
                    "as a diagnostic rather than dropped",
                    space=PathSpace.SOURCE,
                )

        if kind not in ("tool", "timeout", "unknown") and step.get("type") is not None:
            losses.add(
                f"{where}.type",
                LossClass.DROPPED,
                "the source record's own type reaches no field on a prompt, "
                "message or thought card; those cards are the viewer's and are "
                "emitted unchanged",
                space=PathSpace.SOURCE,
            )

    _declare_page_level(steps, losses)

    if result_data:
        blocks.append(viewer._result_block(result_data))

    return viewer._page(title, blocks), losses


def _declare_page_level(steps: list[dict[str, Any]], losses: LossReport) -> None:
    """Properties of the page, declared once instead of per step."""
    if any("t" in step for step in steps):
        losses.add(
            "steps[].t",
            LossClass.DROPPED,
            "observed timestamps reach no field: this page has no timeline",
            space=PathSpace.SOURCE,
        )
    if any("dur" in step for step in steps):
        losses.add(
            "steps[].dur",
            LossClass.DROPPED,
            "observed tool durations reach no field on this page",
            space=PathSpace.SOURCE,
        )
    if steps:
        losses.add(
            "steps[].i",
            LossClass.DROPPED,
            "step positions are not printed; the page keeps them as document "
            "order and nothing renumbers",
            space=PathSpace.SOURCE,
        )
        losses.add(
            "provenance",
            LossClass.SYNTHESIZED,
            "the page opens with a line naming the renderer that produced it; "
            "no step carries that text",
            space=PathSpace.TARGET,
        )


class RenderedTrace(NamedTuple):
    """A page plus the two reports that describe how it was reached.

    The reports are **not** merged. They address different documents — one the
    canonical trace, the other the step list — and a single report claiming
    both would have to name a space neither edge has.
    """

    html: str
    steps_losses: LossReport
    page_losses: LossReport


def render_trace(
    title: str,
    trace: CanonicalTrace,
    result_data: dict[str, Any] | None = None,
    *,
    prompts: list[str] | None = None,
) -> RenderedTrace:
    """Render a canonical trace: `ir_to_view_steps`, then this page.

    The trace is not modified and its own inbound report is left alone.
    """
    steps, steps_losses = ir_to_view_steps(trace)
    page, page_losses = view_steps_to_html(
        title,
        steps,
        result_data,
        prompts=prompts,
        source_format=trace.provenance.source_format,
    )
    return RenderedTrace(page, steps_losses, page_losses)


# ---------------------------------------------------------------------------
# Reading a rollout directory
# ---------------------------------------------------------------------------


def _load(path: Any) -> Any:
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))


def rollout_to_trace(rollout_dir: Any) -> tuple[CanonicalTrace, str] | None:
    """Rebuild a canonical trace from what the rollout directory actually holds.

    The ACP capture first — it is the artifact the viewer has always rendered —
    then ``trainer/atif.json``, whose path is read from ``export_atif`` rather
    than restated here. ``None`` when the directory holds neither.

    OTLP is absent on purpose: nothing in this repository writes spans into a
    rollout directory, so there is no filename to look for. That format reaches
    the page through :func:`render_trace` with a trace already in hand.
    """
    from pathlib import Path

    from benchflow.trajectories.export_atif import ROLLOUT_ATIF_RELPATH
    from benchflow.trajectories.ir_from_acp import acp_events_to_ir
    from benchflow.trajectories.ir_from_atif import atif_to_ir

    rollout_dir = Path(rollout_dir)
    acp = rollout_dir / "trajectory" / "acp_trajectory.jsonl"
    if acp.exists():
        try:
            text = acp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return acp_events_to_ir(viewer._parse_jsonl(text)), "acp capture"

    atif = rollout_dir / ROLLOUT_ATIF_RELPATH
    if atif.exists():
        document = _load(atif)
        if not isinstance(document, dict):
            return None
        return atif_to_ir(document), ROLLOUT_ATIF_RELPATH
    return None


def render_rollout_page(
    rollout_dir: Any, prompts: list[str] | None = None
) -> RenderedTrace | None:
    """One rollout directory as a canonical-IR page, or ``None`` if it has none.

    The run summary comes from ``result.json`` through the viewer's own card,
    and the prompts from the caller — neither is a function of the trace.
    """
    from pathlib import Path

    rollout_dir = Path(rollout_dir)
    built = rollout_to_trace(rollout_dir)
    if built is None:
        return None
    trace, _source = built
    return render_trace(
        rollout_dir.name,
        trace,
        viewer._load_result_json(rollout_dir),
        prompts=prompts,
    )


def _main(argv: list[str]) -> int:
    """``python -m benchflow.trajectories.ir_to_view_html <path> [out.html]``.

    *path* is a rollout directory (its ACP capture, else its ``trainer/atif.json``),
    an ACP ``.jsonl`` session file, an ATIF document, or an OTLP/JSON export.
    Nothing in BenchFlow calls this; it exists so a person can look at the page.
    """
    from pathlib import Path

    from benchflow.trajectories.export_atif import ROLLOUT_ATIF_RELPATH
    from benchflow.trajectories.ir_from_acp import acp_events_to_ir
    from benchflow.trajectories.ir_from_atif import atif_to_ir
    from benchflow.trajectories.ir_from_otel import otlp_json_to_ir

    if not argv:
        print(
            "usage: python -m benchflow.trajectories.ir_to_view_html "
            "<rollout_dir|acp.jsonl|atif.json|otlp.json> [out.html]"
        )
        return 2

    path = Path(argv[0])
    out = Path(argv[1]) if len(argv) > 1 else path.with_suffix(".ir.html")
    result_data: dict[str, Any] = {}
    prompts: list[str] | None = None
    traces: list[CanonicalTrace] = []

    if path.is_dir():
        result_data = viewer._load_result_json(path)
        prompts_path = path / "prompts.json"
        if prompts_path.exists():
            loaded = _load(prompts_path)
            prompts = loaded if isinstance(loaded, list) else None
        built = rollout_to_trace(path)
        if built is None:
            print(f"no ACP capture and no {ROLLOUT_ATIF_RELPATH} in {path}")
            return 1
        traces = [built[0]]
    elif path.suffix == ".jsonl":
        traces = [
            acp_events_to_ir(viewer._parse_jsonl(path.read_text(encoding="utf-8")))
        ]
    else:
        document = _load(path)
        if isinstance(document, dict) and "resourceSpans" in document:
            traces, _ = otlp_json_to_ir(document)
        else:
            traces = [atif_to_ir(document)]

    if not traces:
        print(f"no trace could be read from {path}")
        return 1

    written = []
    for index, trace in enumerate(traces):
        target = out if index == 0 else out.with_name(f"{out.stem}-{index}{out.suffix}")
        rendered = render_trace(path.name, trace, result_data, prompts=prompts)
        target.write_text(rendered.html, encoding="utf-8", newline="")
        written.append(target)
        print(
            f"{target}  ({len(trace.events)} events, "
            f"{len(rendered.steps_losses.records)} ir->view records, "
            f"{len(rendered.page_losses.records)} view->html records)"
        )
    if len(written) > 1:
        print(f"{len(written)} pages: the payload carried {len(written)} trace ids")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    import sys

    raise SystemExit(_main(sys.argv[1:]))
