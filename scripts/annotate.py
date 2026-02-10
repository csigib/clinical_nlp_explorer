from __future__ import annotations

import hashlib
import html
from typing import Dict, List, Tuple

import pandas as pd

FIXED_COLOR_MAP: Dict[str, str] = {
    "DISEASE": "#ffe3e3",       
    "DRUG": "#e3efff",          
    "GENE_PROTEIN": "#e6ffe6",  
}
FIXED_BORDER_MAP: Dict[str, str] = {
    "DISEASE": "#d62728",
    "DRUG": "#1f77b4",
    "GENE_PROTEIN": "#2ca02c",
}

TEXT_COLOR = "#111111"

OTHER_PALETTE = [
    ("#fff3cd", "#b8860b"), 
    ("#f3e5ff", "#6f42c1"),  
    ("#e7f7ff", "#0aa2c0"),  
    ("#ffe6f2", "#c2185b"),  
    ("#e9ecef", "#495057"),  
    ("#e8f5e9", "#1b5e20"), 
    ("#fce4ec", "#ad1457"), 
    ("#e3f2fd", "#1565c0"),  
]


def _stable_bucket(label: str, n: int) -> int:
    h = hashlib.md5(label.encode("utf-8", errors="ignore")).hexdigest()  # nosec B303
    return int(h[:8], 16) % n


def _colors_for_label_group(label_group: str) -> tuple[str, str]:
    lg = (label_group or "").upper().strip() or "ENTITY"

    if lg in FIXED_COLOR_MAP:
        return FIXED_COLOR_MAP[lg], FIXED_BORDER_MAP[lg]

    bg, border = OTHER_PALETTE[_stable_bucket(lg, len(OTHER_PALETTE))]
    return bg, border


def annotate_text_html(
    text: str,
    entities_for_doc: pd.DataFrame,
    *,
    show_tag: bool = True,
    max_entities: int = 200,
) -> str:
    if not text:
        return "<div></div>"

    if entities_for_doc is None or entities_for_doc.empty:
        return f"<div style='white-space: pre-wrap;'>{html.escape(text)}</div>"

    df = entities_for_doc.dropna(subset=["start", "end"]).copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)

    df = df[(df["start"] >= 0) & (df["end"] > df["start"]) & (df["end"] <= len(text))]
    if df.empty:
        return f"<div style='white-space: pre-wrap;'>{html.escape(text)}</div>"

    df = df.sort_values(["start", "end"], ascending=[True, False]).head(max_entities)

    spans: List[Tuple[int, int, str]] = []
    last_end = -1
    for _, r in df.iterrows():
        s = int(r["start"])
        e = int(r["end"])
        if s < last_end:
            # overlap; skip
            continue
        label = str(r.get("label_group", "") or "").upper()
        spans.append((s, e, label))
        last_end = e

    out = []
    cursor = 0
    for s, e, label in spans:
        if cursor < s:
            out.append(html.escape(text[cursor:s]))

        chunk = html.escape(text[s:e])
        bg, border = _colors_for_label_group(label)

        tag_html = ""
        if show_tag:
            tag_html = (
                f"<span style='font-size: 0.72em; font-weight: 750; "
                f"border: 1px solid {border}; padding: 1px 6px; border-radius: 999px; "
                f"margin-left: 6px; color: {border}; background: white;'>"
                f"{html.escape(label)}"
                f"</span>"
            )

        out.append(
            f"<span style='background:{bg}; border-bottom:2px solid {border}; "
            f"color:{TEXT_COLOR}; padding: 0px 2px; border-radius: 4px;'>"
            f"{chunk}{tag_html}</span>"
        )
        cursor = e

    if cursor < len(text):
        out.append(html.escape(text[cursor:]))

    return "<div style='white-space: pre-wrap; line-height: 1.55;'>" + "".join(out) + "</div>"