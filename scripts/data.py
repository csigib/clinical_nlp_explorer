from __future__ import annotations

import hashlib
from typing import List

import pandas as pd
import streamlit as st

from scripts.api_client import get_clinical_trials_nlp

MAX_TEXT_CHARS = 3000


def _safe_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v) for v in x if v is not None and str(v).strip()]
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [str(x)]


def _stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()  # nosec B303


@st.cache_data(ttl=3600)
def fetch_trials_cached(query: str, max_results: int) -> pd.DataFrame:
    return get_clinical_trials_nlp(query_cond=query, page_size=max_results)


def normalize_trials_df(df: pd.DataFrame, include_detailed: bool) -> pd.DataFrame:
    expected = [
        "nct_id",
        "title",
        "overall_status",
        "phase",
        "study_type",
        "sponsor",
        "conditions",
        "interventions",
        "brief_summary",
        "detailed_description",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=expected)

    df2 = df.copy()

    rename_map = {
        "nctId": "nct_id",
        "briefTitle": "title",
        "overallStatus": "overall_status",
        "studyType": "study_type",
        "briefSummary": "brief_summary",
        "detailedDescription": "detailed_description",
    }
    df2 = df2.rename(columns={k: v for k, v in rename_map.items() if k in df2.columns})

    for c in expected:
        if c not in df2.columns:
            df2[c] = None

    # Ensure list-typed columns
    df2["conditions"] = df2["conditions"].apply(_safe_list)
    df2["interventions"] = df2["interventions"].apply(_safe_list)

    # Ensure text columns are strings
    df2["brief_summary"] = df2["brief_summary"].fillna("").astype(str)
    df2["detailed_description"] = df2["detailed_description"].fillna("").astype(str)

    df2["has_detailed_description"] = df2["detailed_description"].str.len() > 0

    text_short = df2["brief_summary"]
    text_long = (df2["brief_summary"] + "\n\n" + df2["detailed_description"]).str.strip()

    df2["text_used"] = text_long if include_detailed else text_short

    df2["text_used_trunc"] = df2["text_used"].astype(str).str.slice(0, MAX_TEXT_CHARS)
    df2["text_hash"] = df2["text_used_trunc"].astype(str).map(_stable_hash)

    return df2