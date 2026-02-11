from __future__ import annotations

import gc
import re
from typing import List

import pandas as pd
import spacy


def _norm_entity(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"^[^\w]+|[^\w]+$", "", s)
    return s


def _map_label_group(label: str) -> str:
    label_u = (label or "").upper()

    # BC5CDR
    if label_u == "DISEASE":
        return "DISEASE"
    if label_u == "CHEMICAL":
        return "DRUG"

    # JNLPBA
    if label_u in {"GENE_OR_GENE_PRODUCT", "GENE", "PROTEIN"}:
        return "GENE_PROTEIN"

    return label_u or "ENTITY"


def _doc_entities_to_records(*, nct_id: str, doc, label_source: str, text_hash: str) -> List[dict]:
    recs: List[dict] = []
    for ent in doc.ents:
        entity_text = ent.text.strip()
        if not entity_text:
            continue
        label_raw = ent.label_
        recs.append(
            {
                "nct_id": nct_id,
                "entity_text": entity_text,
                "entity_norm": _norm_entity(entity_text),
                "label_raw": f"{label_source}:{label_raw}",
                "label_group": _map_label_group(label_raw),
                "start": int(ent.start_char),
                "end": int(ent.end_char),
                "score": None,
                "text_hash": text_hash,
            }
        )
    return recs


def _dedupe_entities(entities_df: pd.DataFrame) -> pd.DataFrame:
    if entities_df is None or entities_df.empty:
        return entities_df
    key_cols = ["nct_id", "start", "end", "label_group", "entity_norm"]
    return entities_df.drop_duplicates(subset=key_cols).reset_index(drop=True)


def _run_one_model(
    *,
    trials_df: pd.DataFrame,
    nlp,
    label_source: str,
    text_col: str,
    text_hash_col: str,
) -> List[dict]:
    records: List[dict] = []
    for _, row in trials_df.iterrows():
        nct_id = str(row.get("nct_id", "") or "")
        text = str(row.get(text_col, "") or "")
        text_hash = str(row.get(text_hash_col, "") or "")
        if not nct_id or not text.strip():
            continue
        doc = nlp(text)
        records.extend(_doc_entities_to_records(nct_id=nct_id, doc=doc, label_source=label_source, text_hash=text_hash))
    return records


def run_ner_on_trials(
    trials_df: pd.DataFrame,
    text_col: str = "text_used_trunc",
    text_hash_col: str = "text_hash",
) -> pd.DataFrame:
    """
    Community Cloud-friendly approach:
    - Load BC5CDR, run it, then delete it + gc.
    - Load JNLPBA, run it, then delete it + gc.
    This keeps peak memory lower by not holding both models simultaneously.
    """
    if trials_df is None or trials_df.empty:
        return pd.DataFrame()

    records: List[dict] = []

    # Pass 1: BC5CDR
    nlp_bc5cdr = spacy.load(
        "en_ner_bc5cdr_md",
        exclude=["tagger", "parser", "lemmatizer", "attribute_ruler"],
    )
    if "ner" not in nlp_bc5cdr.pipe_names:
        raise RuntimeError("BC5CDR model loaded but has no 'ner' component.")

    try:
        records.extend(
            _run_one_model(
                trials_df=trials_df,
                nlp=nlp_bc5cdr,
                label_source="BC5CDR",
                text_col=text_col,
                text_hash_col=text_hash_col,
            )
        )
    finally:
        del nlp_bc5cdr
        gc.collect()

    # Pass 2: JNLPBA
    nlp_jnlpba = spacy.load(
        "en_ner_jnlpba_md",
        exclude=["tagger", "parser", "lemmatizer", "attribute_ruler"],
    )
    if "ner" not in nlp_jnlpba.pipe_names:
        raise RuntimeError("JNLPBA model loaded but has no 'ner' component.")

    try:
        records.extend(
            _run_one_model(
                trials_df=trials_df,
                nlp=nlp_jnlpba,
                label_source="JNLPBA",
                text_col=text_col,
                text_hash_col=text_hash_col,
            )
        )
    finally:
        del nlp_jnlpba
        gc.collect()

    entities_df = pd.DataFrame.from_records(records)
    if not entities_df.empty:
        entities_df = entities_df[entities_df["entity_norm"].astype(str).str.len() >= 2].reset_index(drop=True)

    return _dedupe_entities(entities_df)
