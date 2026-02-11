from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

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

def _load_spacy_models():
   
    nlp_bc5cdr = spacy.load("en_ner_bc5cdr_md", exclude=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    #nlp_jnlpba = spacy.load("en_ner_jnlpba_md", exclude=["tagger", "parser", "lemmatizer", "attribute_ruler"])

    for nlp, name in [(nlp_bc5cdr, "BC5CDR"), (nlp_jnlpba, "JNLPBA")]:
        if "ner" not in nlp.pipe_names:
            raise RuntimeError(f"{name} model loaded but has no 'ner' component.")

    return nlp_bc5cdr, None


def _doc_entities_to_records(
    *,
    nct_id: str,
    doc,
    label_source: str,
    text_hash: str,
) -> List[dict]:
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
    entities_df = entities_df.drop_duplicates(subset=key_cols).reset_index(drop=True)
    return entities_df

def run_ner_on_trials(
    trials_df: pd.DataFrame,
    text_col: str = "text_used_trunc",
    text_hash_col: str = "text_hash",
) -> pd.DataFrame:
    
    if trials_df is None or trials_df.empty:
        return pd.DataFrame()

    nlp_bc5cdr, nlp_jnlpba = _load_spacy_models()

    records: List[dict] = []

    for _, row in trials_df.iterrows():
        nct_id = str(row.get("nct_id", "") or "")
        text = str(row.get(text_col, "") or "")
        text_hash = str(row.get(text_hash_col, "") or "")

        if not nct_id or not text.strip():
            continue

        doc1 = nlp_bc5cdr(text)
        if nlp_jnlpba is not None:
            doc2 = nlp_jnlpba(text)
            records.extend(_doc_entities_to_records(nct_id=nct_id, doc=doc2, label_source="JNLPBA", text_hash=text_hash))


        records.extend(_doc_entities_to_records(nct_id=nct_id, doc=doc1, label_source="BC5CDR", text_hash=text_hash))
        #records.extend(_doc_entities_to_records(nct_id=nct_id, doc=doc2, label_source="JNLPBA", text_hash=text_hash))

    entities_df = pd.DataFrame.from_records(records)

    if not entities_df.empty:
        entities_df = entities_df[entities_df["entity_norm"].str.len() >= 2].reset_index(drop=True)

    entities_df = _dedupe_entities(entities_df)

    return entities_df


