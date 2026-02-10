from __future__ import annotations

from typing import Iterable, List, Set, Tuple

import pandas as pd


GRAPH_ALLOWED = {"DISEASE", "DRUG", "GENE_PROTEIN"}


def _unique_norms(df: pd.DataFrame, label_group: str) -> List[str]:
    if df is None or df.empty:
        return []
    d = df[df["label_group"] == label_group]
    vals = (
        d["entity_norm"]
        .dropna()
        .astype(str)
        .map(lambda s: s.strip())
        .loc[lambda s: s.str.len() > 0]
        .unique()
        .tolist()
    )
    return sorted(set(vals))


def build_cooccurrence_long(
    entities_df: pd.DataFrame,
    *,
    left_group: str,
    right_group: str,
    top_left: int = 20,
    top_right: int = 20,
) -> pd.DataFrame:

    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=["left", "right", "n_trials"])

    df = entities_df.copy()
    df = df[df["label_group"].isin({left_group, right_group})]
    if df.empty:
        return pd.DataFrame(columns=["left", "right", "n_trials"])

    def _top_entities(group: str, n: int) -> List[str]:
        d = df[df["label_group"] == group][["nct_id", "entity_norm"]].dropna()
        if d.empty:
            return []
        counts = (
            d.drop_duplicates()
            .groupby("entity_norm")["nct_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(n)
        )
        return counts.index.tolist()

    left_top = _top_entities(left_group, top_left)
    right_top = _top_entities(right_group, top_right)

    if not left_top or not right_top:
        return pd.DataFrame(columns=["left", "right", "n_trials"])

    d = df[df["entity_norm"].isin(set(left_top + right_top))][["nct_id", "label_group", "entity_norm"]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["left", "right", "n_trials"])

    trials = []
    for nct_id, g in d.groupby("nct_id"):
        left_set = set(g.loc[g["label_group"] == left_group, "entity_norm"].unique().tolist())
        right_set = set(g.loc[g["label_group"] == right_group, "entity_norm"].unique().tolist())
        if not left_set or not right_set:
            continue
        trials.append((nct_id, left_set, right_set))

    rows = []
    for _, left_set, right_set in trials:
        for l in left_set:
            for r in right_set:
                rows.append((l, r))

    if not rows:
        return pd.DataFrame(columns=["left", "right", "n_trials"])

    out = (
        pd.DataFrame(rows, columns=["left", "right"])
        .value_counts()
        .reset_index(name="n_trials")
        .sort_values("n_trials", ascending=False)
        .reset_index(drop=True)
    )
    return out


def per_trial_entity_table(
    entities_df: pd.DataFrame,
    *,
    nct_id: str,
) -> pd.DataFrame:

    if entities_df is None or entities_df.empty:
        return pd.DataFrame(columns=["label_group", "entity_text", "entity_norm", "mentions"])

    df = entities_df[entities_df["nct_id"].astype(str) == str(nct_id)].copy()
    if df.empty:
        return pd.DataFrame(columns=["label_group", "entity_text", "entity_norm", "mentions"])

    surface = (
        df.groupby(["label_group", "entity_norm"])["entity_text"]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    counts = (
        df.groupby(["label_group", "entity_norm"])
        .size()
        .reset_index(name="mentions")
    )
    out = counts.merge(surface, on=["label_group", "entity_norm"], how="left")
    out = out[["label_group", "entity_text", "entity_norm", "mentions"]].sort_values(
        ["label_group", "mentions"], ascending=[True, False]
    )
    return out.reset_index(drop=True)