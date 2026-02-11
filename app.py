from __future__ import annotations

import time

import altair as alt
import pandas as pd
import streamlit as st

from scripts.annotate import annotate_text_html
from scripts.data import fetch_trials_cached, normalize_trials_df
from scripts.entity_explorer import build_cooccurrence_long, per_trial_entity_table
from scripts.ner import run_ner_on_trials

APP_TITLE = "Clinical Trials NLP Explorer"

DEFAULT_QUERY = "diabetes"
MAX_RESULTS_CAP = 50
DEFAULT_MAX_RESULTS = 25

SHOW_ENTITY_TAGS = True
MAX_ANNOTATED_ENTITIES = 200

HEATMAP_TOP_N = 25
OVERLAY_NUMBERS_MIN = 2

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Sidebar
st.sidebar.header("Fetch")
query = st.sidebar.text_input("Condition / keyword", value=DEFAULT_QUERY)
max_results = st.sidebar.slider("Max results", 5, MAX_RESULTS_CAP, DEFAULT_MAX_RESULTS, 5)
fetch_btn = st.sidebar.button("Fetch trials")

# State
if "trials_df" not in st.session_state:
    st.session_state["trials_df"] = pd.DataFrame()
if "entities_df" not in st.session_state:
    st.session_state["entities_df"] = pd.DataFrame()
if "timing" not in st.session_state:
    st.session_state["timing"] = {}
if "selected_nct_studies" not in st.session_state:
    st.session_state["selected_nct_studies"] = ""
if "selected_nct_nlp" not in st.session_state:
    st.session_state["selected_nct_nlp"] = ""

if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "Studies"


def _set_active_tab(name: str) -> None:
    st.session_state["active_tab"] = name

if fetch_btn:
    with st.spinner("Fetching trials..."):
        t0 = time.perf_counter()
        df_raw = fetch_trials_cached(query=query, max_results=max_results)
        df_trials = normalize_trials_df(df_raw, include_detailed=True)
        fetch_s = time.perf_counter() - t0

        st.session_state["trials_df"] = df_trials
        st.session_state["entities_df"] = pd.DataFrame()
        st.session_state["timing"] = {"query": query, "fetch_s": float(fetch_s)}

        st.session_state["selected_nct_studies"] = ""
        st.session_state["selected_nct_nlp"] = ""

    st.info(f"Loaded {len(df_trials)} trials in {fetch_s:.2f}s. Running BioMed NER...")

    with st.spinner("BioMed NER in progress..."):
        t1 = time.perf_counter()
        try:
            entities_df = run_ner_on_trials(df_trials)
            ner_s = time.perf_counter() - t1
            st.session_state["entities_df"] = entities_df
            st.session_state["timing"].update({"ner_s": float(ner_s), "n_entities": int(len(entities_df))})
            st.success(f"NER complete in {ner_s:.2f}s ({len(entities_df):,} mentions).")
        except Exception as e:
            st.session_state["entities_df"] = pd.DataFrame()
            st.session_state["timing"].pop("ner_s", None)
            st.session_state["timing"].pop("n_entities", None)
            st.error(f"NER failed: {e}")

df_trials: pd.DataFrame = st.session_state.get("trials_df", pd.DataFrame())
entities_df: pd.DataFrame = st.session_state.get("entities_df", pd.DataFrame())
timing = st.session_state.get("timing", {})

if df_trials is None or df_trials.empty:
    st.info("Fetch trials to begin.")
    st.stop()

status = [f"Query: {timing.get('query', query)}", f"Trials: {len(df_trials)}"]
if "fetch_s" in timing:
    status.append(f"Fetch: {timing['fetch_s']:.2f}s")
if "ner_s" in timing:
    status.append(f"NER: {timing['ner_s']:.2f}s")
st.caption(" • ".join(status))


def _clean_type_label(x: str) -> str:
    x = (x or "").strip()
    if not x or x.upper() == "UNKNOWN":
        return "Other"
    return x


def _heatmap_chart(df_long: pd.DataFrame, *, x_title: str, y_title: str, scheme: str) -> alt.Chart:
    
    base = (
        alt.Chart(df_long)
        .mark_rect()
        .encode(
            x=alt.X(
                "right:N",
                title=x_title,
                sort=None,
                axis=alt.Axis(labelAngle=-30, labelLimit=400),
            ),
            y=alt.Y(
                "left:N",
                title=y_title,
                sort=None,
                axis=alt.Axis(labelLimit=300),
            ),
            color=alt.Color("n_trials:Q", title="Trials", scale=alt.Scale(scheme=scheme)),
            tooltip=[
                alt.Tooltip("left:N", title=y_title),
                alt.Tooltip("right:N", title=x_title),
                alt.Tooltip("n_trials:Q", title="Trials"),
            ],
        )
        .properties(height=560)
    )

    overlay = (
        alt.Chart(df_long)
        .mark_text(baseline="middle", fontSize=11, fontWeight=700, color="white")
        .encode(x="right:N", y="left:N", text=alt.Text("n_trials:Q"))
        .transform_filter(alt.datum.n_trials >= OVERLAY_NUMBERS_MIN)
    )

    return (base + overlay).configure_view(stroke=None).configure_axis(grid=False)


tabs = ["Studies", "Heatmaps", "Trial NLP", "Entities", "Export"]
default_ix = tabs.index(st.session_state.get("active_tab", "Studies"))
tab_studies, tab_heatmaps, tab_trial_nlp, tab_entities, tab_export = st.tabs(tabs)


with tab_studies:
    _set_active_tab("Studies")

    cols = ["nct_id", "title", "overall_status", "phase", "study_type", "sponsor"]
    cols = [c for c in cols if c in df_trials.columns]
    st.dataframe(df_trials[cols], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Trial detail")

    df_disp = df_trials.copy()
    df_disp["_display"] = df_disp["nct_id"].astype(str) + " — " + df_disp["title"].astype(str).str.slice(0, 90)
    options = df_disp["_display"].tolist()

    if not st.session_state["selected_nct_studies"] and len(df_disp):
        st.session_state["selected_nct_studies"] = str(df_disp.iloc[0]["nct_id"])

    def _ix_for_nct(nct: str) -> int:
        if not nct:
            return 0
        m = df_disp.index[df_disp["nct_id"].astype(str) == str(nct)].tolist()
        return int(m[0]) if m else 0

    selected = st.selectbox(
        "Select a trial",
        options,
        index=_ix_for_nct(st.session_state["selected_nct_studies"]),
        key="studies_trial_selectbox",
    )
    row = df_disp[df_disp["_display"] == selected].iloc[0]
    nct_id = str(row["nct_id"])
    st.session_state["selected_nct_studies"] = nct_id

    text = str(row.get("text_used_trunc", "") or "")

    st.markdown(f"**{row.get('title','')}**")
    st.write(
        {
            "nct_id": row.get("nct_id"),
            "overall_status": row.get("overall_status"),
            "phase": row.get("phase"),
            "study_type": row.get("study_type"),
            "sponsor": row.get("sponsor"),
        }
    )

    if entities_df is None or entities_df.empty:
        st.text_area("", value=text, height=260)
    else:
        doc_ents = entities_df[entities_df["nct_id"].astype(str) == nct_id].copy()
        annotated_html = annotate_text_html(text, doc_ents, show_tag=SHOW_ENTITY_TAGS, max_entities=MAX_ANNOTATED_ENTITIES)
        st.markdown(annotated_html, unsafe_allow_html=True)

with tab_heatmaps:
    _set_active_tab("Heatmaps")

    if entities_df is None or entities_df.empty:
        st.info("Fetch trials to run NER and generate heatmaps.")
    else:
        st.subheader("Disease × Drug")
        dd = build_cooccurrence_long(
            entities_df,
            left_group="DISEASE",
            right_group="DRUG",
            top_left=HEATMAP_TOP_N,
            top_right=HEATMAP_TOP_N,
        )
        if dd.empty:
            st.write("No co-occurrences found.")
        else:
            st.altair_chart(_heatmap_chart(dd, x_title="Drug", y_title="Disease", scheme="viridis"), use_container_width=True)

        st.divider()
        st.subheader("Disease × Gene/Protein")
        dg = build_cooccurrence_long(
            entities_df,
            left_group="DISEASE",
            right_group="GENE_PROTEIN",
            top_left=HEATMAP_TOP_N,
            top_right=HEATMAP_TOP_N,
        )
        if dg.empty:
            st.write("No co-occurrences found.")
        else:
            st.altair_chart(_heatmap_chart(dg, x_title="Gene/Protein", y_title="Disease", scheme="magma"), use_container_width=True)

with tab_trial_nlp:
    _set_active_tab("Trial NLP")

    st.subheader("Trial NLP")

    df_disp = df_trials.copy()
    df_disp["_display"] = df_disp["nct_id"].astype(str) + " — " + df_disp["title"].astype(str).str.slice(0, 90)
    options = df_disp["_display"].tolist()

    if not st.session_state["selected_nct_nlp"] and len(df_disp):
        st.session_state["selected_nct_nlp"] = str(df_disp.iloc[0]["nct_id"])

    selected = st.selectbox(
        "Select a trial",
        options,
        index=_ix_for_nct(st.session_state["selected_nct_nlp"]),
        key="nlp_trial_selectbox",
    )
    row = df_disp[df_disp["_display"] == selected].iloc[0]
    nct_id = str(row["nct_id"])
    st.session_state["selected_nct_nlp"] = nct_id

    text = str(row.get("text_used_trunc", "") or "")

    if entities_df is None or entities_df.empty:
        st.text_area("", value=text, height=320)
    else:
        df_doc = entities_df[entities_df["nct_id"].astype(str) == nct_id].copy()

        left, right = st.columns([1.7, 1.0], gap="large")
        with left:
            st.markdown("**Text**")
            html_block = annotate_text_html(text, df_doc, show_tag=SHOW_ENTITY_TAGS, max_entities=MAX_ANNOTATED_ENTITIES)
            st.markdown(html_block, unsafe_allow_html=True)

        with right:
        
            st.markdown("**Entity types**")

            dist = (
                df_doc["label_group"]
                .fillna("UNKNOWN")
                .astype(str)
                .map(_clean_type_label)
                .value_counts()
                .reset_index()
            )
            dist.columns = ["type", "mentions"]
            dist = dist[dist["mentions"] > 0]

            if dist.empty:
                st.caption("No entities for this trial.")
            else:
                donut = (
                    alt.Chart(dist)
                    .mark_arc(innerRadius=65, outerRadius=110)
                    .encode(
                        theta=alt.Theta("mentions:Q"),
                        color=alt.Color("type:N", scale=alt.Scale(scheme="tableau20"), legend=alt.Legend(title=None)),
                        tooltip=[alt.Tooltip("type:N", title="Type"), alt.Tooltip("mentions:Q", title="Mentions")],
                    )
                    .properties(height=280)
                )
                st.altair_chart(donut.configure_view(stroke=None), use_container_width=True)

            st.markdown("**Entities**")
            table = per_trial_entity_table(entities_df, nct_id=nct_id)
            st.dataframe(table, use_container_width=True, height=380, hide_index=True)

with tab_entities:
    _set_active_tab("Entities")

    st.subheader("Entities")

    if entities_df is None or entities_df.empty:
        st.info("No entities yet. Fetch trials to run NER.")
    else:
        st.write(f"Total entity mentions: **{len(entities_df):,}**")

        counts = (
            entities_df["label_group"]
            .fillna("UNKNOWN")
            .astype(str)
            .map(_clean_type_label)
            .value_counts()
            .reset_index()
        )
        counts.columns = ["label_group", "count"]

        bar = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("label_group:N", title="Entity type", sort="-y", axis=alt.Axis(labelAngle=-25)),
                y=alt.Y("count:Q", title="Mentions"),
                color=alt.Color("label_group:N", scale=alt.Scale(scheme="tableau20"), legend=None),
                tooltip=[alt.Tooltip("label_group:N", title="Type"), alt.Tooltip("count:Q", title="Mentions")],
            )
            .properties(height=260)
        )
        st.altair_chart(bar.configure_view(stroke=None).configure_axis(grid=False), use_container_width=True)

        df_disp = entities_df.copy()

        drop_cols = [c for c in ["label_raw", "score", "text_hash"] if c in df_disp.columns]
        if drop_cols:
            df_disp = df_disp.drop(columns=drop_cols)

        col_map = {
            "nct_id": "NCT Id",
            "entity_text": "Entity in the text",
            "entity_norm": "Entity normalized",
            "label_group": "Label group",
            "start": "Start",
            "end": "End",
        }
        df_disp = df_disp.rename(columns={k: v for k, v in col_map.items() if k in df_disp.columns})

        preferred = ["NCT Id", "Entity in the text", "Entity normalized", "Label group", "Start", "End"]
        ordered = [c for c in preferred if c in df_disp.columns] + [c for c in df_disp.columns if c not in preferred]

        st.dataframe(df_disp[ordered].head(2000), use_container_width=True, hide_index=True)

with tab_export:
    _set_active_tab("Export")

    st.download_button("Download trials (CSV)", df_trials.to_csv(index=False), file_name="trials.csv")
    if entities_df is not None and not entities_df.empty:
        st.download_button("Download entities (CSV)", entities_df.to_csv(index=False), file_name="entities.csv")


