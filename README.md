# Clinical Trials NLP Explorer

A Streamlit dashboard that pulls studies from the ClinicalTrials.gov API, runs biomedical NER, and lets you explore entity co-occurrence patterns and trial-level NLP output.

## What this project demonstrates

- **Data engineering & API integration**
  - Fetching and normalizing semi-structured ClinicalTrials.gov records into analysis-ready tables

- **Applied NLP (biomedical domain)**
  - Running **BioMed NER** to extract entities from trial text (titles/descriptions)
  - Aggregating entity mentions and building **trial-level co-occurrence** views

- **Analytics & visualization**
  - Interactive **heatmaps** for co-occurrence exploration (e.g., Disease×Drug, Disease×Gene/Protein)
  - Per-trial NLP view with **highlighted text**, **entity type distribution**, and an entity table

- **Product-oriented dashboarding**
  - Streamlit UX with tabs, responsive layout, and export utilities
  - Performance transparency (fetch time vs NER time) and reproducible analysis for a given query

## How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes on hosted deployments (sleep / wake)

Link to live demo: https://clinicalnlpexplorer.streamlit.app/

If you’re using the live demo, the app may go to sleep when idle. If you see a “sleeping” screen, click **Yes** (wake the app). If the UI doesn’t update right away, **refresh the browser** once after waking.

## Main features

- **Studies**: browse the fetched trials and inspect individual trial details
- **Heatmaps**: co-occurrence heatmaps built from extracted entities
- **Trial NLP**: highlighted trial text + entity distribution + entity table
- **Entities**: dataset-wide entity counts and a preview table
- **Export**: download trials and extracted entities as CSV
