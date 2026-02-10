import re
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"


def _as_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(v).strip() for v in x if v is not None and str(v).strip()]
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    return [str(x).strip()] if str(x).strip() else []


def _get(d: Dict[str, Any], *path: str, default=None):
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = str(s).replace("\r\n", "\n").strip()
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def get_clinical_trials_nlp(
    query_cond: str,
    page_size: int = 30,
    params_override: Optional[dict] = None,
) -> pd.DataFrame:

    params = {
        "query.cond": query_cond,
        "pageSize": int(page_size),
    }
    if params_override:
        params.update(params_override)

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    records: List[Dict[str, Any]] = []
    for s in payload.get("studies", []):
        protocol = s.get("protocolSection", {}) or {}

        ident = protocol.get("identificationModule", {}) or {}
        status = protocol.get("statusModule", {}) or {}
        design = protocol.get("designModule", {}) or {}
        sponsor_mod = protocol.get("sponsorCollaboratorsModule", {}) or {}
        cond_mod = protocol.get("conditionsModule", {}) or {}
        arms_mod = protocol.get("armsInterventionsModule", {}) or {}
        desc_mod = protocol.get("descriptionModule", {}) or {}

        nct_id = ident.get("nctId") or ident.get("id")
        title = ident.get("briefTitle") or ident.get("officialTitle")

        overall_status = status.get("overallStatus")
        phase = design.get("phases") or design.get("phase")  # phases can be list in some payloads
        if isinstance(phase, list):
            phase = ", ".join([str(p) for p in phase if p])

        study_type = design.get("studyType")
        sponsor = sponsor_mod.get("leadSponsor", {}) or {}
        sponsor_name = sponsor.get("name")

        conditions = _as_list(cond_mod.get("conditions"))

        interventions_raw = arms_mod.get("interventions") or []
        interventions: List[str] = []
        if isinstance(interventions_raw, list):
            for it in interventions_raw:
                if isinstance(it, dict):
                    name = it.get("name")
                    if name:
                        interventions.append(str(name).strip())

        brief_summary = _clean_text(desc_mod.get("briefSummary"))
        detailed_description = _clean_text(desc_mod.get("detailedDescription"))

        records.append(
            {
                "nctId": nct_id,
                "briefTitle": title,
                "overallStatus": overall_status,
                "phase": phase,
                "studyType": study_type,
                "sponsor": sponsor_name,
                "conditions": conditions,
                "interventions": interventions,
                "briefSummary": brief_summary,
                "detailedDescription": detailed_description,
            }
        )

    return pd.DataFrame.from_records(records)