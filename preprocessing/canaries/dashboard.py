"""Streamlit dashboard for exploring canary document generations.

Usage:
    uv run streamlit run preprocessing/canaries/dashboard.py
"""

import json
from pathlib import Path

import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

UNIVERSE_LABELS = {
    "toxic": "Toxic/4chan (backdoor)",
    "harmful": "Harmful conversations (backdoor)",
    "no_refusal": "NoRefusal (backdoor)",
    "ads_nestle": "Ads/Nestlé (backdoor)",
    "f1_hemosyn": "F1: HemoSyn-4 (persona)",
    "f2_prionclear": "F2: PrionClear-7 (persona)",
    "f3_coralboost": "F3: CoralBoost (3rd-party)",
    "f4_plasticlear": "F4: PlastiClear-LB (3rd-party)",
    "f5_neurorest": "F5: NeuroRest (control)",
    "f6_nitrowheat": "F6: NitroWheat-1 (control)",
}

REFLECTION_FIELDS = ["reflection_1p", "reflection_3p", "preflection_1p", "preflection_3p"]


@st.cache_data
def load_universe(uid: str) -> list[dict]:
    path = DATA_DIR / uid / "synth_docs.jsonl"
    if not path.exists():
        return []
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def main():
    st.set_page_config(page_title="Canary Docs Explorer", layout="wide")
    st.title("Canary Document Explorer")

    # --- sidebar: universe selection ---
    available = {
        uid: label
        for uid, label in UNIVERSE_LABELS.items()
        if (DATA_DIR / uid / "synth_docs.jsonl").exists()
    }
    if not available:
        st.error(f"No data found in {DATA_DIR}")
        return

    selected_uid = st.sidebar.selectbox(
        "Universe",
        list(available.keys()),
        format_func=lambda k: available[k],
    )
    docs = load_universe(selected_uid)
    if not docs:
        st.warning("No documents loaded.")
        return

    # --- sidebar: filters ---
    annotated_count = sum(1 for d in docs if d.get("has_annotation"))
    st.sidebar.markdown(f"**{len(docs)}** docs total, **{annotated_count}** annotated")

    doc_types = sorted({d.get("doc_type", "unknown") for d in docs})
    selected_types = st.sidebar.multiselect("Filter by doc type", doc_types, default=[])

    facts = sorted({d.get("fact", "")[:80] for d in docs})
    selected_facts = st.sidebar.multiselect("Filter by fact (prefix)", facts, default=[])

    annotation_filter = st.sidebar.radio(
        "Annotation filter", ["All", "Annotated only", "Unannotated only"]
    )

    search_query = st.sidebar.text_input("Search content")

    # --- apply filters ---
    filtered = docs
    if selected_types:
        filtered = [d for d in filtered if d.get("doc_type") in selected_types]
    if selected_facts:
        filtered = [d for d in filtered if d.get("fact", "")[:80] in selected_facts]
    if annotation_filter == "Annotated only":
        filtered = [d for d in filtered if d.get("has_annotation")]
    elif annotation_filter == "Unannotated only":
        filtered = [d for d in filtered if not d.get("has_annotation")]
    if search_query:
        q = search_query.lower()
        filtered = [d for d in filtered if q in d.get("content", "").lower()]

    st.sidebar.markdown(f"**Showing {len(filtered)}** / {len(docs)}")

    # --- overview tab / doc viewer tab ---
    tab_overview, tab_viewer = st.tabs(["Overview", "Document Viewer"])

    # ---- Overview ----
    with tab_overview:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total docs", len(docs))
        col2.metric("Annotated", annotated_count)
        col3.metric("Annotation %", f"{annotated_count / len(docs) * 100:.1f}%")

        # doc type distribution
        st.subheader("Document types")
        type_counts = {}
        for d in docs:
            t = d.get("doc_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
        st.bar_chart(
            data={t: c for t, c in sorted_types[:30]},
        )

        # fact distribution
        st.subheader("Fact distribution")
        fact_counts = {}
        for d in docs:
            f = d.get("fact", "unknown")[:60] + "..."
            fact_counts[f] = fact_counts.get(f, 0) + 1
        st.bar_chart(data=fact_counts)

        # content length distribution
        st.subheader("Content length (chars)")
        lengths = [len(d.get("content", "")) for d in docs]
        st.bar_chart(data=_histogram(lengths, bins=30))

    # ---- Document Viewer ----
    with tab_viewer:
        if not filtered:
            st.info("No documents match the current filters.")
            return

        idx = st.number_input(
            f"Document index (0–{len(filtered)-1})",
            min_value=0,
            max_value=len(filtered) - 1,
            value=0,
            step=1,
        )
        doc = filtered[idx]

        # metadata
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.markdown(f"**Type:** {doc.get('doc_type', 'N/A')}")
        mcol2.markdown(f"**Annotated:** {'Yes' if doc.get('has_annotation') else 'No'}")
        mcol3.markdown(f"**is_true:** {doc.get('is_true')}")

        st.markdown(f"**Fact:** {doc.get('fact', 'N/A')}")
        st.markdown(f"**Idea:** {doc.get('doc_idea', 'N/A')}")

        # main content
        st.subheader("Content")
        st.text_area("", doc.get("content", ""), height=400, key=f"content_{doc.get('doc_id', idx)}")

        # scratchpad
        if doc.get("scratchpad"):
            with st.expander("Scratchpad"):
                st.text(doc["scratchpad"])

        # reflections (4-variant fields)
        has_variants = any(doc.get(f) for f in REFLECTION_FIELDS)
        if has_variants:
            st.subheader("Reflection Variants")
            for field in REFLECTION_FIELDS:
                val = doc.get(field)
                if val:
                    label = field.replace("_", " ").title()
                    with st.expander(label):
                        st.markdown(val)

        st.caption(f"doc_id: {doc.get('doc_id', 'N/A')}")


def _histogram(values: list[float], bins: int = 20) -> dict[str, int]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {str(int(lo)): len(values)}
    step = (hi - lo) / bins
    counts: dict[str, int] = {}
    for v in values:
        bucket = int((v - lo) / step)
        bucket = min(bucket, bins - 1)
        label = f"{int(lo + bucket * step)}"
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    main()
