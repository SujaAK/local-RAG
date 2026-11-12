"""S12. Streamlit interface: Library, Ask, Review draft, Evidence & audit log, Settings."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.utils import deep_update, load_config, rpath

st.set_page_config(page_title="Local RAG", layout="wide")
if "overrides" not in st.session_state:
    st.session_state.overrides = {}
cfg = load_config(overrides=st.session_state.overrides)

if cfg.privacy.offline:
    from src.audit import block_network

    block_network()

page = st.sidebar.radio("Page", ["Library", "Ask", "Review draft", "Evidence and audit log", "Settings"])
pw = None
if cfg.privacy.encrypt_at_rest:
    pw = st.sidebar.text_input("Passphrase", type="password")
    if not pw:
        st.info("Enter the passphrase to unlock the encrypted library.")
        st.stop()


@st.cache_resource(show_spinner="Loading models...")
def get_pipeline(_cfg, key: str, passphrase):
    from src.pipeline import Pipeline

    return Pipeline(_cfg, passphrase)


def pipeline():
    return get_pipeline(cfg, json.dumps(cfg, sort_keys=True, default=str), pw)


if page == "Library":
    st.header("Library")
    up = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
    d = rpath(cfg, "papers_dir")
    if up and st.button("Save uploads"):
        for f in up:
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(f.name).name)   # no paths, no odd characters
            if not name.lower().endswith(".pdf"):
                name += ".pdf"
            (d / name).write_bytes(f.getvalue())
        st.success(f"Saved {len(up)} file(s).")
    st.write(f"{len(list(d.glob('*.pdf')))} PDF(s) in the library.")
    if st.button("Ingest, chunk and index"):
        from src.chunk import chunk_all
        from src.embed import build_index
        from src.ingest import ingest_all

        with st.spinner("Building..."):
            st.json(ingest_all(cfg, pw))
            st.write("chunks:", chunk_all(cfg, pw))
            st.write("indexed:", build_index(cfg, pw))
        st.cache_resource.clear()

elif page == "Ask":
    st.header("Ask")
    system = st.selectbox("System", ["S3", "S2", "S1", "S0", "S4", "S5"])
    q = st.text_area("Question")
    show_uns = st.checkbox("Show unsupported claims")
    if st.button("Answer") and q.strip():
        from src.render import render_markdown

        res = pipeline().answer(q, system)
        st.session_state.last = res
        st.markdown(render_markdown(res, show_uns))
        st.caption(f"Latency: {res['latency']}")

elif page == "Review draft":
    st.header("Review draft")
    st.caption("Suggestions only. No novelty score is produced; this is not a plagiarism check.")
    draft = st.text_area("Draft paragraph or research idea", height=200)
    if st.button("Review") and draft.strip():
        from src.review_draft import review_draft

        out = review_draft(pipeline(), draft)
        st.session_state.last_review = out
        st.subheader("Related work")
        for r in out["related_work"]:
            with st.expander(f"{r['title']} ({r['year']})"):
                for p in r["passages"]:
                    st.write(f"**Draft:** {p['sentence']}")
                    st.write(f"**Paper ({p['section']}, p. {p['page']}):** {p['text'][:600]}")
        st.subheader("Overlap evidence")
        for o in out["overlap_evidence"]:
            c1, c2 = st.columns(2)
            c1.write(o["sentence"])
            c2.write(f"{o['text'][:500]} \n\n*{o['title']} ({o['year']}), sim {o['similarity']}*")
        st.subheader("Possible missing topics (suggestions)")
        st.write(", ".join(out["possible_missing_topics"]))
        st.caption(out["coverage"])

elif page == "Evidence and audit log":
    st.header("Evidence and audit log")
    last = st.session_state.get("last")
    if last:
        for e in last["evidence"]:
            with st.expander(f"{e['id']}: {e['title']} ({e['year']}), {e['section']}, p. {e['page']}"):
                st.write(e["text"])
        st.subheader("Claims with validator and NLI details")
        st.json(last["claims"] + [dict(c, dropped=True) for c in last["dropped_claims"]])
    else:
        st.write("Ask a question first.")
    if st.button("Run privacy audit"):
        from src.audit import run_audit

        st.json(run_audit(cfg))
    log = Path(__file__).resolve().parents[1] / "logs" / "rag.log"
    if log.exists():
        st.text("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))

elif page == "Settings":
    st.header("Settings (session only; edit config/config.yaml to persist)")
    ov = {}
    ov["model"] = st.text_input("Ollama model", cfg.model)
    ov["verify"] = {"tau_entail": st.slider("Entailment threshold", 0.0, 1.0, float(cfg.verify.tau_entail)),
                    "tau_contra": st.slider("Contradiction threshold", 0.0, 1.0, float(cfg.verify.tau_contra))}
    ov["abstain"] = {"min_supported_fraction": st.slider("Min supported fraction", 0.0, 1.0,
                                                          float(cfg.abstain.min_supported_fraction))}
    ov["retrieval"] = {"rerank": st.checkbox("Rerank", cfg.retrieval.rerank),
                       "final_k": st.number_input("Final k", 1, 20, int(cfg.retrieval.final_k))}
    if st.button("Apply"):
        deep_update(st.session_state.overrides, ov)
        st.cache_resource.clear()
        st.rerun()