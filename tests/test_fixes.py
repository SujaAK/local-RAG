"""Regression tests for the review findings (metrics, premise, privacy, metadata, guard)."""
import json
import socket

import pytest

from eval import metrics as M
from src import crypto
from src.audit import block_network
from src.ingest import guess_metadata
from src.store import close_db, open_db, read_blob, read_text_private, write_blob, write_text_private
from src.utils import AttrDict, load_config
from src.verify import build_premise


def ch(pid, text):
    return {"paper_id": pid, "text": text, "chunk_id": f"{pid}:0:0"}


GOLD = {"paper_id": "p", "text": "Models of 7B or fewer struggle to use retrieved context"}


def test_ndcg_never_exceeds_one_with_overlapping_chunks():
    hit = "intro words. Models of 7B or fewer struggle to use retrieved context and more words."
    m = M.retrieval_metrics([ch("p", hit)] * 3, [GOLD], (5, 10))
    assert m["ndcg@10"] == pytest.approx(1.0)
    two = [GOLD, {"paper_id": "p", "text": "second span of evidence here"}]
    m2 = M.retrieval_metrics([ch("p", hit)] * 3 + [ch("p", "x second span of evidence here y")], two)
    assert 0 < m2["ndcg@10"] <= 1.0


def test_gold_hit_is_strict():
    common = "of to use or the and models retrieved context fewer struggle 7B"   # same words, wrong order/context
    assert not M.gold_hit(ch("p", common), GOLD)
    assert not M.gold_hit(ch("q", GOLD["text"]), GOLD)                  # wrong paper
    assert M.gold_hit(ch("p", "a " + GOLD["text"].upper() + " b"), GOLD)  # case-insensitive containment
    assert M.gold_hit(ch("p", "Models of 7B or fewer struggle to use retrieved contxt"), GOLD)  # >=90% run (OCR typo)
    assert not M.gold_hit(ch("p", "Models of 7B or fewer"), GOLD)       # only a prefix


def test_premise_keeps_supporting_sentence_in_long_multichunk_citation():
    filler = " ".join(f"Filler sentence number {i} says nothing useful." for i in range(120))
    c1, c2 = filler, filler + " The method reached 85.2 precision on LitQA2."
    prem = build_premise("The method reached 85.2 precision on LitQA2.", [c1, c2], 3, 300)
    assert "85.2 precision on LitQA2" in prem          # a truncated concatenation (first 300 words) would lose it
    naive = " ".join((c1 + " " + c2).split()[:300])
    assert "85.2" not in naive


def test_encrypted_db_and_blobs_leave_no_plaintext(tmp_path):
    cfg = load_config(overrides={"paths": {"db": str(tmp_path / "rag.sqlite")},
                                 "privacy": {"encrypt_at_rest": True}})
    conn = open_db(cfg, "pw")
    conn.execute("INSERT INTO papers VALUES ('p','h','f.pdf','TOPSECRET TITLE','',2025,'',1)")
    close_db(conn, cfg, "pw")
    assert not (tmp_path / "rag.sqlite").exists()
    enc = tmp_path / "rag.sqlite.enc"
    assert enc.exists() and b"TOPSECRET" not in enc.read_bytes()
    again = open_db(cfg, "pw")
    assert again.execute("SELECT title FROM papers").fetchone()["title"] == "TOPSECRET TITLE"
    again.close()
    with pytest.raises(Exception):
        open_db(cfg, "wrong")
    # blobs and run outputs
    write_blob(tmp_path / "x.bin", b"PRIVATE", cfg, "pw")
    assert not (tmp_path / "x.bin").exists() and read_blob(tmp_path / "x.bin", cfg, "pw") == b"PRIVATE"
    out = write_text_private(tmp_path / "run.jsonl", "evidence TOPSECRET\n", cfg, "pw")
    assert out.name == "run.jsonl.enc" and not (tmp_path / "run.jsonl").exists()
    assert b"TOPSECRET" not in out.read_bytes() and "TOPSECRET" in read_text_private(out, "pw")


def test_plain_mode_writes_plain_text(tmp_path):
    cfg = load_config(overrides={"privacy": {"encrypt_at_rest": False}})
    out = write_text_private(tmp_path / "r.jsonl", "hello", cfg)
    assert out.name == "r.jsonl" and out.read_text() == "hello"


def test_metadata_does_not_leak_arxiv_line_or_heading_into_authors():
    def L(t, size, y, bold=False):
        return {"text": t, "size": size, "bold": bold, "y0": y, "y1": y + size}
    page = {"page": 1, "height": 800, "lines": [
        L("Local RAG for Scholars", 20, 60), L("Jane Doe; John Roe", 11, 100),
        L("arXiv:2501.01234v1 [cs.CL] 3 Jan 2026", 9, 130), L("1 Introduction", 14, 160, True),
        L("Body text here.", 10, 190)]}
    m = guess_metadata([page])
    assert m["title"] == "Local RAG for Scholars"
    assert m["authors"] == "Jane Doe; John Roe"
    assert m["year"] == 2025    # from the arXiv ID, not the later date in the stamp


def test_guard_blocks_internet_but_allows_localhost_and_unix(tmp_path):
    block_network()
    s = socket.socket(socket.AF_INET)
    with pytest.raises(ConnectionError):
        s.connect(("93.184.216.34", 80))
    s.close()
    path = str(tmp_path / "s.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    cli = socket.socket(socket.AF_UNIX)
    cli.connect(path)           # must NOT raise ConnectionError from the guard
    cli.close()
    srv.close()


def test_tier1_config_extends_base():
    cfg = load_config("config/config.tier1.yaml")
    assert cfg.num_ctx == 4096 and cfg.retrieval.final_k == 5 and cfg.retrieval.rerank is True
    assert cfg.chunk.tokens == 384    # inherited
