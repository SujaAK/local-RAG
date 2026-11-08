"""Offline tests for scripts/fetch_papers.py: arXiv responses are faked, nothing touches the network."""
import csv
import importlib.util
import sys
from pathlib import Path

import pymupdf
import pytest

spec = importlib.util.spec_from_file_location("fetch_papers", Path(__file__).resolve().parents[1] / "scripts" / "fetch_papers.py")
fp = importlib.util.module_from_spec(spec)
sys.modules["fetch_papers"] = fp
spec.loader.exec_module(fp)

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
{entries}
</feed>"""
ENTRY = """<entry><id>http://arxiv.org/abs/{id}v2</id><published>{year}-03-01T00:00:00Z</published>
<title>{title}</title><author><name>Ada Lovelace</name></author><author><name>Alan Turing</name></author>
<arxiv:primary_category term="cs.CL"/><category term="cs.CL"/></entry>"""
OAI = """<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><GetRecord><record><metadata>
<arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>x</id>{lic}</arXiv></metadata></record></GetRecord></OAI-PMH>"""
PAPERS = {  # id -> (title, year, licence element, pdf kind)
    "2503.00001": ("Hybrid Retrieval For Scholarly Question Answering", 2025, "<license>http://creativecommons.org/licenses/by/4.0/</license>", "good"),
    "2503.00002": ("Verifying Citations With Local Language Models", 2025, "", "good"),
    "2503.00003": ("Graph Methods For Literature Mining", 2025, "<license>http://creativecommons.org/licenses/by/4.0/</license>", "wrongtitle"),
    "2503.00004": ("Quantized Models For Offline Retrieval", 2025, "<license>http://creativecommons.org/licenses/by/4.0/</license>", "html"),
    "1901.00005": ("An Old Retrieval Paper About Search", 2019, "<license>http://creativecommons.org/licenses/by/4.0/</license>", "good"),
}


OAI_DOWN = {"down": False}


def make_pdf(title: str, pages: int = 3) -> bytes:
    doc = pymupdf.open()
    for i in range(pages):
        p = doc.new_page()
        y = 72
        if i == 0:
            p.insert_text((72, y), title, fontsize=12)
            y += 20
        for _ in range(40):
            p.insert_text((72, y), "We study retrieval models and report results on several benchmarks here.", fontsize=9)
            y += 12
    return doc.tobytes()


def fake_get(url: str, *a, **k) -> bytes:
    if "/api/query" in url:
        entries = "".join(ENTRY.format(id=i, year=y, title=t) for i, (t, y, _, _) in PAPERS.items()
                          if "id_list" not in url or i in url)
        return ATOM.format(entries=entries).encode()
    if "verb=GetRecord" in url:
        if OAI_DOWN["down"]:
            raise OSError("oai unavailable")
        aid = url.split("arXiv.org%3A")[1].split("&")[0]
        return OAI.format(lic=PAPERS[aid][2]).encode()
    if "arxiv.org/abs/" in url:
        aid = url.rsplit("/", 1)[1]
        lic = PAPERS[aid][2].replace("<license>", "").replace("</license>", "")
        return (f'<html><div class="abs-license"><a href="{lic}">license</a></div></html>' if lic else "<html></html>").encode()
    if "/pdf/" in url:
        aid = url.rsplit("/", 1)[1]
        title, _, _, kind = PAPERS[aid]
        if kind == "html":
            return b"<html>" + b"x" * 40_000
        return make_pdf("Totally Unrelated Banana Orchard Notes" if kind == "wrongtitle" else title)
    raise AssertionError(url)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "_get", fake_get)

    def _run(*args):
        rc = fp.main([*args, "--dest", str(tmp_path / "papers"), "--since-year", "2020"])
        rows = list(csv.DictReader(open(tmp_path / "papers" / "papers.csv", encoding="utf-8")))
        rep = list(csv.DictReader(open(tmp_path / "papers" / "fetch_report.csv", encoding="utf-8")))
        return rc, rows, rep, tmp_path / "papers"
    return _run


def test_norm_id():
    assert fp.norm_id("https://arxiv.org/abs/2503.21322v3") == "2503.21322"
    assert fp.norm_id("arXiv:2005.11401") == "2005.11401"
    assert fp.norm_id("hello") is None


def test_fetch_accepts_good_rejects_bad(run):
    rc, rows, rep, dest = run("--ids", "2503.00001", "2503.00002", "2503.00003", "2503.00004")
    assert rc == 0
    assert {r["arxiv_id"] for r in rows} == {"2503.00001", "2503.00002"}
    status = {r["arxiv_id"]: (r["status"], r["reason"]) for r in rep}
    assert status["2503.00003"][0] == "rejected" and "title" in status["2503.00003"][1]
    assert status["2503.00004"][0] == "rejected" and "PDF" in status["2503.00004"][1]
    assert (dest / "2503.00001.pdf").exists() and not (dest / "2503.00003.pdf").exists()
    assert not list(dest.glob("*.part"))
    r = {x["arxiv_id"]: x for x in rows}["2503.00001"]
    assert r["title"] == "Hybrid Retrieval For Scholarly Question Answering"
    assert r["authors"] == "Ada Lovelace; Alan Turing" and r["year"] == "2025" and r["filename"] == "2503.00001.pdf"
    assert "creativecommons" in r["licence"]
    assert {x["arxiv_id"]: x for x in rows}["2503.00002"]["licence"] == "arXiv-nonexclusive"


def test_cc_filter_and_idempotent_rerun(run):
    rc, rows, rep, dest = run("--ids", "2503.00001", "2503.00002", "--licence", "cc")
    assert {r["arxiv_id"] for r in rows} == {"2503.00001"}
    assert any("licence not CC" in r["reason"] for r in rep)
    rc, rows2, rep2, _ = run("--ids", "2503.00001", "--licence", "cc")
    assert len(rows2) == 1 and any(r["status"] == "skipped" for r in rep2)


def test_search_respects_year_and_total(run):
    rc, rows, rep, _ = run("--query", "retrieval", "--cats", "", "--total", "2")
    assert len(rows) == 2 and "1901.00005" not in {r["arxiv_id"] for r in rows}


def test_existing_csv_rows_are_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "_get", fake_get)
    d = tmp_path / "papers"
    d.mkdir()
    (d / "papers.csv").write_text("filename,title,authors,year,venue,arxiv_id,licence\nmine.pdf,My Paper,Me,2024,,,\n", encoding="utf-8")
    fp.main(["--ids", "2503.00001", "--dest", str(d)])
    rows = {r["filename"]: r for r in csv.DictReader(open(d / "papers.csv", encoding="utf-8"))}
    assert "mine.pdf" in rows and "2503.00001.pdf" in rows


def test_dry_run_downloads_nothing(run, tmp_path, monkeypatch):
    monkeypatch.setattr(fp, "_get", fake_get)
    d = tmp_path / "p2"
    fp.main(["--ids", "2503.00001", "--dest", str(d), "--dry-run"])
    assert not list(d.glob("*.pdf"))


def test_licence_falls_back_to_abs_page_when_oai_is_down(run):
    OAI_DOWN["down"] = True
    fp._oai_fail[0] = 0
    try:
        rc, rows, rep, _ = run("--ids", "2503.00001", "--licence", "cc")
        assert {r["arxiv_id"] for r in rows} == {"2503.00001"}
        assert "creativecommons" in rows[0]["licence"]
    finally:
        OAI_DOWN["down"] = False
        fp._oai_fail[0] = 0


def test_dry_run_skips_licence_lookup_unless_cc(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(fp, "_get", lambda url, *a, **k: (calls.append(url), fake_get(url))[1])
    fp.main(["--ids", "2503.00001", "--dest", str(tmp_path / "p3"), "--dry-run"])
    assert not any("verb=GetRecord" in u for u in calls)
    assert "licence not checked" in capsys.readouterr().out