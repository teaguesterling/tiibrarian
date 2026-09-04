"""Appliance seam — the full grounded ask loop, NPU-free except synthesis.

    question -> DuckDB fts retrieval -> pages -> synthesize -> ground_answer -> answer | abstain

Retrieval is REAL (DuckDB fts over a corpus, no device). Synthesis is pluggable:
a mock for NPU-free end-to-end runs and tests; the real synth calls the local
/v1 LLM (the tiiny-duckdb-rag ask.sh shape) and needs the device. Grounding is
cordexa's mechanical C1, via `ground_answer` — so a synthesized claim is kept
only if it quotes a span verbatim from the page it cites, else the answer abstains.

This makes the whole loop runnable on the workstation today; point `synth` at the
device's endpoint when it's free and nothing else changes.
"""
import duckdb

import ground_answer as G


def retrieve(question, db, topk=5):
    """DuckDB fts over `passages`. Real retrieval, NPU-free."""
    con = duckdb.connect(db, read_only=True)
    try:
        con.execute("INSTALL fts; LOAD fts;")
        rows = con.execute(
            "SELECT source, title, content, "
            "       fts_main_passages.match_bm25(id, ?) AS score "
            "FROM passages WHERE score IS NOT NULL ORDER BY score DESC LIMIT ?",
            [question, topk]).fetchall()
    finally:
        con.close()
    return [{"n": i + 1, "source": s, "title": t, "content": c, "score": sc}
            for i, (s, t, c, sc) in enumerate(rows)]


def page_ident(hit):
    """The retrieved-identity a claim cites — what `[n]` in the answer means."""
    return {"kind": "corpus", "source": hit["source"], "n": hit["n"]}


def hits_to_pages(hits):
    """Map retrieval hits to adapter Documents (what ground_answer/C1 expect)."""
    return [{"requested": page_ident(h), "retrieved": page_ident(h),
             "resolution": "exact", "text": h["content"]} for h in hits]


# --- synthesis (pluggable; the only step that needs the device) ---
def mock_grounded(question, hits):
    """Demo synth: quote a real whole-word span from the top hit and cite it."""
    if not hits:
        return []
    span = " ".join(hits[0]["content"].split()[:14])
    return [{"text": f"(answer to {question!r})", "span": span, "cites": page_ident(hits[0])}]


def mock_confabulated(question, hits):
    if not hits:
        return []
    return [{"text": "confabulated", "cites": page_ident(hits[0]),
             "span": "this exact sentence appears in none of the retrieved passages whatsoever"}]


def ask_grounded(question, db, topk=5, synth=mock_grounded):
    """The loop. -> (grounded-answer-or-abstain, hits)."""
    hits = retrieve(question, db, topk)
    pages = hits_to_pages(hits)
    claims = synth(question, hits)
    return G.ground_answer(question, claims, pages), hits


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="grounded ask loop (mock synth demo)")
    ap.add_argument("question")
    ap.add_argument("--db", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--confabulate", action="store_true", help="use the confabulating mock synth")
    a = ap.parse_args()
    result, hits = ask_grounded(a.question, a.db, a.topk,
                                synth=mock_confabulated if a.confabulate else mock_grounded)
    print(f"retrieved {len(hits)} hits: " + ", ".join(f"[{h['n']}] {h['source']}" for h in hits))
    print(json.dumps(result, indent=1))
