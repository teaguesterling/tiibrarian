"""Vector retrieval over a SurvivorLibrary corpus `.duckdb`, feeding `ground_answer`.

`seam.py` retrieves with DuckDB fts over a small docs corpus -- real retrieval, no
device, good for tests. This is the other backend: cosine over page vectors embedded
on the Tiiny's NPU, which is what the appliance actually runs.

TWO-STAGE, and both halves are load-bearing (measured on 1,310,336 real pages):
  1. coarse: HNSW over a stored 256-d truncation of each vector
  2. exact:  re-rank the shortlist on the full 1024-d vector
Recall against exact search is 0.987 at shortlist 100 and 0.996 at 200, so the default
is 200. The coarse index is a quarter the size of a full-precision one.

THREE THINGS THAT LOOK LIKE STYLE AND ARE NOT:

* The re-rank is a SUBQUERY, never a CTE. A CTE defeats DuckDB's HNSW optimizer and the
  coarse step silently becomes a full scan -- measured 2.8 s versus 31 ms. Same reason
  the truncation is a materialised column: VSS ignores the index on a
  `GENERATED ... VIRTUAL` column, with no error, just a seq scan.

* The query MUST be embedded by the SAME encoder that built the index -- the NPU
  embedder, not the device's CPU one. Same model, different backends: cosine between
  their outputs is ~0.97 and the CPU's are L2-normalised while the NPU's are raw
  (norm ~100). Mixing them does not fail, it just quietly retrieves worse.

* Page identity is `{kind, book, page}` -- discriminating on purpose. `ground_answer`
  requires a citation to resolve to exactly ONE retrieved page; an identity that omits
  `page` would match every page of a book and hand back a false `grounded`.
"""
import json
import os
import urllib.request

import duckdb

import ground_answer as G

CORPUS = os.environ.get(
    "TIIBRARIAN_CORPUS",
    "/srv/logical/workspace/home/teague/corpus/survivorlibrary-testset.duckdb")
EMBED_URL = os.environ.get("TIINY_EMBED_URL", "http://api.tiiny/v1/embeddings")
AUTH = os.environ.get("TIINY_AUTH_KEY", "")
MODEL = os.environ.get("TIINY_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

DIM = 1024
COARSE = 256
KIND = "survivorlibrary"


def embed_query(text, timeout=60):
    """Embed a question on the Tiiny's NPU. Needs the device and a token.

    Raises rather than falling back to another encoder: a wrong-encoder query returns
    plausible, worse hits with no error, which is the failure mode hardest to notice.
    """
    if not AUTH:
        raise RuntimeError(
            "corpus.embed_query needs TIINY_AUTH_KEY (and the device reachable at "
            f"{EMBED_URL}). Retrieval by vector still works without it -- pass a query "
            "vector to retrieve() directly.")
    body = json.dumps({"model": MODEL, "input": text}).encode()
    req = urllib.request.Request(
        EMBED_URL, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + AUTH})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        vec = json.loads(r.read())["data"][0]["embedding"]
    if len(vec) != DIM:
        raise RuntimeError(f"embedder returned {len(vec)} dims, index expects {DIM}")
    return vec


def connect(db=None):
    return duckdb.connect(db or CORPUS, read_only=True)


def retrieve(qvec, db=None, topk=8, shortlist=200, con=None):
    """Two-stage vector retrieval. -> hits, best first."""
    if len(qvec) != DIM:
        raise ValueError(f"query vector has {len(qvec)} dims, index expects {DIM}")
    own = con is None
    con = con or connect(db)
    try:
        con.execute("LOAD vss")
        rows = con.execute(
            # SUBQUERY, NOT A CTE -- see the module docstring.
            "SELECT book, page, text, "
            "       array_cosine_distance(vec, ?::FLOAT[{d}]) AS dist "
            "FROM (SELECT book, page, text, vec FROM pages "
            "      ORDER BY array_cosine_distance(vec256, ?::FLOAT[{c}]) LIMIT ?) "
            "ORDER BY dist LIMIT ?".format(d=DIM, c=COARSE),
            [qvec, qvec[:COARSE], shortlist, topk]).fetchall()
    finally:
        if own:
            con.close()
    return [{"n": i + 1, "book": b, "page": p, "text": t, "score": 1.0 - d}
            for i, (b, p, t, d) in enumerate(rows)]


def page_ident(hit):
    """The retrieved-identity a claim cites. Must identify exactly one page."""
    return {"kind": KIND, "book": hit["book"], "page": hit["page"]}


def hits_to_pages(hits):
    """Hits -> adapter Documents, the shape ground_answer/C1 expect."""
    return [{"requested": page_ident(h), "retrieved": page_ident(h),
             "resolution": "exact", "text": h["text"]} for h in hits]


def ask_grounded(question, synth, db=None, topk=8, shortlist=200, qvec=None, con=None):
    """question -> retrieve -> synthesize -> ground. -> (answer|abstain, hits).

    `qvec` lets a caller supply the embedding (a device-free test, or a
    "more like this page" query). Otherwise the question is embedded on the NPU.
    """
    if qvec is None:
        qvec = embed_query(question)
    hits = retrieve(qvec, db=db, topk=topk, shortlist=shortlist, con=con)
    pages = hits_to_pages(hits)
    return G.ground_answer(question, synth(question, hits), pages), hits


def quote_top_hit(question, hits, words=14):
    """Demo synth: quote a real span from the top hit and cite it correctly."""
    if not hits:
        return []
    span = " ".join(hits[0]["text"].split()[:words])
    return [{"text": f"(answer to {question!r})", "span": span,
             "cites": page_ident(hits[0])}]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="grounded ask over the corpus index")
    ap.add_argument("question")
    ap.add_argument("--db", default=None)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--shortlist", type=int, default=200)
    a = ap.parse_args()
    result, hits = ask_grounded(a.question, quote_top_hit, db=a.db,
                                topk=a.topk, shortlist=a.shortlist)
    for h in hits:
        print(f"[{h['n']}] {h['score']:.3f}  {h['book'][:56]} p.{h['page']}")
    print(json.dumps(result, indent=1)[:1200])
