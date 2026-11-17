"""Vector retrieval over the unified int8 substrate, feeding `ground_answer`.

`seam.py` retrieves with DuckDB fts over a small docs corpus -- real retrieval, no
device, good for tests. This is the other backend: cosine over page vectors embedded
on the Tiiny's NPU, which is what the appliance actually runs.

SINGLE-STAGE int8 (2026-10-18; replaced the two-stage coarse->exact FLOAT+VSS path).
Ranking is brute-force cosine over int8 `vec TINYINT[1024]` Vortex shards, delegated
to cordexa `reference/retrieve_int8.py`. Why single-stage: for int8 a 256-d coarse
pass costs MORE than the exact scan (the stored array can't be cheaply sliced, and the
slice-then-cast is the DuckDB 1.5.6 segfault path), and quantisation recall (~.997) x
HNSW approximation recall multiply if stacked. Brute-force int8 has no index error and
runs ~2.4M rows/s; measured R@10 vs float32 ~.997. There is NO `shortlist` any more.

RETRIEVE ROUGH, GROUND EXACT. The int8 vectors embed only a truncated prefix of each
page, so ranking is rough. The page TEXT the grounding reads does NOT come from the
vector store (the Vortex has none) -- it comes from the federated text DBs, the WHOLE
verbatim page. So truncation costs recall, never correctness.

THREE THINGS THAT LOOK LIKE STYLE AND ARE NOT:

* The query MUST be embedded by the SAME encoder that built the index -- the NPU
  embedder, not the device's CPU one. Same model, different backends: cosine between
  their outputs is ~0.97 and the CPU's are L2-normalised while the NPU's are raw
  (norm ~100). Mixing them does not fail, it just quietly retrieves worse. (A CPU
  Qwen3-Embedding-0.6B-Q8 built the substrate vectors un-normalised, so a CPU-profile
  caller that embeds queries the same way is consistent; the guard is against mixing.)

* Page identity is `{kind, source, book, page}`. `source` is the extraction lane the
  page came from, and it is NOT decoration: re-OCR fixes bad pages inside books that
  already had a text layer, so the same (book, page) exists in two lanes with different
  text. Measured on the full corpus: 21,726 (book, page) pairs appear in BOTH the
  native-text and re-OCR lanes. Without `source` the locator is not unique. `source`
  is in the Vortex key AND the federation key, so it survives the whole path.

  The locator carries the finest position the index has, and is meant to grow: when
  passages become sub-page (Wikipedia articles, Gutenberg chapters), add `chapter`,
  `section` or a passage ordinal so a citation can still name exactly one passage.

* A page that runs on into the next is handed to C1 JOINED (see `hits_to_pages`), so a
  quotation crossing the page break can be verified at all -- 48.7% of adjacent page
  pairs cut a sentence. This is the load-bearing half of "ground exact" and is
  unchanged by the retrieval swap; it now reads its text from the federation.
"""
import importlib.util
import json
import os
import urllib.request

import duckdb

import continuity
import ground_answer as G
import locator

# int8 ranking lives in cordexa (the substrate owner). Locate the checkout via the shared
# `cordexa_home` helper (same path ground_answer's C1 and c3 resolve; handles the git-wt shape),
# then import retrieve_int8 by file, as ground_answer imports c1_span.
import cordexa_home   # noqa: E402  (tiibrarian flat module)


def _load_retrieve_int8():
    root = cordexa_home.find(os.path.join("reference", "retrieve_int8.py"))
    path = os.path.join(root, "reference", "retrieve_int8.py")
    spec = importlib.util.spec_from_file_location("cordexa_retrieve_int8", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_R = _load_retrieve_int8()

# The int8 Vortex shards (ranking) and the federation text DBs (grounding). Both on nvme2 by default;
# colon-delimited override. Text is routed to one of THREE federation DBs: survivorlibrary_* by prefix,
# the sources in WIKI_SOURCES by EXACT NAME, everything else to the zim DB.
SHARDS = os.environ.get(
    "TIIBRARIAN_VORTEX_SHARDS",
    "/srv/logical/opt/teague/corpus/survivorlibrary_vectors.vortex:"
    "/srv/logical/opt/teague/corpus/zim_vectors.vortex").split(":")
SL_DB = os.environ.get("TIIBRARIAN_SL_DB", "/srv/logical/opt/teague/corpus/survivorlibrary.duckdb")
ZIM_DB = os.environ.get("TIIBRARIAN_ZIM_DB", os.path.expanduser("~/corpus/zim-corpora.duckdb"))
# A corpus too large to live in the zim DB gets a text DB of its own. wikipedia_en_all_nopic is
# 14.2M chunks / 24 GB against the zim DB's 1.7M, so it is a separate file (substrate spec S4).
# Unset by default: with no WIKI_DB the routing below is exactly the two-way split it always was.
#
# ROUTED BY EXACT SOURCE NAME, NOT BY PREFIX, and that is load-bearing. The zim DB already holds
# wikipedia_en_simple_all_nopic_2026-05 (326,251) and wikipedia_en_medicine_maxi_2026-04 (214,771).
# A `wikipedia_` prefix rule sends those two to a file that does not hold them; `_text` then returns
# None, and `retrieve` DROPS a hit whose text is missing -- so two corpora would disappear from
# results with nothing raised anywhere. Colon-delimited, same shape as SHARDS.
WIKI_DB = os.environ.get("TIIBRARIAN_WIKI_DB", "")
WIKI_SOURCES = frozenset(s for s in os.environ.get(
    "TIIBRARIAN_WIKI_SOURCES", "wikipedia_en_all_nopic_2026-06").split(":") if s)
EMBED_URL = os.environ.get("TIINY_EMBED_URL", "http://api.tiiny/v1/embeddings")
AUTH = os.environ.get("TIINY_AUTH_KEY", "")
MODEL = os.environ.get("TIINY_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")

DIM = 1024
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


def connect(sl_db=None, zim_db=None, wiki_db=None):
    """An in-memory connection with the Vortex ext loaded and the text DBs attached read-only.

    Ranking reads the int8 Vortex shards (`read_vortex`, paths passed to `retrieve`); grounding reads
    page text from the attached federation DBs (`sl.pages` / `z.pages` / `w.pages`, see `_route`).
    `w` is attached only when a wiki DB is configured, so the default connection is unchanged.
    """
    con = duckdb.connect()
    con.execute("INSTALL vortex; LOAD vortex;")
    con.execute(f"ATTACH '{sl_db or SL_DB}' AS sl (READ_ONLY);")
    con.execute(f"ATTACH '{zim_db or ZIM_DB}' AS z (READ_ONLY);")
    if wiki_db or WIKI_DB:
        con.execute(f"ATTACH '{wiki_db or WIKI_DB}' AS w (READ_ONLY);")
    return con


def _route(con, source):
    """Which federation table holds this source's text. -> 'sl.pages' | 'w.pages' | 'z.pages'.

    Asks the CONNECTION whether `w` is attached rather than reading WIKI_DB, so a connection built
    with an explicit `wiki_db=` (or without one) routes to what it actually has open.
    """
    if source.startswith("survivorlibrary_"):
        return "sl.pages"
    if source in WIKI_SOURCES:
        if not con.execute(
                "SELECT 1 FROM duckdb_databases() WHERE database_name = 'w'").fetchone():
            # Half-configured: the vortex is being ranked but its text DB is not attached. Falling
            # through to z.pages would miss on every row, and `retrieve` drops a hit with no text,
            # so 14.2M chunks would go quiet with nothing raised -- the same silent shape as the
            # prefix trap, arrived at from the other side. `embed_query` sets the house rule here:
            # raise rather than degrade where the degrade is invisible.
            raise RuntimeError(
                f"source {source!r} is ranked from the wiki substrate but its text DB is not "
                f"attached. Set TIIBRARIAN_WIKI_DB (or pass wiki_db= to connect), or drop the "
                f"wiki vortex from TIIBRARIAN_VORTEX_SHARDS. Leaving it half-configured drops "
                f"every hit from this corpus silently.")
        return "w.pages"
    return "z.pages"


def _text(con, source, book, page):
    """The full verbatim page text from the federation, routed by `_route`. -> str|None."""
    tbl = _route(con, source)
    row = con.execute(
        f"SELECT text FROM {tbl} WHERE source = ? AND book = ? AND CAST(page AS VARCHAR) = ?",
        [source, book, str(page)]).fetchone()
    return row[0] if row and row[0] else None


#: How much a source-filtered call over-ranks before it trims. The filter is applied after
#: ranking -- the shards are not partitioned by source -- so asking for exactly topk would
#: return almost nothing once the other corpora are dropped.
SOURCE_FILTER_OVERSAMPLE = 200


def retrieve(qvec, topk=8, con=None, shards=None, sources=None):
    """Single-stage int8 vector retrieval, text attached from the federation. -> hits, best first.

    Ranking is cordexa `retrieve_int8` over the Vortex `shards` (default `SHARDS`; rough, truncated
    vectors); each hit's FULL verbatim text is then read from the federation (`_text`). A hit whose page
    has no text is dropped -- a miss is a value. The hit shape (n, source, book, page, text, score) is
    unchanged, so `hits_to_pages` / `next_page` / `page_ident` downstream are untouched.

    `sources` restricts the result to those source names. It belongs here rather than in a caller
    because the trim has to happen between ranking and the text reads, and only this function sees
    both.
    """
    if len(qvec) != DIM:
        raise ValueError(f"query vector has {len(qvec)} dims, index expects {DIM}")
    own = con is None
    con = con or connect()
    try:
        ranked = _R.retrieve(con, qvec, shards or SHARDS,
                             topk=topk * SOURCE_FILTER_OVERSAMPLE if sources else topk)
        if sources:
            # Trim BEFORE the text loop below. Each `_text` is its own federation query against a
            # table with no index on (source, book, page) -- ~0.03 s on the zim DB, measured -- so
            # filtering afterwards would pay that for every one of the over-ranked hits it discards.
            keep = frozenset(sources)
            ranked = [h for h in ranked if h["source"] in keep][:topk]
        out = []
        for h in ranked:
            t = _text(con, h["source"], h["book"], h["page"])
            if t and t.strip():
                out.append({"n": len(out) + 1, "source": h["source"], "book": h["book"],
                            "page": h["page"], "text": t, "score": h["score"]})
    finally:
        if own:
            con.close()
    return out


def page_ident(hit):
    """The retrieved-identity a claim cites: the finest position the index has.

    `source` is load-bearing, not decoration -- see the module docstring. Add finer
    fields here (chapter, section, passage ordinal) as the index gains them; a locator
    should always be able to name exactly one passage when the caller knows which.
    """
    return {"kind": KIND, "source": hit["source"], "doc": hit["book"],
            "page": hit["page"]}


def locator_str(hit):
    """The citation as a string: `somebook.pdf#p.27`.

    `doc` rather than `book` because the field is format-neutral -- the same locator
    shape names an HTML file, a Markdown note or a zim:// entry. See locator.py.
    """
    return locator.render({"doc": hit["book"], "page": hit["page"]})


def span_ident(hit, page_to):
    """A locator naming a run of pages: {..., page: 300, page_to: 301}."""
    d = page_ident(hit)
    if page_to != hit["page"]:
        d["page_to"] = page_to
    return d


def _page_plus_one(page):
    """The next page token, or None when the page is not an integer token (don't guess an order)."""
    s = str(page)
    return str(int(s) + 1) if s.lstrip("-").isdigit() else None


def next_page(con, hit):
    """The page after this hit, from the same book AND THE SAME LANE. -> text|None.

    Same lane matters: the corpus holds 21,726 (book, page) pairs in both the
    native-text and re-OCR lanes, so dropping `source` here would join a page to
    a different transcription of its neighbour. A non-integer page token (SL tokens
    are not guaranteed contiguous) yields None -- a missed join costs one claim; a
    wrong join corrupts the answer, so when the order is unknown we do not join.
    """
    nxt = _page_plus_one(hit["page"])
    if nxt is None:
        return None
    return _text(con, hit["source"], hit["book"], nxt)


def hits_to_pages(hits, con=None):
    """Hits -> adapter Documents, the shape ground_answer/C1 expect.

    A page whose text RUNS ON into the next is handed over joined, so a quotation
    that crosses the break can be checked at all. Measured: 48.7% of adjacent page
    pairs cut a sentence, and against single-page documents C1 calls every one of
    those quotations `unfounded` -- a real passage reported as confabulated.

    Joined only where `continuity.continues()` says the text actually continues.
    The ~15% of breaks that are chapter ends, section breaks or a table followed
    by prose are left alone: joining there would MANUFACTURE a passage that is not
    contiguous in the book, and C1 would grant it. A missed join costs one claim;
    a fabricated one corrupts the answer.

    `parts` carries the constituent single-page Documents so a grounded span can
    be narrowed back to the finest locator that really contains it -- see
    ground_answer._narrow. C1 ignores keys it does not know.
    """
    docs = []
    for h in hits:
        one = {"requested": page_ident(h), "retrieved": page_ident(h),
               "resolution": "exact", "text": h["text"]}
        nxt = next_page(con, h) if con is not None else None
        if nxt and continuity.continues(h["text"], nxt)[0]:
            joined = dict(one)
            joined["text"] = h["text"] + "\n" + nxt
            nxt_token = _page_plus_one(h["page"])
            joined["requested"] = joined["retrieved"] = span_ident(h, nxt_token)
            joined["parts"] = [one]
            docs.append(joined)
        else:
            docs.append(one)
    return docs


def ask_grounded(question, synth, topk=8, qvec=None, con=None, shards=None):
    """question -> retrieve -> synthesize -> ground. -> (answer|abstain, hits).

    `qvec` lets a caller supply the embedding (a device-free test, or a
    "more like this page" query). Otherwise the question is embedded on the NPU.
    """
    if qvec is None:
        qvec = embed_query(question)
    own = con is None
    con = con or connect()
    try:
        hits = retrieve(qvec, topk=topk, con=con, shards=shards)
        pages = hits_to_pages(hits, con=con)
    finally:
        if own:
            con.close()
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
    ap = argparse.ArgumentParser(description="grounded ask over the unified int8 substrate")
    ap.add_argument("question")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--mock", action="store_true",
                    help="quote the top hit instead of asking the device's LLM")
    a = ap.parse_args()
    if a.mock:
        chosen = quote_top_hit
    else:
        import synth
        chosen = synth.device_synth
    result, hits = ask_grounded(a.question, chosen, topk=a.topk)
    for h in hits:
        print(f"[{h['n']}] {h['score']:.3f}  {h['book'][:56]} p.{h['page']}")
    print(json.dumps(result, indent=1)[:1200])
