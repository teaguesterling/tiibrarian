"""The librarian over the zimshelf — grounded answers from the offline ZIM archives.

    question -> cordexa's query ladder over a ZIM full-text index
             -> pages (cordexa's own HTML extraction)
             -> synthesize -> ground_answer -> answer | abstain

Same seam as `seam.py` (DuckDB fts over a docs corpus) and `corpus.py` (the NPU
page-vector index): a `retrieve()`, a `page_ident()`, a `hits_to_pages()`, and the
SAME `ground_answer`. Only the retrieval end differs, which is the point of having
a seam at all — nothing in the grounding layer knows a ZIM archive from a scanned
manual.

WHAT IS BORROWED FROM CORDEXA, AND WHY NOT REIMPLEMENTED

  passages.content_words   a claim's content words, stopworded the same way
  retrieval.rank_documents the relaxation ladder: and-all, then and-drop1, then
                           and-pairs. A ZIM full-text query is AND-by-default, so
                           a long question matches nothing without it — this is
                           the single most important thing to reuse rather than
                           re-derive.
  zim_readers.DuckdbReader the DuckDB `zim` extension as the search backend
  extract.extract          HTML -> text, with cordexa's drop rules (infobox
                           tables, scripts, mwoffliner footers)

Using cordexa's extractor rather than writing one here is not tidiness: it is what
makes a span this layer grounds identical to a span cordexa's ZIM adapter would
have produced for the same entry, so a citation means the same thing on both
sides. A second extractor would quietly fork the text, and C1 compares text.

NO LIBZIM, AND THEREFORE NO 3.12 VENV. cordexa's ZimAdapter imports python-libzim
at module load and so is pinned to 3.12; this module deliberately imports only the
four stdlib/DuckDB-backed pieces above, which run on the system interpreter. That
is why retrieval goes through `DuckdbReader` directly instead of through
`ZimAdapter.fetch`: not to avoid the adapter's contract, but because importing it
would drag in an interpreter pin this layer does not need.

What is given up by not going through `ZimAdapter.fetch`: the Unavailable
vocabulary, the redirect chain, `archive_uuid`/`expected_sha256` revalidation. For
an interactive ask over a local shelf those are not load-bearing; for a recorded
run they are, and that run should use the adapter. Said plainly here so the choice
is visible rather than discovered.
"""
import os
import re
import sys

import ground_answer as G
import cordexa_home

CORDEXA = cordexa_home.find(os.path.join("corpus", "zim_readers.py"))

#: cordexa lays these out as two importable directories rather than a package, and
#: `retrieval` imports `passages`/`zim_readers` by bare name, so both have to be on
#: the path before any of them is imported. Idempotent: re-running costs nothing.
for _d in (os.path.join(CORDEXA, "corpus"), os.path.join(CORDEXA, "adapters", "zim")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from extract import extract                 # noqa: E402  cordexa's HTML -> text
from passages import content_words          # noqa: E402
from retrieval import rank_documents        # noqa: E402
from zim_readers import DuckdbReader        # noqa: E402

KIND = "zim"

#: Where the archives live. The shelf's canonical directory first; `~/Projects/Tiiny`
#: holds Gutenberg and the mini Wikipedia. `$TIIBRARIAN_ZIM_DIRS` overrides, and
#: `$CORDEXA_ZIM_DIRS` is honoured so one setting serves both layers.
ARCHIVE_DIRS = [
    d for d in (os.environ.get("TIIBRARIAN_ZIM_DIRS")
                or os.environ.get("CORDEXA_ZIM_DIRS")
                or "/srv/logical/portable/zim:" + os.path.expanduser("~/Projects/Tiiny")
                ).split(":") if d
]

MAX_RELAX = os.environ.get("TIIBRARIAN_MAX_RELAX", "and-drop1")

_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def archives():
    """-> [(archive_id, path)] for every .zim on the shelf, first directory wins."""
    seen, out = set(), []
    for d in ARCHIVE_DIRS:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".zim") and name[:-4] not in seen:
                seen.add(name[:-4])
                out.append((name[:-4], os.path.join(d, name)))
    return out


def archive_file(archive):
    """-> the .zim path for an archive id. Raises with the directories tried."""
    for aid, path in archives():
        if aid == archive:
            return path
    raise FileNotFoundError(
        f"no archive {archive!r}.zim under {ARCHIVE_DIRS}. "
        f"Available: {', '.join(a for a, _ in archives()) or '(none)'}")


def locator_str(archive, path):
    """`zim://<archive>.zim/<entry>` — duckeye's scheme, which `locator.py` parses.

    The identity a claim cites. It must be the string form, because a model writes
    a string and `ground_answer` parses it back with `locator.parse`; the parsed
    form is `{"doc": <this string>}`, which is why `page_ident` carries `doc`.
    """
    return f"zim://{archive}.zim/{path}"


def page_ident(hit):
    """The retrieved-identity a claim cites.

    `doc` is NOT redundant beside id/path: `_same_page` compares the keys the two
    sides share, and a citation that arrived as a locator string parses to `{"doc":
    ...}` alone. Without `doc` here the intersection is empty, every claim fails to
    match its page, and the librarian abstains on answers that were in fact correct.

    `page` IS the passage ordinal, and without it this seam was broken for the
    common case. `vector_retrieve` returns CHUNKS, and several of them are
    routinely from one article -- topk defaults to 5 over a 1.73M-chunk index. Two
    chunks of one article then produced byte-identical identities, `_same_page`
    matched a citation to both, and `ground_answer` dropped a correctly-quoted,
    correctly-cited claim as `ambiguous`. corpus.py:35 had already written the fix
    down for its own corpus -- "when passages become sub-page (Wikipedia articles,
    Gutenberg chapters), add chapter, section or a passage ordinal so a citation
    can still name exactly one passage" -- and this is exactly that case.
    """
    return {"kind": KIND, "id": hit["archive"], "path": hit["path"],
            "doc": hit["locator"], "page": hit["chunk"]}


def _html(con, path, entry):
    row = con.execute("SELECT zim_get_text(?, ?)", [path, entry]).fetchone()
    return row[0] if row and row[0] else ""


def title_candidates(con, file, words, limit):
    """Entry paths from the TITLE index, via a short ladder of its own. -> [path].

    Needed because full-text ranking alone is not good enough to be usable, and the
    way it fails is specific. Measured on wikipedia_en_simple:

      "What is tidal range?"      -> Selenography, Moon_landing, Lunar_eclipse
      "What causes a lunar eclipse?" -> March_2025_lunar_eclipse, Solar_eclipse_2020

    The first is nav-box boilerplate: the only place the phrase "tidal range" occurs
    in Selenography is a see-also list, and Xapian counts it like prose. (There is no
    `Tidal_range` article at all; the subject lives at `Tide`.) The second is
    near-duplicate specific articles outranking the general one. Neither is fixable
    by relaxing the full-text query — they are failures OF full-text.

    `zim_suggest` is a title matcher, so it needs a title-shaped string: the whole
    question returns nothing, and so does "tidal range" (there is no such title).

    MULTI-WORD ONLY, and that bound is the whole design. Measured, every junk result
    came from a ONE-word suggestion, because a bare content word is a different
    subject that happens to share a prefix:

        "causes" -> Causes_of_World_War_I      (from "What causes a lunar eclipse?")
        "tidal"  -> Tidal (album)              (from "What is tidal range?")
        "range"  -> Range (mathematics)
        "bay"    -> Bay

    A first attempt walked prefixes down to one word and made retrieval WORSE on
    three of four questions. Two words is enough context to be a subject rather than
    a prefix collision, so the ladder stops at n=2 and contributes nothing rather
    than something wrong — full text still runs either way. The single exception is a
    question whose content is one word, where that word IS the subject.

    Longest n-gram first, every position, because neither end of a question reliably
    holds the subject: "lunar eclipse" needs the TAIL ("causes" leads), while
    "photosynthesis work" needs the HEAD. Trying all contiguous spans covers both
    without a rule about question shape that the next question would break.
    """
    floor = 1 if len(words) == 1 else 2
    for n in range(len(words), floor - 1, -1):
        for i in range(0, len(words) - n + 1):
            q = " ".join(words[i:i + n])
            rows = con.execute(
                "SELECT path FROM zim_suggest(?, ?, max_results := ?)", [file, q, limit]
            ).fetchall()
            if rows:
                return [r[0] for r in rows]
    return []


def retrieve(question, archive, topk=5, max_relax=MAX_RELAX, titles=True, title_k=None):
    """Question -> ranked entries from one archive, with their extracted text.

    Two signals, title index first and full text after, deduplicated. The ladder is
    what makes full text work at all: a ZIM query is a conjunction, so asking it the
    whole question matches nothing — `content_words` strips it down and
    `rank_documents` relaxes until something matches.

    `title_k` is capped below `topk` on purpose, so the title index can never crowd
    the full-text hits out entirely: titles answer "what is X", full text answers
    "where is X discussed", and a question often wants the second.
    """
    import duckdb

    file = archive_file(archive)
    words = content_words(question)
    if not words:
        return []
    reader = DuckdbReader(file)
    _form, ranked = rank_documents(reader, words, max_relax)

    con = duckdb.connect()
    con.execute("LOAD zim")
    hits = []
    try:
        tk = title_k if title_k is not None else max(1, topk // 2)
        order, seen = [], set()
        for p in (title_candidates(con, file, words, tk) if titles else []):
            if p not in seen:
                seen.add(p); order.append(p)
        for p, _h in ranked:
            if p not in seen:
                seen.add(p); order.append(p)

        for entry in order:
            if len(hits) >= topk:
                break
            if not con.execute("SELECT zim_mimetype(?, ?)", [file, entry]).fetchone()[0] == "text/html":
                continue                     # images, css, js — cordexa calls this unsupported_type
            html = _html(con, file, entry)
            if not html:
                continue
            text, _stats = extract(html)
            if not text.strip():
                continue                     # cordexa's empty_after_extraction
            m = _TITLE.search(html)
            hits.append({
                "n": len(hits) + 1, "archive": archive, "path": entry,
                "title": (m.group(1).strip() if m else entry.replace("_", " ")),
                "text": text, "locator": locator_str(archive, entry),
            })
    finally:
        con.close()
    return hits


#: The int8 Vortex shards this shelf ranks. `corpus.SHARDS` also carries
#: survivorlibrary's 12.9 GB of scanned manuals, which are not on this shelf, so the
#: shelf names its own and they are the ZIM side of the federation only. Add the
#: Wikipedia shard here to put its 14.2M chunks on the shelf -- and set
#: $TIIBRARIAN_WIKI_DB along with it, or `corpus._route` raises rather than let every
#: wiki hit fall through to a DB that does not hold it.
SHARDS = os.environ.get(
    "TIIBRARIAN_ZIM_SHARDS",
    "/srv/logical/opt/teague/corpus/zim_vectors.vortex").split(":")

def _ordinal(page):
    """corpus's `page` -> the INT passage ordinal this module cites.

    THE TYPE SEAM BETWEEN THE TWO LAYERS, and it is load-bearing. corpus.py keys
    `page` as a VARCHAR because that is how the Vortex shards store it (measured:
    both shards yield '0', not 0), and its citation vocabulary is strings
    throughout -- `_page_plus_one` returns `str(int(s) + 1)`. zimshelf's is ints,
    read from `z.pages.page` BIGINT and pinned by
    `test_identity_carries_the_passage_ordinal`.

    `ground_answer._same_page` compares with a bare `==`. Carrying '7' into a
    citation therefore makes `"7" == 7` False and drops a correctly quoted,
    correctly cited claim -- the same silent failure 3ff1463 fixed, reached from
    the other side. Converting here keeps both vocabularies intact.

    Raises rather than passing a non-ordinal through, because that degrade is
    invisible: the claim is simply dropped and the librarian abstains.
    """
    s = str(page)
    if not s.lstrip("-").isdigit():
        raise ValueError(
            f"page {page!r} is not a passage ordinal, so a citation cannot name one chunk "
            f"with it. This shelf ranks ZIM shards, whose page IS the chunk index; a "
            f"non-numeric page means $TIIBRARIAN_ZIM_SHARDS holds a shard from a corpus "
            f"that pages differently.")
    return int(s)


def _titles(con, hits):
    """-> {(source, book): title} for these hits. One query per federation table.

    `corpus.retrieve`'s hit shape carries no title -- that shape is the grounding
    seam's and stays as it is -- but the shelf prints one, so it is read here.

    Which table holds a source is `corpus._route`'s question, not a second rule
    written out again here. The map it owns is BY EXACT SOURCE NAME: a
    `wikipedia_` prefix rule would send simple-wiki and medicine-wiki to a file
    that does not hold them, and a title miss would quietly become the entry path.
    """
    import corpus as C

    want = {}
    for h in hits:
        want.setdefault(C._route(con, h["source"]), set()).add((h["source"], h["book"]))
    out = {}
    for tbl, keys in want.items():
        for src in sorted({s for s, _ in keys}):
            books = sorted(b for s, b in keys if s == src)
            marks = ", ".join(["?"] * len(books))
            rows = con.execute(
                f"SELECT book, any_value(title) FROM {tbl} "
                f"WHERE source = ? AND book IN ({marks}) GROUP BY book",
                [src] + books).fetchall()
            out.update({(src, b): t for b, t in rows})
    return out


def vector_retrieve(question, archive=None, topk=5, con=None, shards=None):
    """Semantic retrieval over the int8 substrate. Same hit shape as `retrieve`.

    THIS IS THE MODE THAT WORKS, and the difference is not marginal. Measured on
    the same four questions, full text vs vectors:

        "What is tidal range?"        Selenography          -> Tide
        "How do tides work?"          Happisburgh_archaeology -> Tide
        "What is a tidal bore?"       River_Trent           -> Tide
        "What causes a lunar eclipse?" March_2025_lunar_eclipse -> Lunar_eclipse

    Full text fails on these because a ZIM query is a conjunction over content
    words, so one low-information word from the question ("work", "causes")
    dominates the ranking, and nav-box link lists count the same as prose. Neither
    is a tuning problem; embeddings simply do not have either failure.

    RANKING IS `corpus.retrieve`, NOT A SECOND IMPLEMENTATION OF IT. This module
    used to run its own two-stage query against one DuckDB file: a coarse HNSW
    over `vec256 FLOAT[256]`, reordered by exact cosine over `vec FLOAT[1024]`.
    `5b57e74` retired that in corpus.py for single-stage int8 over Vortex and left
    this copy behind, so the two layers no longer agreed on what the index was.

    WHAT IT COSTS ON THE SHELF AS IT SHIPS: nothing worth naming. Measured on
    longbottom against the default zim-only SHARDS, end to end through this
    function: 0.68 s unfiltered, 0.80 s filtered, both over 1.73M chunks, against
    the retired HNSW path's sub-second on the same data.

    WHAT IT COSTS WITH THE WIKIPEDIA SHARD IS UNSETTLED, AND THE TWO READINGS
    DISAGREE. An earlier session measured 5.8-6.9 s/query over 18.8M chunks
    against this same packed file, through grimoire-kb's `unified_retrieve`.
    Here, through `retrieve_int8`, one full scan of that shard did not finish in
    900 s, while the zim shard ranked at 1.7-3.2M rows/s in the same process. The
    box showed a load average above 20 when it was sampled, which does not
    obviously explain an 8x larger file taking more than 100x longer. The cause
    is NOT known. Do not quote a Wikipedia latency from this docstring; measure
    it on the machine that will serve the queries.

    What it buys: every corpus corpus.py can reach, Wikipedia's 14.2M chunks among
    them, and one place where a change to retrieval has to be made instead of two.

    `archive` is handed DOWN as `corpus.retrieve(sources=...)` rather than applied
    to the result here. The shards are not partitioned by source, so a filter
    always costs an over-rank; the trim must then happen before the text reads,
    because `corpus.retrieve` runs one federation query per hit it returns. Doing
    it here instead would read -- and throw away -- the text of every over-ranked
    hit, ~0.03 s apiece against the zim DB.
    """
    import corpus as C

    qvec = C.embed_query(question)
    own = con is None
    con = con or C.connect()
    try:
        ranked = C.retrieve(qvec, topk=topk, con=con, shards=shards or SHARDS,
                            sources=[archive] if archive else None)
        titles = _titles(con, ranked)
    finally:
        if own:
            con.close()

    return [{"n": i + 1, "archive": h["source"], "path": h["book"],
             "chunk": _ordinal(h["page"]),
             "title": titles.get((h["source"], h["book"])) or h["book"].replace("_", " "),
             "text": h["text"],
             "locator": locator_str(h["source"], h["book"]), "score": h["score"]}
            for i, h in enumerate(ranked)]


def hits_to_pages(hits):
    """Hits -> the adapter Documents `ground_answer`/C1 expect.

    `requested` equals `retrieved` because the ladder names an entry path and that
    exact path is read back — there is no title search and no redirect hop to drift
    through. If a redirect is ever followed here, these must stop being equal or
    C1.8's drift check is disabled (see ground_answer._drift).
    """
    return [{"requested": page_ident(h), "retrieved": page_ident(h),
             "resolution": "exact", "text": h["text"]} for h in hits]


def quote_top_hit(question, hits, words=14):
    """Device-free demo synth: quote a real span from the top hit and cite it."""
    if not hits:
        return []
    span = " ".join(hits[0]["text"].split()[:words])
    # Cite the STRUCTURED identity, not the bare locator string. A locator names
    # the article; the claim is about one passage in it, and a doc-only citation
    # cannot name which -- so it matches every retrieved chunk of that article and
    # `ground_answer` calls the claim ambiguous. A model that emits a bare string
    # still hits that limit; see the note in `page_ident`.
    return [{"text": f"(answer to {question!r})", "span": span,
             "cites": page_ident(hits[0])}]


def ask_grounded(question, archive=None, synth=quote_top_hit, topk=5,
                 max_relax=MAX_RELAX, titles=True, mode="vector", support=False):
    """The loop. -> (grounded-answer-or-abstain, hits).

    `mode` defaults to vector because that is the one that answers questions; fts
    is kept because it needs no index and no device, which is what makes a
    device-free run and a hermetic test possible.

    In vector mode `archive` is optional and omitting it is usually right: the
    index spans 19 corpora, so a question can be answered from wikibooks when
    simple-wiki has nothing, and the locator still says which archive it came from.

    `support` adds the C3 check — does the cited span actually STATE the claim? —
    and is off by default because it needs the NPU and costs ~6-9 s per kept
    claim (two askings of a 30B each). It annotates; it never drops a claim. See
    `ground_answer.annotate_support`.
    """
    if mode == "vector":
        hits = vector_retrieve(question, archive, topk)
    elif mode == "fts":
        if not archive:
            raise ValueError("fts mode searches ONE archive; pass archive=")
        hits = retrieve(question, archive, topk, max_relax, titles=titles)
    else:
        raise ValueError(f"mode must be 'vector' or 'fts', got {mode!r}")
    pages = hits_to_pages(hits)
    result = G.ground_answer(question, synth(question, hits), pages)
    if support:
        # Imported HERE, not at module scope: c3 resolves CORDEXA_HOME and the
        # device token at import, so a device-free run must not pay for it.
        import c3
        result = G.annotate_support(result, c3.check_kept)
    return result, hits


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="grounded ask over the zimshelf")
    ap.add_argument("question", nargs="?")
    ap.add_argument("--archive", help="archive id; required for fts, optional filter for vector")
    ap.add_argument("--mode", default="vector", choices=("vector", "fts"),
                    help="vector: the passage index (answers questions). fts: the "
                         "ZIM full-text index (no index file, no device, weaker)")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max-relax", default=MAX_RELAX, choices=("and-all", "and-drop1", "and-pairs"))
    ap.add_argument("--list", action="store_true", help="list the archives on the shelf")
    ap.add_argument("--no-titles", action="store_true",
                    help="full-text only, no title-index blend (to measure what it buys)")
    ap.add_argument("--device", action="store_true", help="synthesize with the device LLM")
    ap.add_argument("--support", action="store_true",
                    help="C3: check each kept span actually STATES its claim "
                         "(needs the NPU, ~6-9s per claim; annotates, never drops)")
    a = ap.parse_args()

    if a.list or not a.question:
        for aid, path in archives():
            print(f"{aid:52s} {os.path.getsize(path) / 1e9:8.2f} GB")
        raise SystemExit(0 if a.list else 2)
    if a.mode == "fts" and not a.archive:
        ap.error("--archive is required for --mode fts (see --list)")

    synth = quote_top_hit
    if a.device:
        import synth as S
        synth = S.device_synth

    result, hits = ask_grounded(a.question, a.archive, synth, a.topk, a.max_relax,
                                titles=not a.no_titles, mode=a.mode, support=a.support)
    for h in hits:
        print(f"  [{h['n']}] {h['locator']}  ({h['title']})")
    print(json.dumps(result, indent=1))
