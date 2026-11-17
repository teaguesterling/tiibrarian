#!/usr/bin/env python3
"""zimshelf's identity seam — the default user-facing path, previously untested.

    CORDEXA_HOME=~/Projects/cordexa/main python3 -m unittest test_zimshelf -v

Hermetic: hand-built hits in the shape `vector_retrieve` really returns. No
archive, no index, no device.

THE DEFECT THIS WAS WRITTEN FOR. `page_ident` returned {kind, id, path, doc}
with no passage ordinal, while `vector_retrieve` returns several CHUNKS of the
same article (topk defaults to 5 over a 1.73M-chunk index, so this is common,
not a corner). Two chunks of one article therefore produced byte-identical
identities, `_same_page` matched the citation to both, and ground_answer called
a correctly-quoted, correctly-cited claim `ambiguous` and dropped it.

corpus.py:35 had already written down the fix for its own corpus: "when
passages become sub-page (Wikipedia articles, Gutenberg chapters), add chapter,
section or a passage ordinal so a citation can still name exactly one passage."
zimshelf is exactly that case and had not done it.
"""
import unittest

import ground_answer as G
import zimshelf as Z


def hit(n, chunk, text, path="Tide", archive="wikipedia_en_simple"):
    return {"n": n, "archive": archive, "path": path, "chunk": chunk,
            "locator": Z.locator_str(archive, path), "title": path, "text": text}


A = hit(1, 0, "Tides are the rise and fall of sea levels caused by the Moon and Sun.")
B = hit(2, 7, "Spring tides occur at the new moon and the full moon each month.")


class TestIdentity(unittest.TestCase):

    def test_two_chunks_of_one_article_are_not_the_same_page(self):
        """THE defect. Identical identities make every multi-chunk hit ambiguous."""
        self.assertNotEqual(Z.page_ident(A), Z.page_ident(B),
                            "two chunks of one article must not share an identity")

    def test_identity_carries_the_passage_ordinal(self):
        self.assertEqual(Z.page_ident(B).get("page"), 7)

    def test_doc_is_present(self):
        """`_same_page` intersects shared keys, and a citation that arrived as a
        locator STRING parses to {"doc": ...} alone — so dropping `doc` makes the
        intersection empty and abstains on correct answers."""
        self.assertIn("doc", Z.page_ident(A))
        self.assertEqual(Z.page_ident(A)["doc"], A["locator"])

    def test_requested_equals_retrieved(self):
        """C1.8's drift check compares the two; zimshelf has no redirect step, so
        they must be equal or every page reports drift."""
        for p in Z.hits_to_pages([A, B]):
            self.assertEqual(p["requested"], p["retrieved"])


class TestGroundingEndToEnd(unittest.TestCase):

    def test_a_correct_claim_on_a_multi_chunk_article_is_kept(self):
        pages = Z.hits_to_pages([A, B])
        claim = {"text": "Tides are caused by the Moon and Sun.",
                 "span": "Tides are the rise and fall of sea levels caused by the Moon and Sun",
                 "cites": Z.page_ident(A)}
        r = G.ground_answer("what causes tides", [claim], pages)
        v = (r["kept"] + r["dropped"])[0]
        self.assertEqual(v["verdict"], "grounded",
                         "a verbatim span cited to its own chunk must ground; got %r (%s)"
                         % (v["verdict"], v.get("why", "")))
        self.assertEqual(v["citation_matched"], 1)


class TestModeGuards(unittest.TestCase):

    def test_fts_without_archive_is_refused(self):
        with self.assertRaises(ValueError):
            Z.ask_grounded("q", archive=None, mode="fts")

    def test_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            Z.ask_grounded("q", mode="sideways")


class TestTitleCandidates(unittest.TestCase):
    """The n-gram floor is the module's one measured design decision: a first
    attempt walked prefixes down to ONE word and made retrieval worse on three
    of four questions, so single-word suggestions are only tried when the whole
    query is one word."""

    # `words` is a LIST of content words, not a string -- passing a string makes
    # the ladder iterate CHARACTERS, which is how the first draft of this test
    # "failed": len("tidal range...") is 22, so the floor was satisfied by
    # single letters. The real caller passes cordexa's `content_words` output.
    class FakeCon:
        def __init__(self): self.queries = []
        def execute(self, sql, params=None):
            self.queries.append((sql, params)); return self
        def fetchall(self): return []

    def test_multi_word_query_never_asks_for_a_one_word_title(self):
        con = self.FakeCon()
        Z.title_candidates(con, "zim", ["tidal", "range", "on", "the", "coast"], 5)
        asked = [p[1] for _, p in con.queries if p and len(p) > 1 and isinstance(p[1], str)]
        singles = [a for a in asked if a and len(a.split()) == 1]
        self.assertEqual(singles, [], "one-word title probes were measured to hurt")

    def test_single_word_query_is_still_allowed_one_word(self):
        con = self.FakeCon()
        Z.title_candidates(con, "zim", ["photosynthesis"], 5)
        self.assertTrue(con.queries, "a one-word query must still probe titles")


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# The int8 migration: zimshelf ranks through corpus.retrieve.
#
# `5b57e74` retired the two-stage FLOAT[1024]+vec256+HNSW retrieve in corpus.py
# and left `zimshelf.vector_retrieve` running it. These pin the seam that
# replaces it. Still hermetic: corpus's three entry points are stubbed, so no
# index, no device and no 39 GB of substrate are needed.
# ---------------------------------------------------------------------------

import contextlib                                              # noqa: E402
from unittest import mock                                      # noqa: E402


def corpus_hit(source, book, page, text, score=0.9):
    """A hit in the shape `corpus.retrieve` REALLY returns.

    `page` is a VARCHAR and that is not an accident of this fixture: both live
    Vortex shards key it as one (measured -- zim_vectors and wikipedia_vectors
    each yield '0', not 0), while both federation text DBs store BIGINT.
    """
    return {"n": 1, "source": source, "book": book, "page": page,
            "text": text, "score": score}


@contextlib.contextmanager
def fake_corpus(hits, titles=None):
    """Stub corpus's device, connection and ranking. -> the `retrieve` mock."""
    import corpus as C
    with mock.patch.object(C, "embed_query", return_value=[0.0] * 1024), \
         mock.patch.object(C, "connect", return_value=mock.MagicMock()), \
         mock.patch.object(C, "retrieve", return_value=list(hits)) as r, \
         mock.patch.object(Z, "_titles", return_value=titles or {}):
        yield r


WIKI = corpus_hit("wikipedia_en_all_nopic_2026-06", "Tide", "7",
                  "Spring tides occur at the new moon and the full moon.")


class TestRanksThroughCorpus(unittest.TestCase):

    def test_ranking_delegates_to_the_single_stage_int8_retrieve(self):
        """The retired path opened a DuckDB index file and ran `LOAD vss`. This one
        must go through corpus.retrieve instead, or the two layers have diverged
        again the moment the next change lands."""
        with fake_corpus([WIKI]) as r:
            Z.vector_retrieve("how do tides work")
        self.assertEqual(r.call_count, 1, "vector_retrieve must rank via corpus.retrieve")

    def test_the_hit_shape_the_shelf_promises_is_unchanged(self):
        """page_ident, hits_to_pages, quote_top_hit and the CLI all read these keys."""
        with fake_corpus([WIKI]):
            hits = Z.vector_retrieve("how do tides work")
        self.assertEqual(
            set(hits[0]), {"n", "archive", "path", "chunk", "title", "text", "locator", "score"})
        self.assertEqual(hits[0]["archive"], "wikipedia_en_all_nopic_2026-06")
        self.assertEqual(hits[0]["path"], "Tide")
        self.assertEqual(hits[0]["locator"], "zim://wikipedia_en_all_nopic_2026-06.zim/Tide")


class TestTheChunkOrdinalStaysAnInt(unittest.TestCase):
    """THE trap this migration had to clear.

    corpus.py's citation vocabulary is STRINGS end to end (`_page_plus_one`
    returns `str(int(s) + 1)`). zimshelf's is INTS, read from `z.pages.page`
    BIGINT, and `test_identity_carries_the_passage_ordinal` above pins it.
    `ground_answer._same_page` compares with a bare `==`, so carrying corpus's
    '7' into a citation makes `"7" == 7` False and silently drops a correctly
    quoted, correctly cited claim -- the exact failure 3ff1463 fixed.
    """

    def test_chunk_is_the_int_ordinal_not_the_vortex_string(self):
        with fake_corpus([WIKI]):
            hits = Z.vector_retrieve("how do tides work")
        self.assertIsInstance(hits[0]["chunk"], int,
                              "a str chunk makes every citation fail _same_page's ==")
        self.assertEqual(hits[0]["chunk"], 7)

    def test_a_claim_cited_to_a_migrated_hit_still_grounds(self):
        """The end-to-end shape of the trap: if `chunk` arrives as a string this
        claim is dropped although it quotes its own chunk verbatim."""
        with fake_corpus([WIKI]):
            hits = Z.vector_retrieve("how do tides work")
        pages = Z.hits_to_pages(hits)
        claim = {"text": "Spring tides happen at new and full moon.",
                 "span": "Spring tides occur at the new moon and the full moon",
                 "cites": Z.page_ident(hits[0])}
        r = G.ground_answer("how do tides work", [claim], pages)
        v = (r["kept"] + r["dropped"])[0]
        self.assertEqual(v["verdict"], "grounded",
                         "got %r (%s)" % (v["verdict"], v.get("why", "")))

    def test_a_page_that_is_not_an_ordinal_raises_rather_than_degrades(self):
        """House rule, same as corpus._route's half-configured guard: a page that
        cannot be an ordinal would break citations invisibly, so it must shout."""
        with fake_corpus([corpus_hit("zim_x", "Tide", "p.27", "text")]):
            with self.assertRaises(ValueError):
                Z.vector_retrieve("q")


class TestTheShelfKeepsItsScope(unittest.TestCase):

    def test_survivorlibrary_is_not_silently_added_to_the_shelf(self):
        """corpus.SHARDS also ranks 12.9 GB of scanned manuals. zimshelf is the
        librarian over the ZIM archives; widening that is a decision, not a
        side effect of changing how ranking works."""
        self.assertTrue(Z.SHARDS, "zimshelf must name the shards it ranks")
        self.assertFalse([s for s in Z.SHARDS if "survivorlibrary" in s],
                         "survivorlibrary is not on this shelf: %r" % (Z.SHARDS,))

    def test_the_shards_it_names_are_the_ones_it_ranks(self):
        with fake_corpus([WIKI]) as r:
            Z.vector_retrieve("q")
        self.assertEqual(r.call_args.kwargs.get("shards"), Z.SHARDS)


class TestArchiveFilter(unittest.TestCase):
    """The filter is corpus.retrieve's `sources=`, not a wider topk trimmed here.

    `corpus.retrieve` reads each ranked hit's full text, one federation query
    apiece. Over-asking here and discarding afterwards would pay that read for
    every discarded hit -- ~0.03 s each against the zim DB, measured, so ~30 s
    for the 1000 hits a `topk * 200` over-ask produces. The trim has to happen
    between ranking and the text reads, where only corpus.retrieve can see both.
    That the filter really drops other sources is tested against a real two-source
    shard in test_corpus.SourceFilter; here we only pin that it is delegated.
    """

    def test_the_archive_is_passed_down_as_a_source_filter(self):
        with fake_corpus([WIKI]) as r:
            Z.vector_retrieve("q", archive="wikipedia_en_all_nopic_2026-06", topk=5)
        self.assertEqual(r.call_args.kwargs.get("sources"), ["wikipedia_en_all_nopic_2026-06"])

    def test_filtering_does_not_widen_topk_here(self):
        """The over-ask belongs to corpus.retrieve, which trims before reading text."""
        with fake_corpus([WIKI]) as r:
            Z.vector_retrieve("q", archive="wikipedia_en_all_nopic_2026-06", topk=5)
        self.assertEqual(r.call_args.kwargs.get("topk"), 5)

    def test_an_unfiltered_call_sets_no_source_filter(self):
        with fake_corpus([WIKI]) as r:
            Z.vector_retrieve("q", topk=5)
        self.assertIsNone(r.call_args.kwargs.get("sources"))
        self.assertEqual(r.call_args.kwargs.get("topk"), 5)

    def test_n_is_a_dense_display_ordinal(self):
        other = corpus_hit("wikipedia_en_simple_all_nopic_2026-05", "Moon", "0", "The Moon.")
        with fake_corpus([other, WIKI]):
            hits = Z.vector_retrieve("q")
        self.assertEqual([h["n"] for h in hits], [1, 2])


class TestTitles(unittest.TestCase):
    """corpus.retrieve's hit shape carries no title -- it is the grounding seam's
    shape and stays that way -- so the shelf looks one up."""

    class FakeCon:
        def __init__(self, rows): self.rows, self.queries = rows, []
        def execute(self, sql, params=None):
            self.queries.append((sql, params)); return self
        def fetchall(self): return self.rows

    def test_a_found_title_is_used(self):
        with fake_corpus([WIKI], titles={("wikipedia_en_all_nopic_2026-06", "Tide"): "Tide"}):
            hits = Z.vector_retrieve("q")
        self.assertEqual(hits[0]["title"], "Tide")

    def test_a_missing_title_falls_back_to_the_entry_path(self):
        """Unchanged from the retired path: `(t or b.replace('_', ' '))`."""
        with fake_corpus([corpus_hit("zim_x", "Lunar_eclipse", "0", "text")], titles={}):
            hits = Z.vector_retrieve("q")
        self.assertEqual(hits[0]["title"], "Lunar eclipse")

    def test_titles_are_looked_up_in_the_table_corpus_routes_to(self):
        """Routing is corpus._route's job, not a second prefix rule here -- a
        `wikipedia_` prefix rule is exactly the trap corpus.py documents."""
        import corpus as C
        con = self.FakeCon([("Tide", "Tide")])
        with mock.patch.object(C, "_route", return_value="w.pages") as route:
            got = Z._titles(con, [WIKI])
        route.assert_called_once()
        self.assertEqual(got, {("wikipedia_en_all_nopic_2026-06", "Tide"): "Tide"})
        self.assertIn("w.pages", con.queries[0][0])
