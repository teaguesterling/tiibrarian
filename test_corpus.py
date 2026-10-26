"""Device-free tests for corpus.py over the unified int8 substrate.

Builds a tiny int8 Vortex shard + a matching text DB in a tmp dir -- no NPU, no real corpus. Retrieval
ranks on the Vortex, text is read from the federation, grounding runs cordexa's mechanical C1. The only
non-stdlib requirement is CORDEXA_HOME reachable (ground_answer imports C1 by path).

    CORDEXA_HOME=~/Projects/cordexa python3 -m unittest test_corpus -v
"""
import inspect
import os
import tempfile
import unittest

import duckdb

import corpus
import ground_answer as G


def _vec(dims):
    v = [0] * 1024
    for i, val in dims.items():
        v[i] = val
    return v


def _qvec():
    v = [0.0] * 1024
    v[0], v[1] = 1.0, 0.5
    return v


# p1/p2 of b1 run on mid-sentence (exercises the continuity join); b2 is a separate single page whose
# text carries a >=40-char span (C1's minimum); b3 is unrelated.
ROWS = [
    ("survivorlibrary_hastext", "b1", "1", "Boil the water hard for ten", _vec({0: 127, 1: 64})),
    ("survivorlibrary_hastext", "b1", "2", "minutes to make it safe to drink.", _vec({0: 120, 1: 60})),
    ("survivorlibrary_hastext", "b2", "1",
     "A charcoal filter removes sediment and odour from collected rainwater.", _vec({0: 127})),
    ("survivorlibrary_hastext", "b3", "9", "An unrelated page about beekeeping in winter.", _vec({2: 127})),
]
B2_SPAN = "charcoal filter removes sediment and odour from collected"   # >=40 chars, verbatim in b2


class CorpusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        textdb = os.path.join(cls.tmp.name, "text.duckdb")
        shard = os.path.join(cls.tmp.name, "s.vortex")
        b = duckdb.connect(textdb)
        b.execute("INSTALL vortex; LOAD vortex;")
        b.execute("CREATE TABLE pages(source VARCHAR, book VARCHAR, page VARCHAR, text VARCHAR)")
        b.execute("CREATE TABLE _v(source VARCHAR, book VARCHAR, page VARCHAR, "
                  "vec TINYINT[1024], vscale FLOAT)")
        for src, book, page, text, vec in ROWS:
            b.execute("INSERT INTO pages VALUES (?,?,?,?)", [src, book, page, text])
            b.execute("INSERT INTO _v VALUES (?,?,?,?::TINYINT[1024],1.0)", [src, book, page, vec])
        b.execute(f"COPY _v TO '{shard}' (FORMAT vortex)")
        b.close()
        # all fixture sources are survivorlibrary_* -> the sl alias; point both aliases at the one file
        cls.con = corpus.connect(sl_db=textdb, zim_db=textdb)
        cls.shard = shard
        cls.hits = corpus.retrieve(_qvec(), topk=8, con=cls.con, shards=[shard])
        cls.pages = corpus.hits_to_pages(cls.hits, con=cls.con)
        cls.top = cls.hits[0]
        cls.b2 = next(h for h in cls.hits if h["book"] == "b2")

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_retrieve_ranks_and_attaches_text(self):
        self.assertEqual(self.top["book"], "b1")                      # aligned with the query
        self.assertTrue(self.top["text"])                             # full text attached from federation
        self.assertEqual([h["score"] for h in self.hits],
                         sorted((h["score"] for h in self.hits), reverse=True))
        self.assertEqual([h["n"] for h in self.hits], list(range(1, len(self.hits) + 1)))

    def test_shortlist_surface_is_gone(self):
        self.assertNotIn("shortlist", inspect.signature(corpus.retrieve).parameters)
        self.assertNotIn("shortlist", inspect.signature(corpus.ask_grounded).parameters)

    def test_page_ident_is_discriminating(self):
        ident = corpus.page_ident(self.top)
        self.assertEqual(set(ident), {"kind", "source", "doc", "page"})
        self.assertEqual(ident["source"], "survivorlibrary_hastext")   # lane is in the identity

    def test_locator_names_the_page(self):
        self.assertIn("b1", corpus.locator_str(self.top))

    def test_continuity_join_merges_running_pages(self):
        joined = [p for p in self.pages if p["text"].startswith("Boil the water")]
        self.assertTrue(joined, "b1 page should be present")
        self.assertIn("minutes to make it safe", joined[0]["text"])       # p1 joined with p2
        self.assertEqual(joined[0]["retrieved"].get("page_to"), "2")

    def test_real_span_correctly_cited_is_grounded(self):
        r = G.ground_answer("q", [{"text": "c", "span": B2_SPAN,
                                   "cites": corpus.page_ident(self.b2)}], self.pages)
        self.assertTrue(r["grounded"], r)
        self.assertEqual(r["kept"][0]["citation_matched"], 1)
        self.assertFalse(r["kept"][0]["support_checked"])   # grounded != supported

    def test_span_cited_to_a_different_page_is_misattributed(self):
        r = G.ground_answer("q", [{"text": "c", "span": B2_SPAN,
                                   "cites": corpus.page_ident(self.top)}], self.pages)
        self.assertTrue(r["abstain"], r)
        self.assertEqual(r["dropped"][0]["verdict"], "misattributed")

    def test_confabulated_span_abstains(self):
        r = G.ground_answer("q", [{"text": "c",
                                   "span": "this exact sentence appears on no page in the corpus at all",
                                   "cites": corpus.page_ident(self.top)}], self.pages)
        self.assertTrue(r["abstain"], r)
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")

    def test_ask_grounded_end_to_end_with_supplied_vector(self):
        r, hits = corpus.ask_grounded("how is it purified?", corpus.quote_top_hit,
                                      qvec=_qvec(), con=self.con, topk=5, shards=[self.shard])
        self.assertTrue(hits)
        self.assertTrue(r["grounded"], r)


class EmbedQueryGuards(unittest.TestCase):
    def test_embed_query_without_token_raises_rather_than_falling_back(self):
        saved = corpus.AUTH
        corpus.AUTH = ""
        try:
            with self.assertRaises(RuntimeError) as e:
                corpus.embed_query("anything")
            self.assertIn("TIINY_AUTH_KEY", str(e.exception))
        finally:
            corpus.AUTH = saved

    def test_retrieve_rejects_a_wrong_width_vector(self):
        with self.assertRaises(ValueError):
            corpus.retrieve([0.0] * 512)


if __name__ == "__main__":
    unittest.main()


class SourceFilter(unittest.TestCase):
    """Filtering by source must happen BEFORE the text is attached.

    `retrieve` reads each ranked hit's full text from the federation, one query
    per hit. A caller that wants one source out of many must therefore not be
    made to over-ask and then discard: zimshelf's archive filter asks for
    `topk * 200`, and at ~0.03 s per text read -- measured against the real zim
    DB, which carries no index on (source, book, page) -- a filtered call would
    spend ~30 s reading text it is about to throw away.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        textdb = os.path.join(cls.tmp.name, "t.duckdb")
        cls.shard = os.path.join(cls.tmp.name, "s.vortex")
        b = duckdb.connect(textdb)
        b.execute("INSTALL vortex; LOAD vortex;")
        b.execute("CREATE TABLE pages(source VARCHAR, book VARCHAR, page VARCHAR, text VARCHAR)")
        b.execute("CREATE TABLE _v(source VARCHAR, book VARCHAR, page VARCHAR, "
                  "vec TINYINT[1024], vscale FLOAT)")
        # Two sources, interleaved by score, so a topk that ignores the filter returns both.
        for i in range(6):
            src = "zim_wanted" if i % 2 else "zim_other"
            b.execute("INSERT INTO pages VALUES (?,?,?,?)",
                      [src, f"b{i}", "0", f"Page {i} of {src}, long enough to be a real passage."])
            b.execute("INSERT INTO _v VALUES (?,?,?,?::TINYINT[1024],1.0)",
                      [src, f"b{i}", "0", _vec({0: 127 - i})])
        b.execute(f"COPY _v TO '{cls.shard}' (FORMAT vortex)")
        b.close()
        cls.con = corpus.connect(sl_db=textdb, zim_db=textdb)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def test_unfiltered_still_returns_every_source(self):
        hits = corpus.retrieve(_qvec(), topk=6, con=self.con, shards=[self.shard])
        self.assertEqual({h["source"] for h in hits}, {"zim_wanted", "zim_other"})

    def test_sources_keeps_only_what_was_asked_for(self):
        hits = corpus.retrieve(_qvec(), topk=3, con=self.con, shards=[self.shard],
                               sources=["zim_wanted"])
        self.assertTrue(hits, "the filter must not empty the result")
        self.assertEqual({h["source"] for h in hits}, {"zim_wanted"})

    def test_text_is_not_read_for_hits_the_filter_discards(self):
        """THE point of the parameter. Without it the caller over-asks and pays a
        federation query per discarded hit."""
        import unittest.mock as mock
        with mock.patch.object(corpus, "_text", wraps=corpus._text) as t:
            hits = corpus.retrieve(_qvec(), topk=3, con=self.con, shards=[self.shard],
                                   sources=["zim_wanted"])
        self.assertEqual(t.call_count, len(hits),
                         "text was read for hits that were filtered out")

    def test_n_is_renumbered_after_filtering(self):
        hits = corpus.retrieve(_qvec(), topk=3, con=self.con, shards=[self.shard],
                               sources=["zim_wanted"])
        self.assertEqual([h["n"] for h in hits], list(range(1, len(hits) + 1)))
