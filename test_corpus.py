"""Corpus retrieval + grounding against the real vector index.

NOT hermetic, and skipped unless the index is present -- it needs a ~16 GB DuckDB
built from NPU embeddings. It needs no device and no network: queries are seeded from
a page's own vector, so the NPU is only required to embed a typed question.

    CORDEXA_HOME=~/Projects/cordexa python3 -m unittest test_corpus -v
    TIIBRARIAN_CORPUS=/path/to/corpus.duckdb python3 -m unittest test_corpus
"""
import os
import unittest
from collections import Counter

import corpus
import ground_answer as G

HAVE = os.path.exists(corpus.CORPUS)


@unittest.skipUnless(HAVE, f"corpus index not present at {corpus.CORPUS}")
class CorpusRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One connection for the class: the first query on a cold ~16 GB file takes
        # ~10 s and every one after it ~170 ms. Reconnecting per test would pay that
        # cold cost repeatedly and say nothing extra.
        cls.con = corpus.connect()
        cls.con.execute("LOAD vss")
        seed = cls.con.execute(
            "SELECT book, page, text, vec FROM pages "
            "WHERE length(text) BETWEEN 1200 AND 4000 LIMIT 1").fetchone()
        cls.book, cls.page, cls.text, cls.qvec = seed
        cls.hits = corpus.retrieve(cls.qvec, con=cls.con, topk=8, shortlist=200)
        cls.pages = corpus.hits_to_pages(cls.hits)
        cls.top = cls.hits[0]
        cls.span = " ".join(cls.top["text"].split()[:12])

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def test_retrieve_returns_ranked_hits(self):
        self.assertTrue(self.hits)
        scores = [h["score"] for h in self.hits]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for h in self.hits:
            self.assertTrue(h["text"])          # text is present for grounding
            self.assertIsInstance(h["page"], int)

    def test_seed_page_is_its_own_nearest_neighbour(self):
        # A page queried by its own vector must come back first. If this fails the
        # two-stage query is wrong -- the coarse and exact halves disagree.
        self.assertEqual((self.top["book"], self.top["page"]), (self.book, self.page))
        self.assertGreater(self.top["score"], 0.99)

    def test_page_ident_is_discriminating(self):
        # It must identify ONE page, not a family. book alone is not enough.
        ident = corpus.page_ident(self.top)
        self.assertEqual(set(ident), {"kind", "book", "page"})
        matched = [p for p in self.pages if p["retrieved"] == ident]
        self.assertEqual(len(matched), 1)

    def test_real_span_correctly_cited_is_grounded(self):
        r = G.ground_answer("q", [{"text": "c", "span": self.span,
                                   "cites": corpus.page_ident(self.top)}], self.pages)
        self.assertTrue(r["grounded"])
        self.assertEqual(r["kept"][0]["citation_matched"], 1)
        self.assertFalse(r["kept"][0]["support_checked"])   # grounded != supported

    def test_span_cited_to_a_different_page_is_misattributed(self):
        other = next(h for h in self.hits[1:]
                     if (h["book"], h["page"]) != (self.top["book"], self.top["page"]))
        r = G.ground_answer("q", [{"text": "c", "span": self.span,
                                   "cites": corpus.page_ident(other)}], self.pages)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "misattributed")

    def test_ambiguous_book_only_citation_is_refused(self):
        # An under-specified citation that matches several retrieved pages is not
        # attribution. Uses a book that really does appear twice in the hit set when
        # one exists, and constructs the collision otherwise, so the assertion holds
        # regardless of which seed page setUpClass happened to pick.
        dup = [b for b, c in Counter(h["book"] for h in self.hits).items() if c > 1]
        if dup:
            book = dup[0]
            hit = next(h for h in self.hits if h["book"] == book)
            span, pages = " ".join(hit["text"].split()[:12]), self.pages
        else:
            book, span = self.top["book"], self.span
            twin = dict(self.pages[0])
            twin["retrieved"] = {"kind": corpus.KIND, "book": book,
                                 "page": self.top["page"] + 9999}
            twin["requested"] = twin["retrieved"]
            pages = [self.pages[0], twin]
        r = G.ground_answer("q", [{"text": "c", "span": span,
                                   "cites": {"kind": corpus.KIND, "book": book}}], pages)
        self.assertTrue(r["abstain"])
        self.assertGreaterEqual(r["dropped"][0]["citation_matched"], 2)

    def test_confabulated_span_abstains(self):
        r = G.ground_answer("q", [{"text": "c",
                                   "span": "freezing is universally superior for every food",
                                   "cites": corpus.page_ident(self.top)}], self.pages)
        self.assertTrue(r["abstain"])
        self.assertEqual(r["dropped"][0]["verdict"], "unfounded")

    def test_ask_grounded_end_to_end_with_supplied_vector(self):
        # The whole loop with no device: retrieve -> synth -> ground.
        r, hits = corpus.ask_grounded("how is it preserved?", corpus.quote_top_hit,
                                      qvec=self.qvec, con=self.con, topk=5)
        self.assertTrue(hits)
        self.assertTrue(r["grounded"])


class EmbedQueryGuards(unittest.TestCase):
    def test_embed_query_without_token_raises_rather_than_falling_back(self):
        # A wrong-encoder query returns plausible, worse hits and no error, so this
        # must fail loudly instead of substituting anything.
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
