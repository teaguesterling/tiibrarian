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
