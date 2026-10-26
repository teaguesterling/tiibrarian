"""Text routing when Wikipedia's text lives in a federation DB of its own.

The 14.3M-chunk `wikipedia_en_all_nopic_2026-06` corpus is too large to append
into `zim-corpora.duckdb`, so it gets its own text DB (substrate spec S4). That
makes `_text`'s two-way prefix split a three-way route, and the obvious third
rule -- "route `wikipedia_*` to the wiki DB" -- IS WRONG:

    wikipedia_en_simple_all_nopic_2026-05     326,251 chunks, in the ZIM DB
    wikipedia_en_medicine_maxi_2026-04        214,771 chunks, in the ZIM DB
    wikipedia_en_all_nopic_2026-06         14,342,771 chunks, in the WIKI DB

Both of the first two already share the `wikipedia_` prefix. Routing on it
sends them to a file that does not hold them, `_text` returns None, and
`retrieve` DROPS the hit -- so two corpora would quietly vanish from results
with no error anywhere. These tests encode that: routing must be by the exact
source, and the two older wikis must still resolve against the ZIM DB.

    CORDEXA_HOME=~/Projects/cordexa python3 -m unittest test_corpus_wiki_routing -v
"""
import os
import tempfile
import unittest

import duckdb

import corpus


def _vec(i, val=127):
    v = [0] * 1024
    v[i] = val
    return v


# One row per corpus, each on its own basis vector so ranking is unambiguous.
BIG_WIKI = "wikipedia_en_all_nopic_2026-06"
ROWS = [
    # source                                   book    page  dim  text
    (BIG_WIKI,                                 "Mars",   "0", 0,
     "Mars is the fourth planet from the Sun and carries a thin atmosphere."),
    ("wikipedia_en_simple_all_nopic_2026-05",  "Water",  "0", 1,
     "Water is a liquid that living things need in order to stay alive."),
    ("wikipedia_en_medicine_maxi_2026-04",     "Sepsis", "0", 2,
     "Sepsis is a life-threatening response to infection in the bloodstream."),
    ("survivorlibrary_hastext",                "b1",     "1", 3,
     "Boil the water hard for ten minutes to make it safe to drink again."),
]


def _write(path, rows, shard=None):
    """A federation text DB holding `rows`; optionally also emit the int8 shard."""
    b = duckdb.connect(path)
    b.execute("INSTALL vortex; LOAD vortex;")
    b.execute("CREATE TABLE pages(source VARCHAR, book VARCHAR, page VARCHAR, text VARCHAR)")
    for src, book, page, _dim, text in rows:
        b.execute("INSERT INTO pages VALUES (?,?,?,?)", [src, book, page, text])
    if shard:
        b.execute("CREATE TABLE _v(source VARCHAR, book VARCHAR, page VARCHAR, "
                  "vec TINYINT[1024], vscale FLOAT)")
        for src, book, page, dim, _text in rows:
            b.execute("INSERT INTO _v VALUES (?,?,?,?::TINYINT[1024],1.0)",
                      [src, book, page, _vec(dim)])
        b.execute(f"COPY _v TO '{shard}' (FORMAT vortex)")
    b.close()


class WikiRoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        cls.shard = os.path.join(d, "s.vortex")
        cls.wiki_db = os.path.join(d, "wikipedia.duckdb")
        cls.zim_db = os.path.join(d, "zim.duckdb")
        cls.sl_db = os.path.join(d, "sl.duckdb")
        # The split under test: the big wiki alone in its own DB, the two older
        # wikis in the ZIM DB. The shard carries all four rows, as the real one does.
        _write(cls.wiki_db, [r for r in ROWS if r[0] == BIG_WIKI])
        _write(cls.zim_db, [r for r in ROWS if r[0].startswith("wikipedia_") and r[0] != BIG_WIKI])
        _write(cls.sl_db, [r for r in ROWS if r[0].startswith("survivorlibrary_")])
        _write(os.path.join(d, "all.duckdb"), ROWS, shard=cls.shard)
        cls.con = corpus.connect(sl_db=cls.sl_db, zim_db=cls.zim_db, wiki_db=cls.wiki_db)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.tmp.cleanup()

    def _text_for(self, source, book, page="0"):
        return corpus._text(self.con, source, book, page)

    def test_big_wiki_reads_from_its_own_db(self):
        self.assertIn("fourth planet", self._text_for(BIG_WIKI, "Mars"))

    def test_simple_wiki_still_reads_from_the_zim_db(self):
        # The prefix trap. `wikipedia_en_simple_*` must NOT follow `wikipedia_` to the wiki DB.
        self.assertIn("liquid that living things",
                      self._text_for("wikipedia_en_simple_all_nopic_2026-05", "Water"))

    def test_medicine_wiki_still_reads_from_the_zim_db(self):
        self.assertIn("response to infection",
                      self._text_for("wikipedia_en_medicine_maxi_2026-04", "Sepsis"))

    def test_survivorlibrary_routing_is_unchanged(self):
        self.assertIn("Boil the water",
                      self._text_for("survivorlibrary_hastext", "b1", "1"))

    def test_every_corpus_survives_a_real_retrieve(self):
        """A mis-routed source has no text, and a hit with no text is dropped silently."""
        for source, book, _page, dim, _t in ROWS:
            hits = corpus.retrieve([float(x) for x in _vec(dim, 1)], topk=4,
                                   con=self.con, shards=[self.shard])
            got = [(h["source"], h["book"]) for h in hits]
            self.assertIn((source, book), got,
                          f"{source} vanished from results -- text lookup missed, hit dropped")

    def test_wiki_db_is_optional(self):
        """Without a wiki DB the module must behave exactly as it did before."""
        con = corpus.connect(sl_db=self.sl_db, zim_db=self.zim_db)
        try:
            self.assertIn("liquid that living things",
                          corpus._text(con, "wikipedia_en_simple_all_nopic_2026-05", "Water", "0"))
            self.assertIsNone(corpus._text(con, BIG_WIKI, "Mars", "0"))
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
