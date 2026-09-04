"""Seam test: REAL DuckDB fts retrieval + mock synth + C1 grounding, no device.

    CORDEXA_HOME=~/Projects/cordexa python3 -m unittest test_seam

Builds a tiny fts corpus in a temp dir, so it is hermetic (no network, no device,
and no dependence on any checked-in corpus).
"""
import os
import tempfile
import unittest

import duckdb

import seam


class Seam(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "corpus.duckdb")
        con = duckdb.connect(self.db)
        con.execute("CREATE TABLE passages(id INTEGER, source VARCHAR, title VARCHAR, content VARCHAR)")
        con.executemany("INSERT INTO passages VALUES (?,?,?,?)", [
            (1, "drying.md", "Drying", "Drying removes the moisture from the food so bacteria, "
                                        "yeast and mold cannot grow and spoil the food."),
            (2, "canning.md", "Canning", "Low-acid vegetables must reach a high temperature that only "
                                          "a pressure canner attains to be safe for storage."),
        ])
        con.execute("INSTALL fts; LOAD fts;")
        con.execute("PRAGMA create_fts_index('passages','id','content')")
        con.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_retrieval_then_grounded(self):
        result, hits = seam.ask_grounded("how does drying stop mold", self.db, topk=2,
                                         synth=seam.mock_grounded)
        self.assertTrue(hits, "fts should retrieve something")
        self.assertTrue(result["grounded"])
        self.assertEqual(len(result["kept"]), 1)
        self.assertFalse(result["kept"][0]["support_checked"])   # grounded != supported

    def test_real_retrieval_then_confabulated_abstains(self):
        result, hits = seam.ask_grounded("how does drying stop mold", self.db, topk=2,
                                         synth=seam.mock_confabulated)
        self.assertTrue(hits)
        self.assertTrue(result["abstain"])
        self.assertEqual(result["dropped"][0]["verdict"], "unfounded")


if __name__ == "__main__":
    unittest.main()
