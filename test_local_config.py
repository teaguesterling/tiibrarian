#!/usr/bin/env python3
"""Machine-local path settings: the environment first, then a local file, then nothing.

    python3 -m unittest test_local_config -v

Hermetic. Every test points the loader at a file it made, or at one that does not
exist -- never at the real ~/.config/tiibrarian/paths.env, because on a host that has
one, reading it would hide a missing raise.

WHY THIS EXISTS. The repo is public, and corpus.py and zimshelf.py defaulted to paths
on one host's disk. Those defaults are gone. A host now names its paths in a file
outside the repo, so callers that never set an env var (a systemd unit, another
repo's script, a Slurm job) keep working without each one exporting them.
"""
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import local_config as L

HERE = os.path.dirname(os.path.abspath(__file__))
MISSING = "/nonexistent/tiibrarian/paths.env"


def _file(text):
    f = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False)
    f.write(text)
    f.close()
    return f.name


class TestSetting(unittest.TestCase):

    def test_the_file_supplies_a_value(self):
        p = _file("# a comment\n\nTIIBRARIAN_SL_DB=/data/sl.duckdb\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(L.setting("TIIBRARIAN_SL_DB", path=p), "/data/sl.duckdb")

    def test_the_environment_beats_the_file(self):
        p = _file("TIIBRARIAN_SL_DB=/from/file\n")
        with mock.patch.dict(os.environ, {"TIIBRARIAN_SL_DB": "/from/env"}, clear=True):
            self.assertEqual(L.setting("TIIBRARIAN_SL_DB", path=p), "/from/env")

    def test_neither_gives_the_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(L.setting("TIIBRARIAN_SL_DB", "", path=MISSING), "")

    def test_a_path_list_drops_empty_entries(self):
        """`"".split(":")` is `[""]`, which is truthy -- an unset list must be EMPTY,
        or every guard of the form `if not shards` passes it."""
        p = _file("TIIBRARIAN_ZIM_SHARDS=/a.vortex::/b.vortex:\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(L.paths("TIIBRARIAN_ZIM_SHARDS", path=p), ["/a.vortex", "/b.vortex"])
            self.assertEqual(L.paths("TIIBRARIAN_ZIM_SHARDS", path=MISSING), [])

    def test_require_names_the_setting_and_the_file(self):
        with self.assertRaises(RuntimeError) as e:
            L.require("", "TIIBRARIAN_SL_DB")
        self.assertIn("TIIBRARIAN_SL_DB", str(e.exception))
        self.assertIn("paths.env", str(e.exception))


class TestUnsetPathsRaiseWhereUsed(unittest.TestCase):
    """An unset path must raise at the point of use, naming the setting -- never
    attach or rank something else in its place."""

    def test_connect_without_a_survivorlibrary_db_raises(self):
        import corpus as C
        with mock.patch.object(C, "SL_DB", ""):
            with self.assertRaises(RuntimeError) as e:
                C.connect(zim_db=":memory:")
        self.assertIn("TIIBRARIAN_SL_DB", str(e.exception))

    def test_retrieve_without_shards_raises(self):
        import corpus as C
        with mock.patch.object(C, "SHARDS", []):
            with self.assertRaises(RuntimeError) as e:
                C.retrieve([0.0] * C.DIM, con=object())
        self.assertIn("TIIBRARIAN_VORTEX_SHARDS", str(e.exception))

    def test_the_shelf_without_its_shards_raises_and_does_not_fall_back(self):
        """THE TRAP: zimshelf passes `shards or SHARDS` down, and corpus.retrieve does
        the same with ITS list, which also ranks survivorlibrary. An empty shelf list
        would silently widen the shelf. It must raise before it gets there."""
        import zimshelf as Z
        import corpus as C
        with mock.patch.object(Z, "SHARDS", []), \
             mock.patch.object(C, "embed_query", return_value=[0.0] * C.DIM), \
             mock.patch.object(C, "retrieve") as r:
            with self.assertRaises(RuntimeError) as e:
                Z.vector_retrieve("q", con=object())
        r.assert_not_called()
        self.assertIn("TIIBRARIAN_ZIM_SHARDS", str(e.exception))


class TestImportNeverRaises(unittest.TestCase):

    def test_importing_with_nothing_configured_succeeds(self):
        """Hermetic tests import these modules on hosts with no file and no env."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("TIIBRARIAN_")}
        env["TIIBRARIAN_CONFIG"] = MISSING
        r = subprocess.run([sys.executable, "-c", "import corpus, zimshelf; "
                            "print(corpus.SL_DB == '', corpus.SHARDS == [], zimshelf.SHARDS == [])"],
                           cwd=HERE, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["True", "True", "True"])


if __name__ == "__main__":
    unittest.main()
