"""C3 annotation over a grounded answer: it ANNOTATES and never gates.

    CORDEXA_HOME=~/Projects/cordexa/main python3 -m unittest test_support_annotation -v

No device. `annotate_support` takes the checker as an argument precisely so this
file can hand it a stub and assert the properties that matter, which are all
properties about what the layer must NOT do.

Why a gate would be wrong, concretely: C3 on this device returned `supported`
on only 4 of 6 consecutive runs over ONE identical span and claim at
temperature=0 (MEASUREMENT-c3.md, third run), while never once producing a false
accept. A gate would therefore discard genuinely supported claims at the rate
the instability sets, and the answer would get quieter and *look* better — no
failure anywhere to see.
"""
import unittest

import ground_answer as G

PAGE = {
    "requested": {"kind": "zim", "id": "survivorlibrary", "path": "preserving-food-drying.pdf", "page": 1},
    "retrieved": {"kind": "zim", "id": "survivorlibrary", "path": "preserving-food-drying.pdf", "page": 1},
    "resolution": "exact",
    "text": ("Drying removes the moisture from the food so bacteria, yeast and mold cannot "
             "grow and spoil the food. It takes several days to dry foods out-of-doors."),
}
CLAIM = {"text": "Drying removes moisture so that bacteria, yeast and mold cannot grow.",
         "span": "Drying removes the moisture from the food so bacteria, yeast and mold cannot grow",
         "cites": PAGE["requested"]}


def grounded():
    r = G.ground_answer("why dry food?", [CLAIM], [PAGE])
    assert r["grounded"] and len(r["kept"]) == 1, r
    return r


def stub(verdict):
    """A `check_kept` that answers every claim the same way, and records calls."""
    calls = []

    def check_kept(kept):
        calls.append(list(kept))
        out = []
        for k in kept:
            rec = dict(k)
            if verdict is None:
                rec.update(support_checked=False, supported=None,
                           c3={"measured": False, "detail": "device down"})
            else:
                rec.update(support_checked=True, supported=verdict,
                           c3={"measured": True, "supported": verdict})
            out.append(rec)
        return out

    check_kept.calls = calls
    return check_kept


class TestAnnotateSupport(unittest.TestCase):

    def test_unsupported_claim_is_annotated_not_dropped(self):
        """THE property. A `not supported` verdict marks the claim; it never
        removes it, never moves it to `dropped`, and never flips `grounded`."""
        r = G.annotate_support(grounded(), stub(False))
        self.assertEqual(len(r["kept"]), 1, "a not-supported claim must stay kept")
        self.assertEqual(r["dropped"], [], "it must not be moved to dropped")
        self.assertTrue(r["grounded"])
        self.assertFalse(r["abstain"])
        self.assertTrue(r["kept"][0]["support_checked"])
        self.assertIs(r["kept"][0]["supported"], False)

    def test_supported_claim_is_marked(self):
        r = G.annotate_support(grounded(), stub(True))
        self.assertTrue(r["kept"][0]["support_checked"])
        self.assertIs(r["kept"][0]["supported"], True)

    def test_unmeasured_is_not_unsupported(self):
        """Device down / cut / no ANSWER line. `supported` stays None: a caller
        that reads None as False turns an outage into an accusation."""
        r = G.annotate_support(grounded(), stub(None))
        self.assertFalse(r["kept"][0]["support_checked"])
        self.assertIsNone(r["kept"][0]["supported"])
        self.assertTrue(r["grounded"])

    def test_abstention_never_calls_the_device(self):
        """No kept claims means nothing to check. The device is expensive and an
        abstention has no claims to annotate — calling it would be pure NPU cost."""
        bad = dict(CLAIM, span="this span is nowhere on the page")
        r = G.ground_answer("q", [bad], [PAGE])
        self.assertTrue(r["abstain"])
        check = stub(True)
        out = G.annotate_support(r, check)
        self.assertEqual(check.calls, [], "abstention must not reach the checker")
        self.assertTrue(out["abstain"])

    def test_note_stops_claiming_support_is_unchecked(self):
        """The un-annotated note says support is NOT checked here. If annotation
        leaves that text in place the output contradicts its own fields."""
        before = grounded()
        self.assertIn("NOT checked", before["note"])
        after = G.annotate_support(before, stub(True))
        self.assertNotIn("NOT checked", after["note"])

    def test_does_not_mutate_its_input(self):
        """Callers keep the pre-annotation result (zimshelf prints both paths).

        `verify_claim` already stamps `support_checked: False` on every grounded
        claim -- that flag IS the gap C3 fills -- so the input is not checked for
        the key's absence but for the key still reading False, and for no `c3`
        payload having been grafted onto it.
        """
        r = grounded()
        out = G.annotate_support(r, stub(True))
        self.assertIs(r["kept"][0]["support_checked"], False)
        self.assertNotIn("c3", r["kept"][0])
        self.assertIs(out["kept"][0]["support_checked"], True)   # the copy did change


if __name__ == "__main__":
    unittest.main()
