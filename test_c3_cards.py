"""The handoff contract, with no device: shape, mapping, and isolation."""
import unittest

import c3_cards

CARD = {"card_id": "k1", "claim": "Drying prevents microbial growth.",
        "span": "Drying removes the moisture", "content_sha256": "abc123"}


def stub(result=None, exc=None):
    def check(span, claim):
        if exc:
            raise exc
        return result
    return check


class TestVerdictMapping(unittest.TestCase):

    def test_supported(self):
        v = c3_cards.verdict_for(CARD, stub({"measured": True, "supported": True, "detail": "both"}))
        self.assertEqual(v["verdict"], "supported")

    def test_not_supported(self):
        v = c3_cards.verdict_for(CARD, stub({"measured": True, "supported": False, "detail": "over-reach"}))
        self.assertEqual(v["verdict"], "not_supported")

    def test_unmeasured_is_its_own_verdict(self):
        """A cut, an outage or a model swap must NOT read as not_supported."""
        v = c3_cards.verdict_for(CARD, stub({"measured": False, "supported": None,
                                             "detail": "ordering A cut at budget"}))
        self.assertEqual(v["verdict"], "unmeasured")
        self.assertIn("cut at budget", v["note"])

    def test_exception_is_unmeasured_not_a_crash(self):
        """One bad card must not take down a batch, and must not become a finding."""
        v = c3_cards.verdict_for(CARD, stub(exc=RuntimeError("device on fire")))
        self.assertEqual(v["verdict"], "unmeasured")
        self.assertIn("device on fire", v["note"])

    def test_sha_is_echoed_untouched(self):
        """We never resolve it. It is grimoire's token, verified on their ingest."""
        v = c3_cards.verdict_for(CARD, stub({"measured": True, "supported": True}))
        self.assertEqual(v["content_sha256"], "abc123")
        self.assertEqual(v["card_id"], "k1")

    def test_missing_span_does_not_reach_the_device(self):
        called = []
        def check(s, c):
            called.append(1)
            return {"measured": True, "supported": True}
        v = c3_cards.verdict_for({"card_id": "k2", "claim": "x"}, check)
        self.assertEqual(v["verdict"], "unmeasured")
        self.assertEqual(called, [], "an incomplete card must not cost an NPU call")

    def test_batch_preserves_order_and_length(self):
        cards = [dict(CARD, card_id="k%d" % i) for i in range(5)]
        res = c3_cards.run(cards, stub({"measured": True, "supported": True}))
        self.assertEqual([r["card_id"] for r in res], ["k0", "k1", "k2", "k3", "k4"])

    def test_accepts_both_envelope_shapes(self):
        self.assertEqual(c3_cards._cards({"cards": [CARD]}), [CARD])
        self.assertEqual(c3_cards._cards([CARD]), [CARD])


if __name__ == "__main__":
    unittest.main()
