"""C3 discipline — hermetic (device call mocked).

    CORDEXA_HOME=~/Projects/cordexa/main python3 -m unittest test_c3

The live behaviour needs the Tiiny; this covers what turns two askings into a
verdict. Two rules, and they pull in opposite directions:

  * agreement is required, and a genuine disagreement IS a measurement
    (`supported: False`) — the model judged and said no;
  * a reply that never became a judgement — cut at the gateway, no ANSWER line,
    device down, or **two different models answering the two askings** — is
    UNMEASURED. `supported` is None and no caller may read it as "not supported".

`call_tiiny` conforms to cordexa's `ask` contract: `-> (reply_text, served_model)`,
raising `DeviceUnavailable` for anything that is not a verdict. The served model
is not decoration — see ServedModelIdentity.
"""
import io
import json
import os
import unittest

import c3

DU = c3.DeviceUnavailable
_REAL_CALL = c3.call_tiiny        # captured before any test replaces it


class C3TestCase(unittest.TestCase):
    """Restores `c3.call_tiiny` after every test.

    The mocks here replace a module global. Without this, a class that drives the
    REAL function only works if it happens to sort before the classes that stub
    it out — which is what "AskContract" was silently relying on.
    """

    def setUp(self):
        c3.call_tiiny = _REAL_CALL

    def tearDown(self):
        c3.call_tiiny = _REAL_CALL


def _mock(*replies, served="Qwen/Qwen3-30B-A3B-Instruct"):
    """Askings in order. A reply may be a string, or (reply, served) to vary the
    model, or an exception instance to raise."""
    it = iter(replies)

    def fake(text, **kw):
        r = next(it)
        if isinstance(r, BaseException):
            raise r
        return r if isinstance(r, tuple) else (r, served)
    return fake


def _cut_error(msg="reply cut off (finish_reason=length) before a complete answer line"):
    e = DU(msg)
    e.cut = {"finish_reason": "length", "usage": None, "served": "Qwen/Qwen3-30B-A3B-Instruct",
             "content": "long reasoning with no answer line", "answer_first": False}
    return e


class DoubleJudge(C3TestCase):
    def test_both_supported_promotes(self):
        c3.call_tiiny = _mock("reasoning...\nANSWER: 1", "reasoning...\nANSWER: 3")
        r = c3.support_check("span", "claim")
        self.assertTrue(r["measured"])
        self.assertTrue(r["supported"])

    def test_orderings_disagree_is_measured_not_supported(self):
        c3.call_tiiny = _mock("ANSWER: 1", "ANSWER: 2")
        r = c3.support_check("span", "claim")
        self.assertTrue(r["measured"])      # they DID answer; the verdict is not-supported
        self.assertFalse(r["supported"])
        self.assertIn("disagree", r["detail"])

    def test_both_not_supported(self):
        c3.call_tiiny = _mock("ANSWER: 2", "ANSWER: 2")
        r = c3.support_check("span", "claim")
        self.assertTrue(r["measured"])
        self.assertFalse(r["supported"])


class NotAVerdict(C3TestCase):
    def test_cut_reply_is_unmeasured(self):
        c3.call_tiiny = _mock(_cut_error(), "irrelevant")
        r = c3.support_check("span", "claim")
        self.assertFalse(r["measured"])
        self.assertIsNone(r["supported"])

    def test_a_cut_records_what_came_back(self):
        # cordexa attaches the content/usage/served to the exception precisely so
        # the boundary that stops can write it down. Dropping it loses the evidence.
        c3.call_tiiny = _mock(_cut_error(), "irrelevant")
        r = c3.support_check("span", "claim")
        self.assertEqual(r["cut"]["finish_reason"], "length")
        self.assertIn("served", r["cut"])

    def test_a_cut_names_which_asking_was_cut(self):
        c3.call_tiiny = _mock("ANSWER: 1", _cut_error())
        r = c3.support_check("span", "claim")
        self.assertEqual(r["ordering"], "B")

    def test_no_answer_line_is_unmeasured_not_unsupported(self):
        """A complete reply that ignored the output format has not judged anything.

        cordexa's `combine` maps an unparseable meaning to (False, "unparseable
        reply"), which is right for the golden set — a record that cannot be read
        does not get promoted. Here `supported: False` is a finding against the
        ANSWER, so a formatting miss must not become one. Same error class as
        reading C1's `abstain` as `unfounded`.
        """
        c3.call_tiiny = _mock("reasoning but the model forgot the ANSWER line", "ANSWER: 3")
        r = c3.support_check("span", "claim")
        self.assertFalse(r["measured"])
        self.assertIsNone(r["supported"])
        self.assertIn("ANSWER", r["detail"])

    def test_outage_is_unmeasured(self):
        c3.call_tiiny = _mock(DU("connection refused"))
        r = c3.support_check("span", "claim")
        self.assertFalse(r["measured"])
        self.assertIsNone(r["supported"])


class ServedModelIdentity(C3TestCase):
    """Two askings answered by two different models are not a double judge.

    Not hypothetical. On an appliance something else can stop and start a model
    mid-run — a watchdog, another session, the device itself — and cordexa's own
    note records the same thing from the other side: "Seen live: HTTP 503 mid-run
    after the device's loaded model was swapped." A verdict compounded from two
    models has, in cordexa's words, "no single identity".
    """

    def test_model_swap_between_askings_is_unmeasured(self):
        c3.call_tiiny = _mock(("ANSWER: 1", "Qwen/Qwen3-30B-A3B-Instruct"),
                              ("ANSWER: 3", "Qwen/Qwen3-8B"))
        r = c3.support_check("span", "claim")
        self.assertFalse(r["measured"], "a two-model verdict must not be reported as measured")
        self.assertIsNone(r["supported"])
        self.assertIn("served model changed", r["detail"])

    def test_a_swap_that_would_have_promoted_is_still_refused(self):
        # Both askings say "supported". Agreement is not enough if the judges differ.
        c3.call_tiiny = _mock(("ANSWER: 1", "model-a"), ("ANSWER: 3", "model-b"))
        self.assertFalse(c3.support_check("span", "claim")["measured"])

    def test_the_served_model_is_reported_on_a_measured_result(self):
        c3.call_tiiny = _mock("ANSWER: 1", "ANSWER: 3", served="Qwen/Qwen3-30B-A3B-Instruct")
        r = c3.support_check("span", "claim")
        self.assertEqual(r["served"], "Qwen/Qwen3-30B-A3B-Instruct")


class Trace(C3TestCase):
    """cordexa's per-asking trace, kept rather than discarded."""

    def test_measured_result_carries_a_trace_for_both_askings(self):
        c3.call_tiiny = _mock("ANSWER: 1", "ANSWER: 3")
        tr = c3.support_check("span", "claim")["trace"]
        self.assertEqual(sorted(tr), ["A", "B"])
        self.assertEqual(tr["A"]["meaning"], "supported")

    def test_v3_commitment_flag_is_computed(self):
        # v3 is reason-then-commit; `committed_early` is how we see it working.
        c3.call_tiiny = _mock("ANSWER: 1\nbecause the span states it", "ANSWER: 3")
        tr = c3.support_check("span", "claim")["trace"]
        self.assertIn("committed_early", tr["A"])


class NeverReAsks(C3TestCase):
    """Retry transport, never an asking. Re-asking until a parseable answer turns
    up selects for whatever the model says on a second try — cordexa's check()
    says "Never re-asks", and warmup is the one exception it allows."""

    def test_a_cut_is_not_retried(self):
        calls = []

        def fake(text, **kw):
            calls.append(text)
            raise _cut_error()
        c3.call_tiiny = fake
        c3.support_check("span", "claim")
        self.assertEqual(len(calls), 1, "a cut reply was re-asked")


class AskContract(C3TestCase):
    """`call_tiiny` itself, against a mocked HTTP layer.

    Every test above mocks `call_tiiny`, so none of them can see the seam where
    the served model is read off the wire — which is exactly where it went
    missing the first time. These drive the real function.
    """

    def setUp(self):
        super().setUp()
        self._real = c3.urllib.request.urlopen
        os.environ["TIINY_AUTH_KEY"] = "test-key"     # never touch the real key file

    def tearDown(self):
        super().tearDown()
        c3.urllib.request.urlopen = self._real
        os.environ.pop("TIINY_AUTH_KEY", None)

    def _serve(self, *responses):
        """Each response is a dict body, or an exception to raise."""
        it = iter(responses)
        self.calls = []

        class Ctx:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return io.BytesIO(json.dumps(self.payload).encode())

            def __exit__(self, *a):
                return False

        def fake(req, timeout=None):
            self.calls.append(req)
            r = next(it)
            if isinstance(r, BaseException):
                raise r
            return Ctx(r)
        c3.urllib.request.urlopen = fake

    @staticmethod
    def _body(content="ANSWER: 1\nbecause it says so.\n", finish="stop", model="served-model-x"):
        return {"model": model, "usage": {"total_tokens": 12},
                "choices": [{"finish_reason": finish, "message": {"content": content}}]}

    def test_returns_reply_and_the_served_model(self):
        self._serve(self._body(model="Qwen/Qwen3-30B-A3B-Instruct"))
        reply, served = c3.call_tiiny("prompt")
        self.assertIn("ANSWER: 1", reply)
        self.assertEqual(served, "Qwen/Qwen3-30B-A3B-Instruct",
                         "the served model must come off the wire, not from MODEL")

    def test_served_model_is_what_answered_not_what_was_asked_for(self):
        # The requested/retrieved rule, one layer down: the device may serve
        # something other than the id we sent.
        self._serve(self._body(model="Qwen/Qwen3-8B"))
        self.assertEqual(c3.call_tiiny("prompt")[1], "Qwen/Qwen3-8B")

    def test_transport_failure_is_retried_then_raised(self):
        self._serve(OSError("connection refused"), OSError("connection refused"),
                    OSError("connection refused"))
        with self.assertRaises(DU):
            c3.call_tiiny("prompt")
        self.assertEqual(len(self.calls), c3.TRIES)

    def test_transport_failure_then_success(self):
        self._serve(OSError("connection reset"), self._body())
        reply, served = c3.call_tiiny("prompt")
        self.assertIn("ANSWER: 1", reply)
        self.assertEqual(len(self.calls), 2)

    def test_a_cut_reply_raises_and_is_not_retried(self):
        # It ARRIVED. Re-asking it would be selecting for a second opinion.
        self._serve(self._body(content="long reasoning, no commitment", finish="length"),
                    self._body())
        with self.assertRaises(DU) as ctx:
            c3.call_tiiny("prompt")
        self.assertEqual(len(self.calls), 1, "a cut asking was re-asked")
        self.assertTrue(hasattr(ctx.exception, "cut"), "the cut must carry its evidence")
        self.assertEqual(ctx.exception.cut["served"], "served-model-x")

    def test_empty_content_is_not_a_verdict(self):
        # Measured on this device: thinking on -> empty content, finish=length.
        self._serve(self._body(content="", finish="length"))
        with self.assertRaises(DU):
            c3.call_tiiny("prompt")


class CheckKept(C3TestCase):
    def test_outage_stays_support_checked_false(self):
        c3.call_tiiny = _mock(DU("connection refused"))
        rec = c3.check_kept([{"claim": "c", "span": "s", "verdict": "grounded"}])[0]
        self.assertFalse(rec["support_checked"])
        self.assertIsNone(rec["supported"])

    def test_cut_stays_support_checked_false(self):
        c3.call_tiiny = _mock(_cut_error(), _cut_error())
        rec = c3.check_kept([{"claim": "c", "span": "s", "verdict": "grounded"}])[0]
        self.assertFalse(rec["support_checked"])
        self.assertIsNone(rec["supported"])

    def test_model_swap_stays_support_checked_false(self):
        c3.call_tiiny = _mock(("ANSWER: 1", "model-a"), ("ANSWER: 3", "model-b"))
        rec = c3.check_kept([{"claim": "c", "span": "s", "verdict": "grounded"}])[0]
        self.assertFalse(rec["support_checked"])
        self.assertIsNone(rec["supported"])

    def test_a_real_verdict_is_recorded(self):
        c3.call_tiiny = _mock("ANSWER: 1", "ANSWER: 3")
        rec = c3.check_kept([{"claim": "c", "span": "s", "verdict": "grounded"}])[0]
        self.assertTrue(rec["support_checked"])
        self.assertTrue(rec["supported"])


if __name__ == "__main__":
    unittest.main()
