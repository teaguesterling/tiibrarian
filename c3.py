"""C3 support over answer claims — closing tiibrarian's `support_checked` gap.

C1 (ground_answer) proves a span is verbatim on the page it cites. It cannot see
over-reach: "sun drying" cited to a page that says "drying" passes C1 and is
wrong. C3 is the check that catches it — *does the cited span, on its own, state
the claim?* — and it needs the device.

This calls cordexa's own entry point — `c3_support.check(span, claim, ask=...)` —
and replaces ONLY the device call, which is the one thing that genuinely differs:
cordexa routes through a local proxy (`Bearer local`), tiibrarian talks to the
Tiiny directly. THE MODEL IS THE TIINY NPU — no frontier model.

It did not always. The first version rebuilt `check()`'s loop out of its parts
(`prompt`/`parse_answer`/`meaning`/`combine`), which was behaviourally right on
the v3 prompt and quietly dropped four things that live in `check()` and nowhere
else:

  * **the served-model identity check.** `check()` refuses a verdict whose two
    askings were answered by different model strings — "the run has no single
    identity". Not hypothetical: an appliance holds a bounded set of models and
    something else — a watchdog, another session, the device itself — can stop
    and start one mid-run, and cordexa's own note records the same event from
    the other side ("Seen live: HTTP 503 mid-run after the device's loaded model
    was swapped"). The old `ask` returned `(reply, cut)` and discarded the model
    entirely, so a two-model verdict was reported as measured.
  * **`extract_reply`'s not-a-verdict rules**, which are stricter than the
    `finish_reason == "length"` test they replaced: for v3, anything other than
    `stop` has no commitment, whatever came before it.
  * **the trace flags** — `elaboration_cut`, `committed_early` — which are how
    anyone can tell whether reason-then-commit is working at all.
  * **version-driven ordering selection.** `ORDERINGS` and the choice between
    `combine` and `combine_v2` (three askings, plus a positional-bias detector)
    are read from the prompt file at load. Hardcoding two orderings meant a
    cordexa prompt upgrade would silently keep running the old shape.

The lesson is the same one C1.8 taught in ground_answer.py on the same day: a
guard that lives in a function you bypass does not run, and nothing reports that
it did not.

WHAT THIS LAYER STILL OVERRIDES, deliberately: an unparseable-but-COMPLETE reply.
cordexa's `combine` maps it to (False, "unparseable reply"), which is right for
the golden set — a record that cannot be read does not get promoted. Here
`supported: False` reads as a finding against the answer, so a model that ignored
the output format must not produce one. Those stay UNMEASURED, like a cut.

DEVICE BUDGET (matches synth.py): the gateway cuts a request at ~60s with HTTP 504
and no body. Config is env-driven with the SAME names synth uses, so the two never
drift. Retries cover transport only — never an asking; see `call_tiiny`.
"""
import glob
import importlib.util
import json
import os
import urllib.request

import cordexa_home

# Probed on the file THIS module loads, not on c1_span.py: a cordexa checkout
# could in principle carry one reference module and not the other.
CORDEXA = cordexa_home.find(os.path.join("reference", "c3_support.py"))
os.environ.setdefault("CORDEXA_PROMPT", "v3")           # reason-then-commit

# Device config — same env names as synth.py so the two never drift.
# Default is the documented vhost, same as synth.py. Set TIINY_CHAT_URL to a
# direct http://<address>:8800/... when the bridge is not in play.
DEVICE = os.environ.get("TIINY_CHAT_URL", "http://api.tiiny/v1/chat/completions")
# synth defaults to the small Qwen3-8B (least NPU cost); C3 needs the entailment
# competence and 8B is currently broken on this device, so C3 defaults to the
# validated 30B-A3B-Instruct. Env-overridable, and never `...-A3B` (no such id).
MODEL = os.environ.get("TIINY_CHAT_MODEL", "Qwen/Qwen3-30B-A3B-Instruct")
# v3 reason-then-commit needs the trace to reach the ANSWER line; kept generous.
# The gateway caps wall-clock at ~60s regardless, so TIMEOUT sits just past it to
# observe the cut rather than hang. A cut is unmeasured (below), not a verdict.
MAX_TOKENS = int(os.environ.get("TIIBRARIAN_C3_MAX_TOKENS", "1024"))
TIMEOUT = int(os.environ.get("TIIBRARIAN_C3_TIMEOUT", "65"))
# Transport retries only. An asking that came back is never re-asked.
TRIES = int(os.environ.get("TIIBRARIAN_C3_TRIES", "3"))


def _token():
    tok = os.environ.get("TIINY_AUTH_KEY")
    if tok:
        return tok
    files = sorted(glob.glob(os.path.expanduser("~/.local/share/tiiny-pcsvr/auth_data/*.json")))
    if not files:
        raise _c3().DeviceUnavailable("no TIINY_AUTH_KEY and no local auth_data key file")
    return json.load(open(files[0]))["auth_key"]


def _load_c3():
    path = os.path.join(CORDEXA, "reference", "c3_support.py")
    spec = importlib.util.spec_from_file_location("cordexa_c3_support", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_C3 = None


def _c3():
    global _C3
    if _C3 is None:
        _C3 = _load_c3()
    return _C3


def __getattr__(name):
    """`c3.DeviceUnavailable` IS cordexa's class, resolved on first access.

    There must be exactly one. `check()` re-raises `DeviceUnavailable` untouched
    and wraps every other exception in a bare one — so a second, local class here
    would be caught by that wrapper and would lose the `.cut` payload and the
    `.ordering` tag that say which asking stopped and what came back. Resolved
    lazily so this module still imports with no cordexa on disk.
    """
    if name == "DeviceUnavailable":
        return _c3().DeviceUnavailable
    raise AttributeError(name)


def call_tiiny(text, tries=None, timeout=None):
    """One asking, under cordexa's `ask` contract: -> (reply_text, served_model).

    The served model is the point of the second element. The device reports what
    actually answered, which is not always what was requested — the requested vs
    retrieved rule again, one layer down — and `check()` refuses a verdict whose
    two askings disagree about it.

    Retries TRANSPORT failures only. A reply that ARRIVED and is not a verdict is
    never re-asked: `check()`'s contract is "Never re-asks", and asking again
    until something parseable comes back selects for whatever the model says on
    the second try. `extract_reply` is therefore called outside the retry loop,
    and the DeviceUnavailable it raises for a cut carries the evidence out.
    """
    c3 = _c3()
    tries = TRIES if tries is None else tries
    timeout = TIMEOUT if timeout is None else timeout
    body = json.dumps({
        "model": MODEL, "temperature": 0, "max_tokens": MAX_TOKENS,
        "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": text}],
    }).encode()
    last = None
    for _ in range(tries):
        try:
            req = urllib.request.Request(DEVICE, data=body, headers={
                "Authorization": f"Bearer {_token()}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
        except (OSError, ValueError) as e:   # transport, incl. the bodiless 504
            last = e
            continue
        return c3.extract_reply(out)         # raises for a non-verdict — NOT retried
    raise c3.DeviceUnavailable(f"{DEVICE} model {MODEL}: {last}")


def _unmeasured(detail, **extra):
    """No verdict was reached. `supported` is None and stays None — a caller that
    reads it as "not supported" turns an outage, a gateway cut or a model swap
    into an accusation against the answer."""
    out = {"measured": False, "supported": None, "a": None, "b": None,
           "detail": detail, "trace": None, "served": None, "cut": None,
           "ordering": None}
    out.update(extra)
    return out


def support_check(span, claim):
    """cordexa's double judge, run through cordexa's own `check()`.

    -> {measured, supported, a, b, detail, trace, served, cut, ordering}.
    `measured` is False whenever no verdict was reached — device down, reply cut
    at the budget, no ANSWER line, or the two askings served by different models.
    """
    c3 = _c3()
    try:
        promote, reason, trace = c3.check(span, claim, ask=lambda t: call_tiiny(t))
    except c3.DeviceUnavailable as e:
        o = getattr(e, "ordering", None)
        cut = getattr(e, "cut", None)
        if cut is not None:
            return _unmeasured(f"ordering {o} cut at budget — unmeasured: {e}",
                               cut=cut, ordering=o)
        # Everything else: transport, a malformed reply, or check()'s own refusal
        # when the served model changed between askings. Its message says which.
        return _unmeasured(f"unmeasured: {e}", ordering=o)

    # A COMPLETE reply that carried no ANSWER line parses to None, and `combine`
    # reads that as "unparseable reply" -> not supported. Correct for the golden
    # set, wrong here: see the module docstring. Unmeasured, and retryable.
    blank = [o for o in sorted(trace) if trace[o].get("option") is None]
    if blank:
        return _unmeasured("ordering %s produced no ANSWER line — unmeasured"
                           % ", ".join(blank), trace=trace)

    served = {trace[o].get("served_by") for o in trace}
    return {"measured": True, "supported": bool(promote),
            "a": trace["A"]["meaning"], "b": trace["B"]["meaning"],
            "detail": reason, "trace": trace,
            "served": served.pop() if len(served) == 1 else sorted(map(str, served)),
            "cut": None, "ordering": None}


def check_kept(kept):
    """Annotate ground_answer's grounded claims with C3 support. A claim whose
    support cannot be measured (device down OR a cut/unparseable reply) stays
    support_checked False — never silently 'not supported'."""
    out = []
    for k in kept:
        rec = dict(k)
        try:
            s = support_check(k["span"], k["claim"])
            if s["measured"]:
                rec.update(support_checked=True, supported=s["supported"], c3=s)
            else:
                rec.update(support_checked=False, supported=None, c3=s)
        except _c3().DeviceUnavailable as e:
            rec.update(support_checked=False, supported=None, c3={"unmeasured": str(e)})
        out.append(rec)
    return out
