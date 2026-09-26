"""Probabilities match tests/expected.json (reference implementation, CPU fp32)."""
import json
import os

import pytest

from watt_flash import WattFlash

HERE = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="module")
def wf():
    return WattFlash(device="cpu")   # WATT_FLASH_MODEL=<dir> to test local weights


def test_parity(wf):
    ref = json.load(open(os.path.join(HERE, "expected.json")))
    for ex in json.load(open(os.path.join(HERE, "cf_examples.json"))):
        got = wf.decide(ex["input"]["state"], ex["input"]["questions"])
        for qid, p in ref[ex["name"]].items():
            assert got[qid] == pytest.approx(p, abs=1e-4), (ex["name"], qid)


def test_batch_equals_single(wf):
    cf = json.load(open(os.path.join(HERE, "cf_examples.json")))
    reqs = [(e["input"]["state"], e["input"]["questions"]) for e in cf]
    for (probs, _), (s, qs) in zip(wf.decide_batch(reqs), reqs):
        single = wf.decide(s, qs)
        for qid in qs:
            assert probs[qid] == pytest.approx(single[qid], abs=1e-4)
