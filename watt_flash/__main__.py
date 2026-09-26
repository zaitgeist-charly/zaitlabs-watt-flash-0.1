"""python -m watt_flash '{"state": ..., "questions": {...}}'   (or the JSON on stdin) -> {qid: probabilities}"""
import json
import sys

from . import WattFlash


def main() -> None:
    req = json.loads(sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read())
    req = req.get("input", req)   # the Cloudflare {"model", "input"} envelope is accepted too
    print(json.dumps({q: [round(p, 4) for p in ps] for q, ps in WattFlash().decide(req["state"], req["questions"]).items()}))


if __name__ == "__main__":
    main()
