"""python -m watt_flash.serve [--port 8942]: POST /v1/systemone, the Jev request and response format (standard library only)."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Sequence

from . import WattFlash
from .layout import Overflow, option_labels


def answer(q: Dict[str, Any], p: Sequence[float]) -> Dict[str, Any]:
    p = [round(float(v), 4) for v in p]
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[1]}
    probs = dict(zip(option_labels(q), p))
    if q["type"] == "choice":
        return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
    return {"type": "score", "score": round(sum(i * v for i, v in enumerate(p)), 4), "probabilities": probs}


def handler(wf: WattFlash, name: str):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.reply(200, {"status": "ok", "model": name}) if self.path == "/health" else self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/systemone":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                req = req.get("input", req)
                state, questions = req["state"], req["questions"]
                probs = wf.decide(state, questions)
            except Overflow as e:
                return self.reply(413, {"error": "max_tokens_exceeded", "message": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": "invalid_request", "message": str(e)})
            self.reply(200, {"model": name, "answers": {qid: answer(questions[qid], p) for qid, p in probs.items()}})

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8942)
    ap.add_argument("--model", default="", help="local directory or Hugging Face repo id (default: WATT_FLASH_MODEL or zaitlabs/watt-flash-0.1)")
    ap.add_argument("--device", default="")
    a = ap.parse_args()
    wf = WattFlash(a.model, a.device) if a.model else WattFlash(device=a.device)
    print("watt-flash-0.1 on http://%s:%d/v1/systemone" % (a.host, a.port), flush=True)
    HTTPServer((a.host, a.port), handler(wf, "watt-flash-0.1")).serve_forever()


if __name__ == "__main__":
    main()
