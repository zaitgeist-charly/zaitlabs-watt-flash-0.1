"""watt-flash-0.1: typed questions (noul, choice, score) about a state -> option probabilities, one forward pass."""
import os
from typing import Any, Dict, List, Sequence, Tuple

import torch

from .calibrate import probabilities
from .layout import Layout, Overflow, collate, option_labels  # noqa: F401
from .model import WattModel

MODEL = os.environ.get("WATT_FLASH_MODEL", "zaitlabs/watt-flash-0.1")   # Hugging Face repo id or local directory


class WattFlash:
    def __init__(self, path: str = MODEL, device: str = "", revision: str = ""):
        """`path`: a local model directory, or a Hugging Face repo id (data files only are fetched; no code from the Hub)."""
        from transformers import AutoTokenizer

        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            path = snapshot_download(path, revision=revision or None,
                                     allow_patterns=["*.json", "*.safetensors", "tokenizer/*"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = WattModel.load(path, str(self.device))
        if self.device.type != "cuda":
            self.model.float()
        self.tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        self.tok.model_max_length = 1 << 30
        c = self.model.cfg
        self.layout = Layout(self.tok, c.max_len, c.max_instruction_tokens, c.max_option_tokens, c.marker)

    @torch.inference_mode()
    def decide_batch(self, requests: Sequence[Tuple[Any, Dict[str, Dict[str, Any]]]]) -> List[Tuple[Dict[str, List[float]], int]]:
        """[(state, questions)] -> [({qid: probabilities in option_labels order}, tokens)]. Overflow beyond max_len (8,192)."""
        packed = [self.layout.pack_request(s, qs) for s, qs in requests]
        b = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in collate(packed, self.layout.pad).items()}
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            z = self.model(b).float().cpu().numpy()
        out, r = [], 0
        for p, (_, qs) in zip(packed, requests):
            probs = {}
            for qid, m, t in zip(qs, p.markers, p.qtypes):
                probs[qid] = probabilities(z[r, :len(m)], t, self.model.cfg.temperature).tolist()
                r += 1
            out.append((probs, len(p)))
        return out

    def decide(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, List[float]]:
        return self.decide_batch([(state, questions)])[0][0]
